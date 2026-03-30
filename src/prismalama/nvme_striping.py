"""
HC-52: Multi-NVME striping support for Prismalama weight streaming.

Provides parallel reads across striped NVME drives to saturate weight streaming
throughput on workstations with 4+ NVME drives (e.g. 4x Samsung 990 Pro in RAID0).

Usage:
  AIRLLM_NVME_MOUNTS=/mnt/nvme0:/mnt/nvme1:/mnt/nvme2:/mnt/nvme3
  AIRLLM_SHARD_MOUNTS=/path/to/shard0:/mnt/nvme0;/path/to/shard1:/mnt/nvme1

Striping is opt-in — single NVME path works without these vars.
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Public helpers (called by airllm_runner.py before AutoModel import)
# ─────────────────────────────────────────────────────────────────────────────


def discover_nvme_mounts() -> List[str]:
    """
    Read AIRLLM_NVME_MOUNTS env var (colon-separated) and validate each mount.

    Returns:
        List of valid, accessible NVME mount paths.
        Empty list if AIRLLM_NVME_MOUNTS is not set or all mounts are invalid.
    """
    raw = os.environ.get("AIRLLM_NVME_MOUNTS", "")
    if not raw:
        logger.debug("AIRLLM_NVME_MOUNTS not set — single-NVME mode")
        return []

    mounts = [p.strip() for p in raw.split(":") if p.strip()]
    if not mounts:
        logger.warning("AIRLLM_NVME_MOUNTS is set but empty — single-NVME mode")
        return []

    valid = []
    for mount in mounts:
        if not os.path.exists(mount):
            logger.warning("NVME mount does not exist (skipping): %s", mount)
            continue
        if not os.access(mount, os.R_OK):
            logger.warning("NVME mount not accessible (skipping): %s", mount)
            continue
        try:
            total, used, free = shutil.disk_usage(mount)
            logger.info(
                "NVME mount validated: %s  total=%.1f GB  free=%.1f GB",
                mount, total / (1024**3), free / (1024**3),
            )
            valid.append(mount)
        except Exception as ex:
            logger.warning("Failed to query disk_usage for %s (skipping): %s", mount, ex)

    if not valid:
        logger.warning(
            "AIRLLM_NVME_MOUNTS=%s — no valid mounts found; falling back to single-NVME",
            raw,
        )
        return []

    logger.info("HC-52: discovered %d valid NVME mount(s): %s", len(valid), valid)
    return valid


def _build_shard_to_nvme_map(model_path: str) -> Dict[str, str]:
    """
    Build a mapping from shard file path → NVME mount path.

    Sources (in priority order):
      1. AIRLLM_SHARD_MOUNTS env var (semicolon-separated, format:
         "/path/shard0:/nvme0;/path/shard1:/nvme1")
      2. Shard manifest at ``{model_path}/.airllm_shard_mounts.json``
      3. Inferred: if shard path is under a known NVME mount subdirectory
    """
    mapping: Dict[str, str] = {}

    # ── 1. AIRLLM_SHARD_MOUNTS env var ─────────────────────────────────────
    raw_shards = os.environ.get("AIRLLM_SHARD_MOUNTS", "")
    if raw_shards:
        parts = [p.strip() for p in raw_shards.split(";") if p.strip()]
        i = 0
        while i < len(parts) - 1:
            shard_path = parts[i]
            nvme_mount = parts[i + 1]
            if not os.path.isabs(shard_path):
                logger.warning(
                    "AIRLLM_SHARD_MOUNTS: shard path is not absolute, skipping: %s",
                    shard_path,
                )
                i += 1
                continue
            mapping[os.path.abspath(shard_path)] = nvme_mount
            logger.debug("Shard mount mapped (env): %s → %s", shard_path, nvme_mount)
            i += 2

        if mapping:
            logger.info(
                "HC-52: built shard→mount map from AIRLLM_SHARD_MOUNTS: %d entries",
                len(mapping),
            )
            return mapping

    # ── 2. Shard manifest JSON ──────────────────────────────────────────────
    manifest_path = Path(model_path) / ".airllm_shard_mounts.json"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            if isinstance(manifest, dict):
                for shard_path, nvme_mount in manifest.items():
                    mapping[os.path.abspath(str(shard_path))] = str(nvme_mount)
            logger.info(
                "HC-52: loaded shard→mount map from manifest: %d entries",
                len(mapping),
            )
        except Exception as ex:
            logger.warning("Failed to load shard manifest %s: %s", manifest_path, ex)

    if mapping:
        return mapping

    # ── 3. Infer from NVME mount subdirectories ──────────────────────────────
    nvme_mounts = discover_nvme_mounts()
    if not nvme_mounts:
        return mapping

    model_p = Path(model_path)
    for mount in nvme_mounts:
        mount_path = Path(mount)
        try:
            for candidate in model_p.rglob("*.safetensors"):
                try:
                    rel = candidate.relative_to(mount_path)
                    mapping[str(candidate.resolve())] = mount
                    logger.debug("Inferred mount: %s → %s", candidate, mount)
                except ValueError:
                    pass
        except Exception as ex:
            logger.debug("Error walking model path for inference: %s", ex)

    if mapping:
        logger.info(
            "HC-52: inferred %d shard→mount mappings from mount subdirectories",
            len(mapping),
        )
    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# Parallel read test
# ─────────────────────────────────────────────────────────────────────────────

def _parallel_read_test(nvme_mounts: List[str]) -> Dict[str, float]:
    """
    Run a small concurrent read benchmark across multiple NVME mounts.

    Reads a 128 MB test buffer from each mount concurrently and logs
    measured throughput per mount and aggregate throughput.
    """
    if len(nvme_mounts) < 2:
        return {}

    test_size_mb = 128
    test_size_bytes = test_size_mb * 1024 * 1024
    test_buffer = io.BytesIO(b"\x00" * test_size_bytes)

    def _read_from_mount(mount: str) -> Tuple[str, float, Optional[str]]:
        test_file = os.path.join(mount, f".airllm_read_test_{os.getpid()}.tmp")
        try:
            with open(test_file, "wb") as f:
                f.write(test_buffer.getvalue())
            test_buffer.seek(0)
            start = time.perf_counter()
            with open(test_file, "rb") as f:
                _ = f.read(test_size_bytes)
            elapsed = time.perf_counter() - start
            throughput = test_size_mb / elapsed if elapsed > 0 else 0
            return mount, throughput, None
        except Exception as ex:
            return mount, 0.0, str(ex)
        finally:
            try:
                os.remove(test_file)
            except Exception:
                pass

    logger.info(
        "HC-52: running parallel read benchmark — %d mounts, %d MB per mount",
        len(nvme_mounts), test_size_mb,
    )

    results: Dict[str, float] = {}
    try:
        with ThreadPoolExecutor(max_workers=len(nvme_mounts)) as executor:
            futures = {
                executor.submit(_read_from_mount, mount): mount
                for mount in nvme_mounts
            }
            for future in as_completed(futures):
                mount, throughput, error = future.result()
                if error:
                    logger.warning("HC-52: read test failed for %s: %s", mount, error)
                else:
                    results[mount] = throughput
                    logger.info("HC-52: mount %s read throughput: %.1f MB/s", mount, throughput)
    except Exception as ex:
        logger.warning("HC-52: parallel read benchmark threw: %s", ex)
        return {}

    if results:
        aggregate = sum(results.values())
        logger.info(
            "HC-52: aggregate read throughput: %.1f MB/s across %d mount(s)",
            aggregate, len(results),
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Striping-aware model persister
# ─────────────────────────────────────────────────────────────────────────────

class NVMEStripingModelPersister:
    """
    Wraps a SafetensorModelPersister with multi-NVME parallel reads.

    Architecture:
      - Each NVME mount gets its own file handle kept open for inference duration
      - Reads for different NVME mounts issued concurrently via ThreadPoolExecutor
      - Read errors on one mount do NOT affect shards on other mounts
      - Single NVME fallback when AIRLLM_NVME_MOUNTS has < 2 mounts

    Usage (called by airllm_runner.py before AutoModel.from_pretrained):
      from prismalama.nvme_striping import inject_striped_persister
      inject_striped_persister(model_path)
    """

    def __init__(
        self,
        wrapped,  # SafetensorModelPersister
        shard_to_mount: Optional[Dict[str, str]] = None,
        max_workers: int = 4,
    ):
        self._wrapped = wrapped
        self._shard_to_mount: Dict[str, str] = shard_to_mount or {}
        self._max_workers = max_workers
        self._mount_handles: Dict[str, io.BufferedReader] = {}
        self._executor: Optional[ThreadPoolExecutor] = None
        self._nvme_mounts: List[str] = []
        self._initialized = False

    def _lazy_init(self):
        """Initialise NVME subsystem lazily on first read."""
        if self._initialized:
            return
        self._initialized = True

        self._nvme_mounts = discover_nvme_mounts()
        if len(self._nvme_mounts) >= 2:
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
            _parallel_read_test(self._nvme_mounts)
            logger.info(
                "HC-52: NVME striping active — %d mounts, max_workers=%d",
                len(self._nvme_mounts), self._max_workers,
            )
        else:
            logger.info(
                "HC-52: single NVME mount — striping disabled, using direct reads",
            )

    def _get_mount_handle(self, mount: str, file_path: str) -> io.BufferedReader:
        """Return an open BufferedReader for file_path on the given mount."""
        key = f"{mount}:{file_path}"
        if key not in self._mount_handles:
            self._close_mount_handles_for(mount)
            try:
                fd = os.open(file_path, os.O_RDONLY | os.O_SYNC)
                self._mount_handles[key] = open(fd, "rb", buffering=128 * 1024)
            except OSError as ex:
                logger.error(
                    "HC-52: failed to open %s on %s: %s — falling back to direct",
                    file_path, mount, ex,
                )
                return open(file_path, "rb")  # type: ignore[return-value]
        return self._mount_handles[key]

    def _close_mount_handles_for(self, mount: str):
        """Close all open handles for a specific mount."""
        for key in [k for k in self._mount_handles if k.startswith(f"{mount}:")]:
            try:
                self._mount_handles[key].close()
            except Exception:
                pass
            del self._mount_handles[key]

    def close_all_handles(self):
        """Close all open file handles. Call at inference teardown."""
        for f in self._mount_handles.values():
            try:
                f.close()
            except Exception:
                pass
        self._mount_handles.clear()
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None
        self._initialized = False

    def _striped_read(self, layer_name: str, checkpoint_path: str):
        """Read a shard file, routing to the appropriate NVME mount."""
        from safetensors.torch import load_file

        shard_path = str(Path(checkpoint_path) / (layer_name + ".safetensors"))
        self._lazy_init()

        if not self._nvme_mounts or shard_path not in self._shard_to_mount:
            return load_file(shard_path, device="cpu")

        mount = self._shard_to_mount[shard_path]
        if mount not in self._nvme_mounts:
            logger.debug(
                "HC-52: mapped mount %s not in valid mounts, direct read for %s",
                mount, shard_path,
            )
            return load_file(shard_path, device="cpu")

        logger.debug("HC-52: reading %s from mount %s", shard_path, mount)
        handle = self._get_mount_handle(mount, shard_path)

        try:
            handle.seek(0)
            file_bytes = handle.read()
        except Exception as ex:
            logger.error(
                "HC-52: mount read failed for %s on %s: %s — fallback to direct read",
                shard_path, mount, ex,
            )
            return load_file(shard_path, device="cpu")

        try:
            import tempfile
            tmp_suffix = f".airllm_{os.getpid()}_{layer_name.replace('.', '_')}.tmp"
            tmp_path = os.path.join(mount, tmp_suffix)
            try:
                with open(tmp_path, "wb") as tmp_f:
                    tmp_f.write(file_bytes)
                return load_file(tmp_path, device="cpu")
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        except Exception as ex:
            logger.error(
                "HC-52: failed to parse safetensors bytes for %s: %s",
                shard_path, ex,
            )
            return load_file(shard_path, device="cpu")

    # ── ModelPersister interface ─────────────────────────────────────────────

    def model_persist_exist(self, layer_name: str, saving_path: str) -> bool:
        return self._wrapped.model_persist_exist(layer_name, saving_path)

    def persist_model(self, state_dict, layer_name: str, saving_path: str):
        return self._wrapped.persist_model(state_dict, layer_name, saving_path)

    def load_model(self, layer_name: str, checkpoint_path: str):
        """Load a layer shard, routing to the appropriate NVME mount when active."""
        if self._executor is not None and self._nvme_mounts:
            return self._executor.submit(
                self._striped_read, layer_name, checkpoint_path,
            ).result()
        return self._striped_read(layer_name, checkpoint_path)


# ─────────────────────────────────────────────────────────────────────────────
# Monkey-patch injection (called by airllm_runner.py)
# ─────────────────────────────────────────────────────────────────────────────

def inject_striped_persister(model_path: str):
    """
    Build and inject a NVMEStripingModelPersister into the airllm.persist
    module so that subsequent AutoModel.load calls use multi-NVME reads.

    Must be called BEFORE ``from airllm import AutoModel``.
    """
    nvme_mounts = discover_nvme_mounts()
    if len(nvme_mounts) < 2:
        logger.debug("HC-52: fewer than 2 NVME mounts — not injecting striped persister")
        return None

    shard_to_mount = _build_shard_to_nvme_map(model_path)
    if not shard_to_mount:
        logger.debug(
            "HC-52: no shard→mount mapping found — not injecting striped persister",
        )
        return None

    from airllm.persist.safetensor_model_persister import SafetensorModelPersister
    from airllm.persist import model_persister as mp_module

    wrapped = SafetensorModelPersister()
    striped = NVMEStripingModelPersister(
        wrapped=wrapped,
        shard_to_mount=shard_to_mount,
        max_workers=len(nvme_mounts),
    )
    mp_module.model_persister = striped

    logger.info(
        "HC-52: NVMEStripingModelPersister injected — %d shards mapped across %d mounts",
        len(shard_to_mount), len(nvme_mounts),
    )
    return striped
