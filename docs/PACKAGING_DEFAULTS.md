# Prismalama packaging defaults — single inventory

> Single source of truth for what env vars each Prismalama artifact
> (Arch PKGBUILD, Dockerfiles) ships with.
> Maintained by Phase 0 / [JAISIU-2160](https://pryzmat.youtrack.cloud/issue/JAISIU-2160).
> Audit workflow: `make print-defaults`.

## Legend

| Column | Meaning |
|--------|---------|
| **Artifact** | The file that ships the default (PKGBUILD heredoc → `/etc/default/ollama`, Dockerfile `ENV`, etc.) |
| **Env var** | The variable name |
| **Default** | The value that lands on the host / container when no operator override is present |
| **Rationale** | Why this default (cite `envconfig` code or docs) |
| **Doc ref** | Where this is documented for operators |

## Inventory (snapshot 2026-08-13)

### Arch Linux — `PKGBUILD` writes `/etc/default/ollama`

| Artifact | Env var | Default | Rationale | Doc ref |
|----------|---------|---------|-----------|---------|
| PKGBUILD | `OLLAMA_MODELS` | `${_model_dir}` (typically `/var/lib/ollama`) | Standard Ollama path | `docs/PRISMALAMA_PRINCIPLE.md` |
| PKGBUILD | `OLLAMA_HOST` | `127.0.0.1:11434` | Loopback only; systemd unit uses it | `README-PKGBUILD.md` |
| PKGBUILD | `OLLAMA_NUM_PARALLEL` | `1` | Conservative — single-user | `envconfig/config.go` |
| PKGBUILD | `OLLAMA_KEEP_ALIVE` | `5m` | Matches `envconfig` default | `envconfig/config.go` |
| PKGBUILD | `OLLAMA_LIBRARY_PATH` | `/usr/lib/ollama/rocm` | Where `makepkg` installs the GGML libs | `PKGBUILD` |
| PKGBUILD | `OLLAMA_LAYER_STREAMING` | `1` | **Native GGUF streaming default ON** (Arch-specific; Go code default is `0`) | `docs/DEVELOPER.md`, `docs/RUNTIME_DISPATCH.md` |
| PKGBUILD | `OLLAMA_USE_AIRLLM` | `0` | GGML-first; opt-in for AirLLM | `docs/PRISMALAMA_PRINCIPLE.md` |
| PKGBUILD | `AIRLLM_COMPRESSION` | `4bit` | AirLLM default | `runner/airllmrunner/airllm_runner.py` |
| PKGBUILD | `AIRLLM_DEVICE` | `cuda:0` | ROCm uses the CUDA device API name | `docs/RUNTIME_DISPATCH.md` |
| PKGBUILD | `HIP_VISIBLE_DEVICES` (amd/all profiles) | `0` | First AMD GPU only | ROCm convention |
| PKGBUILD | `HSA_OVERRIDE_GFX_VERSION` (amd/all profiles) | `11.0.0` | gfx1100 (RX 7900 XTX / Nav31) override | ROCm compat |

### Docker — `Dockerfile.gpu`

| Artifact | Env var | Default | Rationale | Doc ref |
|----------|---------|---------|-----------|---------|
| Dockerfile.gpu | `OLLAMA_HOST` | `0.0.0.0:11434` | Containers expose on all interfaces (vs Arch loopback) | docker/gpu/README.md |
| Dockerfile.gpu | `OLLAMA_MODELS` | `/root/.ollama` | Container default | docker/gpu/README.md |
| Dockerfile.gpu | `OLLAMA_MAX_LOADED_MODELS` | `2` | Higher than Arch's implicit 1 | docker/gpu/README.md |
| Dockerfile.gpu | `OLLAMA_NUM_PARALLEL` | `4` | Higher than Arch's `1` (container has headroom) | docker/gpu/README.md |
| Dockerfile.gpu | `OLLAMA_LIBRARY_PATH` | `/usr/lib/ollama/rocm` | Matches PKGBUILD layout | docker/gpu/README.md |

### Docker — `docker/arch/Dockerfile`

| Artifact | Env var | Default | Rationale | Doc ref |
|----------|---------|---------|-----------|---------|
| docker/arch/Dockerfile | `PRISMALAMA_AMDGPU_TARGETS` | `${PRISMALAMA_AMDGPU_TARGETS}` (build-arg) | Lets the operator pin GPU ISA at image build | `README-PKGBUILD.md` |
| docker/arch/Dockerfile | `OLLAMA_HOST` | `0.0.0.0:11434` | Container default | docker/arch/README.md |
| docker/arch/Dockerfile | `OLLAMA_MODELS` | `/var/lib/ollama` | Arch path (matches `/etc/default/ollama` contract) | docker/arch/README.md |
| docker/arch/Dockerfile | `OLLAMA_LIBRARY_PATH` | `/usr/lib/ollama/rocm` | Matches PKGBUILD | docker/arch/README.md |

### Docker — `docker/arch/Dockerfile.prebuilt`

| Artifact | Env var | Default | Rationale | Doc ref |
|----------|---------|---------|-----------|---------|
| docker/arch/Dockerfile.prebuilt | `OLLAMA_HOST` | `0.0.0.0:11434` | Container default | docker/arch/README.md |
| docker/arch/Dockerfile.prebuilt | `OLLAMA_MODELS` | `/var/lib/ollama` | Arch path | docker/arch/README.md |
| docker/arch/Dockerfile.prebuilt | `OLLAMA_LIBRARY_PATH` | `/usr/lib/ollama/rocm` | Matches PKGBUILD | docker/arch/README.md |

### Docker — `docker/gpu/Dockerfile` (ROCm dev base, Ubuntu)

| Artifact | Env var | Default | Rationale | Doc ref |
|----------|---------|---------|-----------|---------|
| docker/gpu/Dockerfile | `HIP_PATH` | `/opt/rocm` | ROCm install layout | docker/gpu/README.md |
| docker/gpu/Dockerfile | `OLLAMA_LIBRARY_PATH` | `/usr/lib/ollama/rocm` | Matches PKGBUILD | docker/gpu/README.md |
| docker/gpu/Dockerfile | `OLLAMA_HOST` | `0.0.0.0:11434` | Container default | docker/gpu/README.md |

### Docker — `docker/test/Dockerfile` (CPU-only test image)

| Artifact | Env var | Default | Rationale | Doc ref |
|----------|---------|---------|-----------|---------|
| docker/test/Dockerfile | `OLLAMA_BIN` | `/usr/bin/ollama` | Where the test image installs the binary | docker/test/README.md |
| docker/test/Dockerfile | `OLLAMA_LIBRARY_PATH` | `/usr/lib/ollama` | **No `/rocm` suffix** — CPU-only GGML libs (intentional divergence) | docker/test/README.md |

### Go `envconfig` code defaults (the canonical source for any env var that maps to a getter)

| Env var | Code default | Used by |
|---------|--------------|---------|
| `OLLAMA_USE_AIRLLM` | unset (heuristic) | `runner/dispatch.go` |
| `OLLAMA_MULTI_GGUF` | unset | `runner/dispatch.go` |
| `OLLAMA_LAYER_STREAMING` | `false` (Go) / `1` (Arch PKGBUILD) | `envconfig.LayerStreaming()` |
| `OLLAMA_STREAMING_BUDGET` | `4 * format.GibiByte` (4 GiB) | `envconfig.StreamingBudgetBytes()` |
| `OLLAMA_GPU_OVERHEAD` | `3 * format.GibiByte` (3 GiB) | `envconfig.GpuOverhead()` |
| `OLLAMA_MMAP_ALLOW_LOW_RAM` | unset (`false`) | `envconfig.MmapAllowLowRamLinux()` |
| `OLLAMA_KEEP_ALIVE` | `5m` | `envconfig.KeepAlive()` |
| `OLLAMA_VULKAN` | unset (`false`) | `envconfig.EnableVulkan(true)` |
| `OLLAMA_MEMORY_POLICY` | unset (`performance` per `docs/RUNTIME_DISPATCH.md`) | `envconfig.MemoryPolicy()` |
| `AIRLLM_COMPRESSION` | unset (Python default `4bit`) | `airllm_runner.py` |
| `AIRLLM_DEVICE` | unset (Python default `cuda:0`) | `runner/airllmrunner/runner.go` |
| `OLLAMA_NUM_PARALLEL` | unset (Go default `1`) | `envconfig.NumParallel()` |
| `OLLAMA_MAX_LOADED_MODELS` | unset (Go default `2 × num_gpu`) | `envconfig.MaxLoadedModels()` |

## Intentional divergences (DO NOT "fix")

| Where | What | Why |
|-------|------|-----|
| PKGBUILD `OLLAMA_HOST=127.0.0.1` vs Dockerfiles `0.0.0.0` | Bind address | systemd unit on the host uses loopback; containers must accept external connections |
| Docker test `OLLAMA_LIBRARY_PATH=/usr/lib/ollama` (no `/rocm`) vs PKGBUILD `/usr/lib/ollama/rocm` | Library subdir | CPU-only test image does not need ROCm libs |
| PKGBUILD `OLLAMA_NUM_PARALLEL=1` vs Dockerfile.gpu `=4` | Concurrency | Arch defaults are conservative; containers ship with headroom |
| PKGBUILD `OLLAMA_LAYER_STREAMING=1` (Arch) vs Go `envconfig` default `false` | Streaming opt-in | Arch package sets the env explicitly; bare-binary builds must opt in via envconfig or the env var |

## How to audit (and re-audit)

```bash
# Print the inventory in CI:
make print-defaults

# Verify the doc matches the artifacts:
diff <(make print-defaults) docs/PACKAGING_DEFAULTS.md

# When adding a new artifact default, add a row above AND update
# make print-defaults so the two never drift.
```

## Related

- [JAISIU-2160](https://pryzmat.youtrack.cloud/issue/JAISIU-2160) — Phase 0 packaging audit
- [JAISIU-2156](https://pryzmat.youtrack.cloud/issue/JAISIU-2156) — North-Star Epic
- `PKGBUILD` — Arch source of truth for `/etc/default/ollama`
- `Dockerfile.gpu`, `Dockerfile`, `docker/arch/Dockerfile`, `docker/arch/Dockerfile.prebuilt`, `docker/test/Dockerfile`, `docker/gpu/Dockerfile` — Docker sources
- `envconfig/config.go` — Go canonical defaults
- `scripts/operator-env-large-models.sh` — Operator-facing large-model guidance (must agree with the inventory above)