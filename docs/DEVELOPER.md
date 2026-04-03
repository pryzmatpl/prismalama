# Prismalama developer guide

![Prismalama Logo](../logo.jpg)

This document describes how the repository is wired so humans and automation can work on it without guessing. It is the source of truth for layout, runners, memory behavior, and tests.

## Goals (product)

- **Architectural key:** GGUF goes through **llama.cpp / GGML**; **AirLLM-style** layer / NVMe streaming for HF layouts is a **separate** Python stack — not replicated inside GGML. See **`docs/PRISMALAMA_PRINCIPLE.md`** and **`GET /api/prismalama/capabilities`** on a running server.
- **Ollama-like UX** for local inference on modest hardware (“potato machine”), **without** requiring heavy external Python ML stacks for the default path.
- **Vulkan** via llama.cpp / GGML for broad GPU support (see `ml/backend/ggml`) — **first-class**; packaged defaults (`OLLAMA_USE_AIRLLM=0`) assume **GGUF + GGML** only.
- **Weight streaming** for models larger than VRAM: GGUF uses llama.cpp mmap/offload/streaming semantics where enabled. PyTorch **AirLLM** streams Hugging Face–style checkpoints only when **`OLLAMA_USE_AIRLLM=1`** (and deps installed); with **`OLLAMA_USE_AIRLLM=0`** (package default), **only GGUF** is routed to the native runner — HF safetensors trees require opting in.
- **Small models** (e.g. Qwen3.5-9B-class) should run on the standard llama.cpp runner when quantized to GGUF; large HF-only layouts can use the AirLLM runner when explicitly enabled and heuristics match.

## Repository map

| Area | Role |
|------|------|
| `llm/` | Ollama-compatible server: scheduling, loads, API. |
| `runner/` | Process entry: chooses **llama** vs **AirLLM** Python runner vs other engines. |
| `runner/airllmrunner/` | Go HTTP front on **`port`** + `airllm_runner.py` on **`port+1`** (PyTorch AirLLM); see **`docs/RUNTIME_DISPATCH.md` § AirLLM runner**. |
| `ml/` | GGML backends (CUDA, ROCm, **Vulkan**, Metal, CPU). |
| `src/airllm/air_llm/` | Vendored/custom AirLLM Python package (`airllm` on `PYTHONPATH`). |
| `integration/` | Tag-gated Go integration tests (`//go:build integration`). |
| `ARCHITECTURE.md` | High-level diagrams (server → scheduler → runner → backend). |

## Which runner is used?

`runner/runner.go` implements `isAirLLMModel(modelPath)`:

- Directory contains `model.safetensors.index.json` or `*.safetensors` → **AirLLM**.
- `config.json` mentions safetensors / `torch_dtype` / transformers-style hints → **AirLLM**.
- Glob `*-00001-of-*.gguf` (multi-part GGUF) → **AirLLM** when **`OLLAMA_USE_AIRLLM` is not an explicit opt-out** (`0` / `false` / `no`); otherwise **GGML** (default package sets opt-out so multi-part uses native GGML).
- `OLLAMA_USE_AIRLLM=1` or `true` → **AirLLM** even for plain `.gguf` (testing / overrides).

Otherwise the **llama.cpp** runner is used (GGUF, Vulkan when built).

**Important:** Full **143GB+ class** models in **Hugging Face safetensors** layout use AirLLM’s layer streaming. **MiniMax / Kimi “GGUF-only”** installs are not loaded by PyTorch AirLLM unless you also have an HF-compatible tree or you rely on the multi-part GGUF + `OLLAMA_USE_AIRLLM` path that routes to the AirLLM runner (see integration tests under `minimax` / `weight_streaming` tags—they assume local paths under `/nvme3/...`).

## Environment variables (frequently used)

| Variable | Effect |
|----------|--------|
| `OLLAMA_USE_AIRLLM` | **Arch package sets `0`**: GGML/llama.cpp for typical GGUF; no PyTorch deps. Set **`1`** / **`true`** to opt into AirLLM. **`0`** / **`false`** / **`no`** disables **all** AirLLM routing. If **unset**, layout heuristics may still pick AirLLM (e.g. multipart GGUF) — see `docs/RUNTIME_DISPATCH.md`. |
| `OLLAMA_MULTI_GGUF` | Treat as AirLLM-style when `1`. |
| `AIRLLM_COMPRESSION` | e.g. `4bit`, `8bit`, `none` (passed to `AutoModel.from_pretrained`). |
| `AIRLLM_DEVICE` | PyTorch device string, default `cuda:0` (ROCm uses the same API). |
| `PRISMALAMA_AIRLLM_PYTHONPATH` | Optional prepend for `PYTHONPATH` when automatic dev-tree detection does not match your layout (colon-separated). |
| `AIRLLM_POST_INFER_CLEANUP` | If `0`, skip post-inference GPU cache flush in `airllm_runner.py` (default: on). |
| `OLLAMA_LAYER_STREAMING` | GGUF layer streaming: load block from NVMe → compute → evict (default **enabled** in package; `OLLAMA_LAYER_STREAMING=0` disables). See **`ml/streaming`** and **`docs/PRISMALAMA_PRINCIPLE.md`**. |
| `OLLAMA_STREAMING_BUDGET` | Byte budget for the streaming buffer pool (default 4 GiB). |
| `PYTORCH_CUDA_ALLOC_CONF` | e.g. `expandable_segments:True` to reduce allocator fragmentation (set by user; not modified by the repo). |

## GPU memory after inference (PyTorch / AirLLM)

1. **Per-layer:** `AirLLMBaseModel.forward` in `src/airllm/air_llm/airllm/airllm_base.py` moves layers to `"meta"` and calls `clean_memory()` after each layer.
2. **Post-request:** `airllm_runner.py` calls `finalize_inference_memory()` from `airllm.utils` after each completion: `torch.cuda.synchronize()` (when CUDA/ROCm is available), `torch.cuda.empty_cache()`, `gc`, and `malloc_trim` where applicable.

Residual VRAM reported by `nvidia-smi` may still reflect PyTorch’s **caching allocator** holding pools for speed; `empty_cache()` returns unused cached blocks to the driver when possible. The Go scheduler also implements **VRAM recovery waits** around runner unload (`server/sched.go`, `waitForVRAMRecovery`) because driver-reported free memory can lag.

## Vulkan

GGML Vulkan implementation lives under `ml/backend/ggml/ggml/src/ggml-vulkan/`. Backend selection is through the normal Ollama/Prismalama device and build flags—not duplicated here; see `src/ollama/docs/gpu.md` in the vendored tree if present, or your distribution’s build docs.

## Integration tests and coverage

Tests are **tag-gated** so CI and laptops do not require 143GB models.

```bash
# Core integration (no GPU-specific tags)
go test -tags=integration ./integration -count=1 -timeout 10m

# Coverage (lines hit depend on skips and machine)
go test -tags=integration ./integration -coverprofile=/tmp/integration.cov -covermode=atomic -timeout 10m
go tool cover -func=/tmp/integration.cov | tail -5
```

| Build tags (examples) | Purpose |
|------------------------|---------|
| `integration` | Shared harness, basic API tests. |
| `integration,airllm` | Needs `OLLAMA_TEST_AIRLLM=1` and often a model env. |
| `integration,gpu` | GPU / VRAM related checks. |
| `integration,minimax` | Large local paths; skips if missing. |
| `integration,weight_streaming` | Multi-part GGUF / streaming checks. |
| `integration,perf` | Benchmark-style tests. |

See `integration/TEST_README.md` for the full table and env vars.

## Ship gate (feature → package)

For every **promised Prismalama feature**: (1) land the code, (2) **prove it with integration tests** (`go test -tags=integration …` plus any extra tags/env that feature needs), (3) **build the Arch package** from the repo root (`./build-rocm.sh` or `makepkg -sf`; see **`README-PKGBUILD.md`**). Bump **`pkgrel`** (or **`pkgver`**) in **`PKGBUILD`** when the shipped artifact should change.

**Automation:** **`make ship-check`** runs **`scripts/ship-check.sh`** (integration, then `build-rocm.sh`). **`make ship-check-fast`** runs **`TestBlueSky` only** and skips the package. Env: **`SHIP_SKIP_PKG=1`** (tests only), **`SHIP_GO_TEST_EXTRA`** (e.g. `-run TestFoo`), **`SHIP_INTEGRATION_TIMEOUT`**.

## Python AirLLM from a git checkout

The packaged install uses `/usr/share/ollama/airllm`. For development, `runner/airllmrunner/runner.go` prepends `../../src/airllm/air_llm` next to the runner when that tree exists so `import airllm` resolves to the repo copy.

## Upstream engine: prismallama.cpp (GGUF / llama.cpp)

The **default GGUF path** is **llama.cpp** embedded under `llama/` (Go bindings in `llama*.go`). Prismalama does **not** track ggml-org/llama.cpp directly for day-to-day work.

| Item | Location |
|------|----------|
| Canonical fork | **https://github.com/piotroxp/prismallama.cpp** |
| Vendoring into this repo | Root **`Makefile.sync`** (`UPSTREAM`, `FETCH_HEAD`, `llama/vendor` → rsync to `llama/llama.cpp` and `ml/backend/ggml/ggml`) |
| Maintainer workflow | **`llama/README.md`** (apply patches, sync, pin commits for releases) |

**Arch Linux / global deploys:** pin `FETCH_HEAD` in `Makefile.sync` to a **commit SHA** for reproducible binaries; branch names are fine for development only.

**Arch package (this repo):** root **`PKGBUILD`** builds Prismalama from source (CMake GGML CPU/HIP/Vulkan → `/usr/lib/ollama/rocm`, Go `ollama`, AirLLM under `/usr/share/ollama`). Run **`makepkg -sf`** or **`./build-rocm.sh`**; see **`README-PKGBUILD.md`**. Set **`PRISMALAMA_AMDGPU_TARGETS`** before `makepkg` if not `gfx1100`. After any change to **Go runners** or **`llm/`**, rebuild and reinstall, then restart the service — a stale **`/usr/bin/ollama`** will not pick up fixes. On Arch, the usual loop is **`sudo makepkg -sfi`** in the directory containing **`PKGBUILD`**, then **`sudo systemctl restart ollama`** (**`-s`** deps, **`-f`** force rebuild, **`-i`** install).

**Submodule note:** `src/ollama` may still ship its own `Makefile.sync` from upstream Ollama. Align it with the Prismalama fork when you merge the submodule or build from a unified tree.

## Docker

| Image / target | Dockerfile | Role |
|----------------|------------|------|
| `prismalama-test` | `docker/test/Dockerfile` | **CPU-only** GGML + `ollama` for CI and **`make ship-check-fast`** (no GPU). |
| `prismalama-gpu` | `docker/gpu/Dockerfile` | **AMD ROCm (HIP) + Vulkan + CPU** GGML under `/usr/lib/ollama/rocm`, Ubuntu/ROCm dev base; for GPU without Arch on the host. |
| `prismalama-arch` | `docker/arch/Dockerfile` | **Arch Linux** image: **`makepkg`** the root **`PKGBUILD`** inside Docker (or **`Dockerfile.prebuilt`** + `docker/arch/prismalama.pkg.tar.zst`) so the container matches a native **`pacman -U prismalama-ollama`** install. |

Build/run: **`make docker-test-build`** / **`make docker-test`** vs **`make docker-gpu-build`** / **`make docker-gpu-run`** vs **`make docker-arch-build`** / **`make docker-arch-prebuilt-build`** / **`make docker-arch-run`**. Kubernetes notes and example manifests: **`docker/gpu/README.md`**, **`docker/gpu/k8s/example-deployment.yaml`**. Arch package image: **`docker/arch/README.md`**.

## Related docs

- `PRISMALAMA_PRINCIPLE.md` — **north star**: GGML vs AirLLM, dispatch, capabilities endpoint.
- `ARCHITECTURE.md` — diagrams and component list.
- `README.md` — user-facing overview.
- `README-PKGBUILD.md` — Arch **`prismalama-ollama`** package build.
- `llama/README.md` — vendoring **prismallama.cpp** into `llama/`.
- `integration/TEST_README.md` — test tags and hardware expectations.
- `docker/gpu/README.md` — AMD GPU container (ROCm HIP + Vulkan) and Kubernetes.
- `docker/arch/README.md` — Arch **`prismalama-ollama`** package as a container (same files as `pacman -U`).
- `docs/RUNTIME_DISPATCH.md` — which runner (**llama** vs **AirLLM**) handles a model; read **`runner dispatch`** logs.
