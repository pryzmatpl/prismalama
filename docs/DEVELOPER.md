# Prismalama developer guide

This document describes how the repository is wired so humans and automation can work on it without guessing. It is the source of truth for layout, runners, memory behavior, and tests.

## Goals (product)

- **Ollama-like UX** for local inference on modest hardware (“potato machine”).
- **Vulkan** via llama.cpp / GGML for broad GPU support (see `ml/backend/ggml`).
- **Weight streaming** for models larger than VRAM: PyTorch **AirLLM** streams Hugging Face–style checkpoints from disk layer-by-layer; GGUF path uses llama.cpp’s mmap/streaming semantics where enabled.
- **Small models** (e.g. Qwen3.5-9B-class) should run on the standard llama.cpp runner when quantized to GGUF; large HF-only models use the AirLLM runner when the model directory matches the heuristics below.

## Repository map

| Area | Role |
|------|------|
| `llm/` | Ollama-compatible server: scheduling, loads, API. |
| `runner/` | Process entry: chooses **llama** vs **AirLLM** Python runner vs other engines. |
| `runner/airllmrunner/` | Go HTTP front + `airllm_runner.py` subprocess (PyTorch AirLLM). |
| `ml/` | GGML backends (CUDA, ROCm, **Vulkan**, Metal, CPU). |
| `src/airllm/air_llm/` | Vendored/custom AirLLM Python package (`airllm` on `PYTHONPATH`). |
| `integration/` | Tag-gated Go integration tests (`//go:build integration`). |
| `ARCHITECTURE.md` | High-level diagrams (server → scheduler → runner → backend). |

## Which runner is used?

`runner/runner.go` implements `isAirLLMModel(modelPath)`:

- Directory contains `model.safetensors.index.json` or `*.safetensors` → **AirLLM**.
- `config.json` mentions safetensors / `torch_dtype` / transformers-style hints → **AirLLM**.
- Glob `*-00001-of-*.gguf` (multi-part GGUF) → **AirLLM** (forced path for weight-streaming experiments).
- `OLLAMA_USE_AIRLLM=1` or `true` → **AirLLM** even for plain `.gguf` (testing / overrides).

Otherwise the **llama.cpp** runner is used (GGUF, Vulkan when built).

**Important:** Full **143GB+ class** models in **Hugging Face safetensors** layout use AirLLM’s layer streaming. **MiniMax / Kimi “GGUF-only”** installs are not loaded by PyTorch AirLLM unless you also have an HF-compatible tree or you rely on the multi-part GGUF + `OLLAMA_USE_AIRLLM` path that routes to the AirLLM runner (see integration tests under `minimax` / `weight_streaming` tags—they assume local paths under `/nvme3/...`).

## Environment variables (frequently used)

| Variable | Effect |
|----------|--------|
| `OLLAMA_USE_AIRLLM` | Force AirLLM runner when set to `1` or `true`. |
| `OLLAMA_MULTI_GGUF` | Treat as AirLLM-style when `1`. |
| `AIRLLM_COMPRESSION` | e.g. `4bit`, `8bit`, `none` (passed to `AutoModel.from_pretrained`). |
| `AIRLLM_DEVICE` | PyTorch device string, default `cuda:0` (ROCm uses the same API). |
| `PRISMALAMA_AIRLLM_PYTHONPATH` | Optional prepend for `PYTHONPATH` when automatic dev-tree detection does not match your layout (colon-separated). |
| `AIRLLM_POST_INFER_CLEANUP` | If `0`, skip post-inference GPU cache flush in `airllm_runner.py` (default: on). |
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

**Submodule note:** `src/ollama` may still ship its own `Makefile.sync` from upstream Ollama. Align it with the Prismalama fork when you merge the submodule or build from a unified tree.

## Related docs

- `ARCHITECTURE.md` — diagrams and component list.
- `README.md` — user-facing overview.
- `llama/README.md` — vendoring **prismallama.cpp** into `llama/`.
- `integration/TEST_README.md` — test tags and hardware expectations.
