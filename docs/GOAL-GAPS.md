# Goals and gaps (Prismalama)

This document tracks **intent vs shipped behavior** and **known limits**. It complements **`PRISMALAMA_PRINCIPLE.md`** (architecture) and **`WEIGHT_STREAMING_STRATEGY.md`** (options).

## Product goals (stable)

| Goal | Status |
|------|--------|
| Ollama-compatible HTTP API and CLI | Shipped; additive endpoint `GET /api/prismalama/capabilities` |
| GGUF inference via prismallama.cpp / GGML (HIP, Vulkan, CPU) | Shipped |
| Explicit GGML vs AirLLM dispatch (`runner/dispatch.go`) | Shipped; tested |
| Optional AirLLM runner for HF-style trees | Shipped; **`OLLAMA_USE_AIRLLM=1`** required when package sets opt-out |
| Documented semantics (mmap/offload vs PyTorch streaming) | **`docs/RUNTIME_DISPATCH.md`**, capabilities JSON |

## GGUF “streaming” vs AirLLM streaming

| Capability | GGML / GGUF path today | AirLLM path |
|------------|-------------------------|-------------|
| Larger-than-RAM via mmap / disk | Yes (OS + loader policy; see **`OLLAMA_MMAP_ALLOW_LOW_RAM`**) | Different mechanics |
| Layer-wise NVMe execution (PyTorch sense) | **`OLLAMA_LAYER_STREAMING`**: optional **`LoadStreaming`**, **`InferenceStreamer`**, eval callback — **requires backend support**; otherwise falls back to normal load | Native domain for HF checkpoints |
| Peak VRAM bound to “one block + KV” | Target for streaming compute path; **depends on model/backend** | Where supported by AirLLM |

**Gap:** Full parity with AirLLM across **all** GGUF architectures and backends is **not** guaranteed; see runner logs for whether streaming load/inference activated.

## Operational gaps

| Topic | Gap / note |
|-------|------------|
| **Operator defaults for huge GGUF** | Bare-binary defaults: **`OLLAMA_LAYER_STREAMING` off**, **`OLLAMA_MEMORY_POLICY=performance`**, Linux **Vulkan backends skipped** unless **`OLLAMA_VULKAN=1`**. Use **`scripts/operator-env-large-models.sh`** (source before `ollama serve`) and **`scripts/verify-prismalama-runtime.sh`** to inspect **`GET /api/prismalama/capabilities`** (`operator_hints`). |
| **`OLLAMA_USE_AIRLLM=0`** | Disables **all** AirLLM routing (including safetensors/multipart heuristics). Arch package ships **`0`** — GGUF-first; HF workflows must opt in with **`1`**. |
| **`OLLAMA_LAYER_STREAMING` unset** | Go default is **off** (`envconfig`); Arch **`/etc/default/ollama`** sets **`1`**. Document “default” with context (source vs package). |
| **`OLLAMA_GPU_OVERHEAD`** | Reserved VRAM per GPU; default **3 GiB** in code (`envconfig`), not 2 GiB. |
| Tests | Full **`go test ./server/...`** can be slow on cold builds; use timeouts or `-short` / scoped packages for iteration. |
| Module path | `go.mod` remains **`github.com/ollama/ollama`** for upstream alignment; imports read like upstream. |

## Roadmap (from principle)

Medium/long term: push **GGUF block streaming** deeper into **prismallama.cpp / GGML** (I/O, eviction, graph lifetime) while keeping **AirLLM** for HF layouts that never become a single GGUF — see **`WEIGHT_STREAMING_STRATEGY.md`**.
