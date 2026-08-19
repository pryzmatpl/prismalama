# Prismalama CODEMAP

> **Source of truth for the prismalama codebase layout.**
> Architecture and product principle: [`docs/PRISMALAMA_PRINCIPLE.md`](docs/PRISMALAMA_PRINCIPLE.md).
> Runtime engine selection: [`docs/RUNTIME_DISPATCH.md`](docs/RUNTIME_DISPATCH.md).
> Developer guide: [`docs/DEVELOPER.md`](docs/DEVELOPER.md).
> Living roadmap: [`NEXT.md`](NEXT.md).

Prismalama is an **Ollama-compatible fork** with two inference engines:
**GGML** (prismallama.cpp — GGUF, Vulkan/HIP/CUDA/Metal/CPU) and
**AirLLM** (Python — HF safetensors, NVMe layer streaming).
Dispatch is explicit and testable (`runner/dispatch.go`).

---

## Root layout

```
prismalama/
├── main.go                      — Binary entry → cmd.NewCLI()
├── go.mod / go.sum              — Module github.com/ollama/ollama (upstream alignment)
├── CMakeLists.txt               — Build orchestration (GGML backends, llama-server)
├── PKGBUILD                     — Arch prismalama-ollama package definition
├── Makefile                     — Build/test/ship targets
├── Makefile.sync                — Upstream prismallama.cpp sync automation
├── build.sh / build-rocm.sh     — Platform build scripts
├── airllm_runner.py             — Python AirLLM HTTP server (port+1)
├── NEXT.md                      — Living roadmap (current phase, tickets)
├── AGENTS.md / CLAUDE.md        — Agent operating instructions
├── README.md                    — User-facing overview
├── CONTRIBUTING.md              — Contribution guide
└── SECURITY.md                  — Vulnerability reporting
```

---

## Go packages

### Entry and CLI (`cmd/`)

| Package       | Purpose                                                                                     |
| ------------- | ------------------------------------------------------------------------------------------- |
| `cmd/`        | Cobra root command, `serve`, `run`, `create`, `pull`, `push`, `list`, `ps`, `show`, `embed` |
| `cmd/runner/` | Subprocess entry for runners — `main()` → `runner.Execute()`                                |
| `cmd/launch/` | Integration launchers (Claude Code, Codex, Copilot, Droid, OpenClaw, OpenCode)              |
| `cmd/config/` | Config file I/O, OAuth token storage                                                        |
| `cmd/tui/`    | Interactive model selector (Bubbletea)                                                      |
| `cmd/bench/`  | Benchmarking command                                                                        |

### Server and scheduling (`server/`, `llm/`)

| Package                             | Purpose                                                          |
| ----------------------------------- | ---------------------------------------------------------------- |
| `server/`                           | HTTP API routes (Gin), model management, auth, quantization      |
| `server/routes.go`                  | `/api/generate`, `/api/chat`, `/api/ps`, CRUD endpoints          |
| `server/sched.go`                   | `Scheduler` — request queue, GPU memory, load/evict/recovery     |
| `server/prismalama_capabilities.go` | `GET /api/prismalama/capabilities` — operator-facing JSON        |
| `llm/`                              | Load lifecycle (`LoadRequest` → commit → inference), mmap policy |
| `llm/server.go`                     | `llamaServer` / `ollamaServer` — runner subprocess management    |
| `llm/gpu_layers.go`                 | GPU layer offload calculation (`GPULayersList`)                  |
| `llm/ollama_engine_server.go`       | Ollama-engine client (new `runner --ollama-engine` path)         |
| `llm/adaptive_context.go`           | Context window adaptation                                        |
| `llm/distributed/`                  | Distributed inference helpers                                    |
| `llm/warmup/`                       | Model warmup utilities                                           |

### Runner dispatch (`runner/`)

| Package                | Purpose                                                      |
| ---------------------- | ------------------------------------------------------------ |
| `runner/dispatch.go`   | **`DecideEngine`** — single testable GGML vs AirLLM decision |
| `runner/runner.go`     | `Execute()` — subprocess router, reason logging              |
| `runner/llamarunner/`  | llama.cpp GGUF runner (CGo, streaming policy)                |
| `runner/ollamarunner/` | Ollama-engine runner (in-process, multimodal cache)          |
| `runner/airllmrunner/` | AirLLM Go→Python proxy (HTTP on port, Python on port+1)      |
| `runner/common/`       | Shared utilities (stop sequences, Unicode handling)          |

### ML backends (`ml/`)

| Package                             | Purpose                                                                    |
| ----------------------------------- | -------------------------------------------------------------------------- |
| `ml/`                               | `Backend`, `Tensor`, `Context` interfaces; device detection                |
| `ml/backend.go`                     | Backend registry, `NewBackend()` factory                                   |
| `ml/device.go`                      | GPU discovery, `DeviceInfo`, priority (CUDA > ROCm > Vulkan > Metal > CPU) |
| `ml/backend/ggml/`                  | CGo bindings to ggml.h — context, tensor, graph management                 |
| `ml/backend/ggml/ggml_streaming.go` | `StreamingBackend` / `StreamingComputeBackend` interfaces                  |
| `ml/backend/vulkan/`                | Vulkan backend experiments                                                 |
| `ml/streaming/`                     | Layer/NVMe streaming orchestration (see below)                             |
| `ml/streaming/inference.go`         | `InferenceStreamer` — block-by-block weight rotation                       |
| `ml/streaming/layermap.go`          | `LayerMap` — GGUF block boundary detection                                 |
| `ml/streaming/budget.go`            | VRAM budget tracker (default 4 GiB)                                        |
| `ml/streaming/prefetch.go`          | NVMe read-ahead scheduler                                                  |
| `ml/streaming/streamer.go`          | Orchestrator                                                               |
| `ml/weightimage/`                   | BC4 weight-as-image compression                                            |
| `ml/quantization/`                  | Quantization utilities                                                     |
| `ml/nn/`                            | NN components: attention, RoPE, MoE, pooling                               |
| `ml/attention/`                     | Attention helpers                                                          |
| `ml/cache/`                         | KV cache management                                                        |

### Model architectures (`model/`)

| Package                     | Purpose                                                |
| --------------------------- | ------------------------------------------------------ |
| `model/`                    | Model registry, config parsing, architecture detection |
| `model/models/llama/`       | Llama family                                           |
| `model/models/llama4/`      | Llama 4                                                |
| `model/models/qwen2/`       | Qwen2                                                  |
| `model/models/qwen3/`       | Qwen3                                                  |
| `model/models/qwen25vl/`    | Qwen2.5-VL                                             |
| `model/models/qwen3vl/`     | Qwen3-VL                                               |
| `model/models/qwen35/`      | Qwen3.5                                                |
| `model/models/qwen3next/`   | Qwen3-Next (DeltaNet/GDN — `ggml_gated_delta_net`)     |
| `model/models/gemma2/`      | Gemma 2                                                |
| `model/models/gemma3/`      | Gemma 3                                                |
| `model/models/gemma3n/`     | Gemma 3n                                               |
| `model/models/gemma4/`      | Gemma 4                                                |
| `model/models/deepseek2/`   | DeepSeek V2                                            |
| `model/models/deepseekocr/` | DeepSeek OCR                                           |
| `model/models/mistral3/`    | Mistral 3                                              |
| `model/models/mllama/`      | MLlama                                                 |
| `model/models/glm4moelite/` | GLM4-MoE-Lite                                          |
| `model/models/glmocr/`      | GLM OCR                                                |
| `model/models/bert/`        | BERT                                                   |
| `model/models/nomicbert/`   | Nomic-BERT                                             |
| `model/models/laguna/`      | Laguna                                                 |
| `model/models/lfm2/`        | LFM2                                                   |
| `model/models/olmo3/`       | OLMo 3                                                 |
| `model/models/gptoss/`      | GPT-OSS                                                |
| `model/parsers/`            | Modelfile/config parsers                               |
| `model/renderers/`          | Modelfile renderers                                    |
| `model/imageproc/`          | Image preprocessing                                    |
| `model/input/`              | Input processing                                       |

### File format (`fs/`)

| Package              | Purpose                                          |
| -------------------- | ------------------------------------------------ |
| `fs/`                | Config types, file management                    |
| `fs/gguf/`           | GGUF reader — tensor iteration, metadata parsing |
| `fs/gguf/stream/`    | GGUF streaming reader                            |
| `fs/ggml/`           | GGML format detection, `OllamaEngineRequired()`  |
| `fs/util/bufioutil/` | Buffered I/O utilities                           |

### API and protocol (`api/`, `openai/`)

| Package      | Purpose                                                          |
| ------------ | ---------------------------------------------------------------- |
| `api/`       | REST API types, `PrismalamaCapabilitiesResponse`, dispatch types |
| `openai/`    | OpenAI-compatible endpoint adapter                               |
| `anthropic/` | Anthropic API adapter                                            |

### Infrastructure

| Package                | Purpose                                                 |
| ---------------------- | ------------------------------------------------------- |
| `discover/`            | Hardware/GPU discovery (HIP, CUDA, Vulkan, CPU probing) |
| `envconfig/`           | Environment variable parsing and defaults               |
| `format/`              | Human-readable bytes, time formatting                   |
| `progress/`            | Progress bar / display                                  |
| `readline/`            | Interactive readline                                    |
| `logutil/`             | Logging utilities                                       |
| `manifest/`            | Model manifest management                               |
| `auth/`                | Authentication                                          |
| `version/`             | Version info                                            |
| `convert/`             | Model format conversion (Qwen, Llama, Gemma, etc.)      |
| `parser/`              | Modelfile parser                                        |
| `template/`            | Go template helpers                                     |
| `tokenizer/`           | Tokenizer utilities                                     |
| `sample/`              | Sampling (top-k, top-p, temperature, mirostat)          |
| `thinking/`            | Thinking/reasoning token handling                       |
| `tools/`               | Tool-call parsing                                       |
| `kvcache/`             | KV cache types                                          |
| `harmony/`             | Harmony utilities                                       |
| `metrics/`             | Metrics collection                                      |
| `middleware/`          | HTTP middleware                                         |
| `internal/cloud/`      | Cloud inference proxy                                   |
| `internal/modelref/`   | Model reference parsing                                 |
| `internal/orderedmap/` | Ordered map                                             |

### Desktop and app (`app/`)

| Package        | Purpose                                |
| -------------- | -------------------------------------- |
| `app/cmd/app/` | Desktop app lifecycle (Darwin/Windows) |
| `app/ui/`      | Desktop UI rendering                   |
| `app/auth/`    | OAuth/signin                           |
| `app/tools/`   | Web search, browser integration        |
| `app/webview/` | WebView component                      |
| `app/store/`   | App store helpers                      |
| `app/updater/` | Auto-update                            |
| `app/server/`  | Embedded server                        |

### Experimental (`x/`)

| Package          | Purpose                                   |
| ---------------- | ----------------------------------------- |
| `x/mlxrunner/`   | MLX engine runner (Apple Metal)           |
| `x/imagegen/`    | MLX-based image generation (Flux, Zimage) |
| `x/models/`      | Experimental model architectures          |
| `x/tokenizer/`   | Fast BPE tokenizer                        |
| `x/safetensors/` | Safetensors extractor                     |
| `x/transfer/`    | Model download / sparse transfer          |
| `x/tools/`       | Web search / fetch integration            |
| `x/server/`      | Experimental server                       |
| `x/agent/`       | Agent helpers                             |
| `x/create/`      | Model creation helpers                    |

### AirLLM Python (`src/airllm/`, `airllm_runner.py`)

| File                  | Purpose                                                         |
| --------------------- | --------------------------------------------------------------- |
| `airllm_runner.py`    | Python HTTP server: `/load`, `/status`, `/completion`, `/embed` |
| `src/airllm/air_llm/` | Vendored AirLLM package (layer-by-layer HF inference)           |

---

## C/C++ layer (`llama/`)

Vendored **prismallama.cpp** fork (https://github.com/piotroxp/prismallama.cpp).
Synced via `Makefile.sync` (`FETCH_HEAD` pin → rsync to `llama/llama.cpp` + `ml/backend/ggml/ggml`).

```
llama/
├── llama.go                — Go CGo bindings
├── llama_test.go           — Binding tests
├── sampling_ext.h          — Sampling C extensions
├── build-info.cpp          — Build info
├── patches/                — 32+ local patches (audited in patches/README.md)
│   ├── 0002-pretokenizer.patch
│   ├── 0007-sort-devices-by-score.patch
│   ├── 0024-GPU-discovery-enhancements.patch
│   ├── 0027-interleave-multi-rope.patch
│   └── README.md           — Patch audit table + bisect policy
└── llama.cpp/              — Vendored source tree
    └── src/
        ├── ggml-cuda/      — CUDA backend
        ├── ggml-hip/       — ROCm/HIP backend
        ├── ggml-vulkan/    — Vulkan backend (first-class in Prismalama)
        ├── ggml-metal/     — Metal backend
        └── ggml-cpu/       — CPU backend (all variants)
```

**Sync commands:**

```bash
make -f Makefile.sync sync                # Apply patches + rsync
make -f Makefile.sync sync-audit-check    # CI guard: README vs FETCH_HEAD
make -f Makefile.sync print-patches-audit # List every patch + subject
```

---

## Build system

| Target       | Command                                                | Purpose                                   |
| ------------ | ------------------------------------------------------ | ----------------------------------------- |
| Full build   | `cmake -B build . && cmake --build build --parallel 8` | GGML backends + llama-server              |
| Go-only      | `go build .`                                           | Go binary (uses pre-built native payload) |
| Arch package | `makepkg -sf` or `./build-rocm.sh`                     | `prismalama-ollama` .pkg.tar.zst          |
| Ship gate    | `make ship-check`                                      | Integration tests + package build         |
| Ship fast    | `make ship-check-fast`                                 | TestBlueSky only, no packaging            |

### CMake backends (conditional)

| Backend  | Flag                | Target                    |
| -------- | ------------------- | ------------------------- |
| CPU      | Always on           | `ggml-cpu` (all variants) |
| CUDA     | `LLAMA_CUDA=ON`     | `ggml-cuda`               |
| HIP/ROCm | `LLAMA_HIPBLAS=ON`  | `ggml-hip`                |
| Vulkan   | Auto on Linux       | `ggml-vulkan`             |
| Metal    | Auto on macOS arm64 | `ggml-metal`              |
| MLX      | `MLX_ENGINE=ON`     | Apple Metal via MLX       |

### Docker images

| Image             | Dockerfile               | Purpose                          |
| ----------------- | ------------------------ | -------------------------------- |
| `prismalama-test` | `docker/test/Dockerfile` | CPU-only GGML for CI             |
| `prismalama-gpu`  | `docker/gpu/Dockerfile`  | ROCm HIP + Vulkan + CPU (Ubuntu) |
| `prismalama-arch` | `docker/arch/Dockerfile` | Arch-based from PKGBUILD         |

---

## Engine dispatch

```
DecideEngine(modelPath):
├── OLLAMA_USE_AIRLLM ∈ {0, false, no}     → GGML (early exit)
├── OLLAMA_MULTI_GGUF=1                     → AirLLM
├── model path missing                       → GGML
├── model.safetensors.index.json             → AirLLM
├── *.safetensors shards                     → AirLLM
├── config.json HF heuristic                → AirLLM
├── *-00001-of-*.gguf (multipart)           → AirLLM
├── OLLAMA_USE_AIRLLM ∈ {1, true}           → AirLLM
└── else                                     → GGML
```

Arch package default: `OLLAMA_USE_AIRLLM=0` (GGML-only unless opt-in).

---

## Key environment variables

| Variable                    | Default                 | Purpose                                           |
| --------------------------- | ----------------------- | ------------------------------------------------- |
| `OLLAMA_USE_AIRLLM`         | `0` (Arch)              | `0`/`false`/`no` → GGML only; `1`/`true` → AirLLM |
| `OLLAMA_LAYER_STREAMING`    | `1` (Arch), unset (raw) | Block-by-block GGUF load when backends support it |
| `OLLAMA_STREAMING_BUDGET`   | 4 GiB                   | VRAM budget for streaming buffer pool             |
| `OLLAMA_GPU_OVERHEAD`       | 3 GiB                   | Reserved VRAM per GPU                             |
| `OLLAMA_VULKAN`             | unset                   | `1` to enable Vulkan backends on Linux            |
| `OLLAMA_KEEP_ALIVE`         | `5m`                    | Model unload timeout                              |
| `AIRLLM_DEVICE`             | `cuda:0`                | PyTorch device (ROCm uses same API)               |
| `AIRLLM_COMPRESSION`        | unset                   | `4bit`, `8bit`, `none`                            |
| `AIRLLM_POST_INFER_CLEANUP` | `1`                     | `0` to skip post-inference GPU cache flush        |

---

## Test coverage

**359 test files** across the codebase (excluding vendored `llama/`).

| Area                | Test files | Tags                                                                       | Notes                                             |
| ------------------- | ---------- | -------------------------------------------------------------------------- | ------------------------------------------------- |
| `integration/`      | 34         | `integration`, `+airllm`, `+gpu`, `+minimax`, `+weight_streaming`, `+perf` | Tag-gated; skips when models/hardware missing     |
| `server/`           | 30         | —                                                                          | Routes, scheduler, capabilities, model management |
| `cmd/launch/`       | 25         | —                                                                          | Integration launchers                             |
| `model/renderers/`  | 21         | —                                                                          | Modelfile rendering                               |
| `model/parsers/`    | 19         | —                                                                          | Modelfile/config parsing                          |
| `convert/`          | 19         | —                                                                          | Format converters                                 |
| `llm/`              | 10         | —                                                                          | Load lifecycle, GPU layers, mmap                  |
| `discover/`         | 7          | —                                                                          | GPU/CPU detection                                 |
| `sample/`           | 3          | —                                                                          | Sampling transforms                               |
| `ml/streaming/`     | 5          | —                                                                          | Streaming orchestrator, budget, inference         |
| `runner/`           | 2+3+1+1    | —                                                                          | Dispatch, llamarunner, ollamarunner, airllmrunner |
| `x/` (experimental) | ~50        | —                                                                          | MLX, image gen, tokenizer, models                 |

**Running tests:**

```bash
go test ./...                                          # Unit tests
go test -tags=integration ./integration -timeout 10m   # Integration
go test -tags=integration,gpu ./integration            # GPU-specific
make ship-check                                        # Full ship gate
```

---

## Documentation index

| Document                                                                 | Purpose                                                                           |
| ------------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| [`docs/DEVELOPER.md`](docs/DEVELOPER.md)                                 | **Primary reference**: layout, env vars, runners, tests, ship gate, upstream sync |
| [`docs/PRISMALAMA_PRINCIPLE.md`](docs/PRISMALAMA_PRINCIPLE.md)           | **North star**: GGML vs AirLLM, why two engines                                   |
| [`docs/RUNTIME_DISPATCH.md`](docs/RUNTIME_DISPATCH.md)                   | Which engine runs, logs, GPU usage differences                                    |
| [`docs/GOAL-GAPS.md`](docs/GOAL-GAPS.md)                                 | Shipped behavior vs product goals                                                 |
| [`docs/WEIGHT_STREAMING_STRATEGY.md`](docs/WEIGHT_STREAMING_STRATEGY.md) | Streaming architecture options and tradeoffs                                      |
| [`docs/STREAMING_BENCHMARK.md`](docs/STREAMING_BENCHMARK.md)             | Performance data on test hardware                                                 |
| [`docs/PACKAGING_DEFAULTS.md`](docs/PACKAGING_DEFAULTS.md)               | Arch package environment defaults                                                 |
| [`docs/SHIP_CHECK.md`](docs/SHIP_CHECK.md)                               | Feature → package validation                                                      |
| [`docs/TECHNICAL_DOCUMENTATION.md`](docs/TECHNICAL_DOCUMENTATION.md)     | Class diagrams, timestep diagrams, memory layout                                  |
| [`NEXT.md`](NEXT.md)                                                     | Living roadmap (Phase 1 Streaming Core)                                           |
| [`README-PKGBUILD.md`](README-PKGBUILD.md)                               | Arch package build instructions                                                   |
| [`llama/patches/README.md`](llama/patches/README.md)                     | Patch audit table + bisect policy                                                 |

---

## Quick "where do I change X?"

| Goal                             | Start here                                               |
| -------------------------------- | -------------------------------------------------------- |
| Engine dispatch (GGML vs AirLLM) | `runner/dispatch.go`                                     |
| HTTP API / routes                | `server/routes.go`                                       |
| Request scheduling / VRAM        | `server/sched.go`, `llm/gpu_layers.go`                   |
| Model loading lifecycle          | `llm/server.go`                                          |
| Capabilities endpoint            | `server/prismalama_capabilities.go`, `api/prismalama.go` |
| GPU discovery                    | `discover/`                                              |
| Layer streaming                  | `ml/streaming/`, `ml/backend/ggml/ggml_streaming.go`     |
| New model architecture           | `model/models/<name>/`                                   |
| Sampling (top-k/p/temp)          | `sample/`                                                |
| Environment defaults             | `envconfig/`                                             |
| AirLLM Python runner             | `airllm_runner.py`, `runner/airllmrunner/`               |
| Ollama-engine runner             | `runner/ollamarunner/`, `llm/ollama_engine_server.go`    |
| Upstream sync (prismallama.cpp)  | `Makefile.sync`, `llama/patches/`                        |
| Arch package                     | `PKGBUILD`, `README-PKGBUILD.md`                         |
| Docker images                    | `docker/test/`, `docker/gpu/`, `docker/arch/`            |
| MLX / Apple Silicon              | `x/mlxrunner/`, `x/imagegen/`                            |
| CLI commands                     | `cmd/`                                                   |
| Integration tests                | `integration/`                                           |

---

## Maintenance

When you add a vital package, endpoint, runner, or backend:

1. Add one line here under the matching section.
2. If the change is architectural (dispatch, streaming), update [`docs/PRISMALAMA_PRINCIPLE.md`](docs/PRISMALAMA_PRINCIPLE.md).
3. Update [`NEXT.md`](NEXT.md) if it affects the current phase.
