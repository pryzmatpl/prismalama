# Prismalama Technical Documentation

> Status note: this file is a long-form technical map and may lag behind active refactors.
> For current promise boundaries and runtime truth, verify against:
> `docs/PRISMALAMA_PRINCIPLE.md`, `docs/RUNTIME_DISPATCH.md`, `docs/GOAL-GAPS.md`,
> `server/prismalama_capabilities.go`, and `runner/dispatch.go`.

## Project Overview

**Prismalama** is an Ollama-compatible distribution focused on GGUF via GGML (including Vulkan/HIP
build targets) plus optional AirLLM routing for HF-style layouts. Large-model behavior depends on
engine dispatch, memory policy, and backend support for streaming interfaces.

### Key Capabilities
- Vulkan compute backend for hardware-agnostic GPU acceleration (AMD, NVIDIA, Intel)
- Multiple runner interfaces: llama.cpp (GGUF), AirLLM (weight streaming), Ollama runner
- GGUF format support with intelligent weight streaming
- Arch Linux packaging for streamlined deployment

---

## Architecture Layers

### 1. Client Layer (`cmd/`)
- REST API server on port 11434
- CLI via `ollama run` commands
- Interactive terminal chat interface

### 2. Server Layer (`llm/`, `server/`)
Core orchestration for model loading, scheduling, and response streaming.

#### Server (`llm/server.go`)
- Request handling via REST endpoints (`/api/generate`, `/api/chat`)
- Model discovery and loader coordination
- GPU allocation through scheduler
- Runner selection and dispatch
- SSE response streaming

#### Scheduler (`server/sched.go`)
- `Scheduler` struct manages request queuing and model loading
- Key functions:
  - `GetRunner()` - acquires runner for model request
  - `load()` - loads model with GPU memory estimation
  - `evict()` - removes idle runners to free VRAM
- `LlmRequest` tracks pending/in-flight requests
- VRAM recovery waits after model unload

### 3. Runner Layer (`runner/`)

#### Runner Dispatch (`runner/dispatch.go`, `DecideEngine`)
The dispatcher selects **GGML** vs **AirLLM** from directory layout and env (see **`docs/RUNTIME_DISPATCH.md`**). **`OLLAMA_USE_AIRLLM=0` / `false` / `no` is evaluated first** and forces **GGML for all layouts** (including safetensors and multipart GGUF). The Arch package ships **`OLLAMA_USE_AIRLLM=0`**.

```
DecideEngine(modelPath):
├── OLLAMA_USE_AIRLLM ∈ {0, false, no} → GGML (early exit)
├── OLLAMA_MULTI_GGUF=1 → AirLLM
├── model path missing → GGML
├── model.safetensors.index.json → AirLLM
├── *.safetensors shards → AirLLM
├── config.json HF heuristic → AirLLM
├── *-00001-of-*.gguf (multipart) → AirLLM
├── OLLAMA_USE_AIRLLM ∈ {1, true} → AirLLM
└── else → GGML (llama.cpp / GGUF)
```

#### Runners
| Runner | Path | Use Case |
|--------|------|----------|
| **LlamaRunner** | `runner/llamarunner/` | Standard GGUF via llama.cpp |
| **OllamaRunner** | `runner/ollamarunner/` | Ollama-native inference |
| **AirLLMRunner** | `runner/airllmrunner/` | NVME streaming for large models |

#### AirLLMRunner Implementation (`runner/airllmrunner/runner.go`)
- Go HTTP proxy on port N
- Spawns `airllm_runner.py` on port N+1 (Python HTTP server)
- Proxies requests to Python backend
- `pythonLoadBody()` converts LoadRequest to JSON with snake_case keys
- Flash attention enum mapping: `ml.FlashAttentionType` → Python strings

#### AirLLM Python Runner (`airllm_runner.py`)
- Ollama-compatible HTTP server
- Implements endpoints: `/load`, `/status`, `/completion`, `/embed`
- `LoadRequest`, `CompletionRequest`, `EmbeddingRequest` dataclasses
- Uses AirLLM for layer-by-layer inference

### 4. Compute Backend (`ml/`)

#### Backend Interface (`ml/backend.go`)
```go
type Backend interface {
    Close()
    Load(ctx context.Context, progress func(float32)) error
    BackendMemory() BackendMemory
    Config() fs.Config
    Get(name string) Tensor
    NewContext() Context
    NewContextSize(size int) Context
    BackendDevices() []DeviceInfo
}
```

#### Tensor Operations
Full tensor API including:
- Arithmetic: `Add`, `Sub`, `Mul`, `Div`
- Matrix: `Mulmat`, `MulmatFullPrec`, `MulmatID`
- Normalization: `LayerNorm`, `RMSNorm`
- Activation: `GELU`, `SILU`, `Tanh`, `RELU`, `Sigmoid`
- Attention: `ScaledDotProductAttention`
- Manipulation: `Reshape`, `Permute`, `Slice`, `Concat`

#### Data Types
```go
const (
    DTypeOther DType = iota
    DTypeF32
    DTypeF16
    DTypeQ80  // Q8_0 quantization
    DTypeQ40  // Q4_0 quantization
    DTypeI32
    DTypeMXFP4 // MXFP4 quantization
)
```

### 5. GGML Backend (`ml/backend/ggml/`)

#### GGML Backend (`ml/backend/ggml/ggml.go`)
- CGO bindings to `ggml.h`, `ggml-cpu.h`, `ggml-backend.h`
- Device initialization via `ggml_backend_dev_count()`, `ggml_backend_dev_get()`
- Device types: CPU, ACCEL, GPU, iGPU (integrated)
- Backend registry for GGML backend registration

### 6. Device Management (`ml/device.go`)

#### Device Priority
1. CUDA (NVIDIA) - Preferred
2. ROCm (AMD) - Preferred
3. Vulkan - Fallback for any GPU
4. Metal - Apple GPUs
5. CPU - Universal fallback

#### Key Types
```go
type DeviceInfo struct {
    ID           string
    Library      string   // "CUDA", "ROCm", "Vulkan", "Metal", "CPU"
    Name         string
    TotalMemory  uint64
    FreeMemory   uint64
    ComputeMajor int
    ComputeMinor int
}

type GPULayers struct {
    DeviceID
    Layers []int
}

type BackendMemory struct {
    InputWeights uint64
    CPU          DeviceMemory
    GPUs         []DeviceMemory
}
```

---

## Memory Management

### VRAM Allocation Flow
1. `GPULayersList` calculates memory per device
2. Distributes layers across available GPUs
3. Scheduler implements `waitForVRAMRecovery` (5s default)

### AirLLM Memory Cleanup
1. Per-layer: `AirLLMBaseModel.forward()` moves layers to `"meta"`, calls `clean_memory()`
2. Post-request: `airllm_runner.py` calls `finalize_inference_memory()`:
   - `torch.cuda.synchronize()`
   - `torch.cuda.empty_cache()`
   - `gc.collect()`
   - `malloc_trim` (where applicable)

---

## Environment Variables

| Variable | Effect |
|----------|--------|
| `OLLAMA_USE_AIRLLM` | **`0`/`false`/`no`** ⇒ GGML only (**all** layouts). **`1`/`true`** ⇒ AirLLM when combined with layout/env rules. **Unset** ⇒ `DecideEngine` heuristics may pick AirLLM (safetensors, multipart GGUF). See **`docs/RUNTIME_DISPATCH.md`**. |
| `OLLAMA_MULTI_GGUF` | `1` = treat as AirLLM-style |
| `AIRLLM_COMPRESSION` | e.g., `4bit`, `8bit`, `none` |
| `AIRLLM_DEVICE` | PyTorch device string (default `cuda:0`) |
| `PRISMALAMA_AIRLLM_PYTHONPATH` | Prepend to PYTHONPATH |
| `AIRLLM_POST_INFER_CLEANUP` | `0` = skip GPU cache flush |

---

## Integration Tests

### Test Structure (`integration/`)
- Build tag-gated: `integration`, `airllm`, `gpu`, `minimax`, `weight_streaming`, `perf`

### Test Categories
| Tag | Tests |
|-----|-------|
| `integration` | Basic API, model loading |
| `airllm` | AirLLM runner-specific |
| `gpu` | GPU utilization, VRAM |
| `minimax` | Large model (143GB+ class) |
| `weight_streaming` | Multi-part GGUF |
| `perf` | Benchmarks |

### Run Commands
```bash
# Basic
go test -tags=integration ./integration -timeout 10m

# Coverage
go test -tags=integration ./integration -coverprofile=/tmp/integration.cov

# Specific tags
go test -tags=integration,airllm ./integration
```

---

## Build System

### Build Targets
- `make ship-check` - integration tests + package build
- `make ship-check-fast` - TestBlueSky only (no packaging)
- `build-rocm.sh` - ROCm-specific package build
- `makepkg -sf` - Arch package from PKGBUILD

### Package Output
- `prismalama-ollama` - Main binary
- Installs to: `/usr/bin/ollama`, `/usr/lib/ollama/rocm`, `/usr/share/ollama/airllm`

---

## Class Diagrams

### Core Class Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SERVER LAYER                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐         ┌─────────────────────────────────────────┐    │
│  │     Server      │         │              Scheduler                   │    │
│  │  (routes.go)    │────────▶│             (sched.go)                  │    │
│  ├─────────────────┤         ├─────────────────────────────────────────┤    │
│  │ - sched         │         │ - pendingReqCh  chan *LlmRequest         │    │
│  │ - defaultNumCtx │         │ - loaded       map[string]*runnerRef     │    │
│  │ - aliases       │         │ - activeLoading llm.LlamaServer          │    │
│  └─────────────────┘         │ - loadFn                               │    │
│          │                   │ - waitForRecovery time.Duration         │    │
│          │                   └─────────────────────────────────────────┘    │
│          │                                     │                             │
│          │         ┌─────────────────────────┼─────────────────────────┐    │
│          │         │                         │                         │    │
│          ▼         ▼                         ▼                         ▼    │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
│  │   LlmRequest   │    │   runnerRef     │    │   runnerRef     │            │
│  │  (sched.go)    │    │  (sched.go)     │    │  (sched.go)     │            │
│  ├─────────────────┤    ├─────────────────┤    ├─────────────────┤            │
│  │ - ctx           │    │ - llama         │    │ - llama         │            │
│  │ - model         │    │   llm.LlamaServer│   │   llm.LlamaServer│           │
│  │ - opts          │    │ - refCount      │    │ - refCount      │            │
│  │ - successCh     │    │ - vramSize      │    │ - vramSize      │            │
│  │ - errCh         │    │ - totalSize     │    │ - totalSize     │            │
│  └─────────────────┘    │ - gpus          │    │ - gpus          │            │
│                         └─────────────────┘    └─────────────────┘            │
│                                      ▲                                      │
│                                      │                                      │
│         ┌────────────────────────────┴────────────────────────────┐         │
│         │                                                      │             │
│         ▼                                                      ▼             │
│  ┌─────────────────┐                            ┌─────────────────────────┐  │
│  │  llamaServer    │                            │      llmServer          │  │
│  │  (llm/server.go)│                           │     (llm/server.go)     │  │
│  ├─────────────────┤                            ├─────────────────────────┤  │
│  │ - llmServer     │                            │ - port     int          │  │
│  │ - llamaModel    │──────────┐                 │ - cmd      *exec.Cmd   │  │
│  │   *llama.Model  │          │                 │ - status   *StatusWriter│ │
│  └─────────────────┘          │                 │ - options  api.Options │  │
│                                │                 │ - modelPath string     │  │
│  ┌─────────────────┐          │                 │ - ggml     *ggml.GGML │  │
│  │  ollamaServer   │          │                 └─────────────────────────┘  │
│  │  (llm/server.go)│          │                                                     │
│  ├─────────────────┤          │                                                     │
│  │ - llmServer     │          │                                                     │
│  │ - tokenizer     │          │                                                     │
│  │   tokenizer.    │          │                                                     │
│  │   Tokenizer     │          │                                                     │
│  └─────────────────┘          │                                                     │
│                                │                                                     │
│                     ┌──────────┴──────────┐                                      │
│                     │                     │                                      │
│                     ▼                     ▼                                      │
│         ┌─────────────────┐    ┌─────────────────┐                             │
│         │   LlamaRunner   │    │  AirLLMRunner   │                             │
│         │ runner/llamarunner│ │runner/airllmrunner│                             │
│         │ (CGO llama.cpp) │    │ (Go HTTP Proxy)  │                             │
│         └─────────────────┘    ├─────────────────┤                             │
│                                │ - pythonCmd      │                             │
│                                │ - httpClient     │                             │
│                                │ - pythonPort    │                             │
│                                └─────────────────┘                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Backend Interface Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COMPUTE BACKEND (ml/)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐         ┌─────────────────────────────────────────┐    │
│  │     Backend     │◄─────────│            Context                      │    │
│  │  (backend.go)   │          │           (backend.go)                    │    │
│  ├─────────────────┤          ├─────────────────────────────────────────┤    │
│  │ + Close()       │          │ + Empty(dtype, shape...) Tensor         │    │
│  │ + Load(...)     │          │ + Forward(...Tensor) Context             │    │
│  │ + BackendMemory │          │ + Compute(...Tensor)                     │    │
│  │ + Get(name)     │          │ + Reserve()                              │    │
│  │ + NewContext()  │          │ + Input() Context                        │    │
│  └────────┬────────┘          └─────────────────────────────────────────┘    │
│           │                                    │                               │
│           │         ┌─────────────────────────┼─────────────────────────┐    │
│           │         │                         │                         │    │
│           ▼         ▼                         ▼                         ▼    │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
│  │  Tensor (iface) │    │    Scaled       │    │  BackendCache   │            │
│  │  (backend.go)   │    │  DotProduct     │    │    Config       │            │
│  ├─────────────────┤    │  Attention      │    │  (backend.go)   │            │
│  │ + Dim, Stride   │    │  (interface)    │    ├─────────────────┤            │
│  │ + Shape, DType  │    ├─────────────────┤    │ + CachePadding  │            │
│  │ + Mul, Add, ..  │    │ + SDPA(ctx,     │    │ + PermutedV     │            │
│  │ + Softmax,      │    │   key, value,   │    │ + MaskDType     │            │
│  │   LayerNorm,..  │    │   mask, sinks,  │    └─────────────────┘            │
│  └─────────────────┘    │   vmla, scale,  │                                  │
│                         │   cacheConfig)   │                                  │
│                         │   Tensor        │                                  │
│                         └─────────────────┘                                  │
│                                      │                                       │
│                         ┌────────────┴────────────┐                          │
│                         │                         │                          │
│                         ▼                         ▼                          │
│              ┌─────────────────┐       ┌─────────────────┐                   │
│              │   GGML Backend  │       │  Vulkan Backend │                   │
│              │ (backend/ggml/) │       │(backend/vulkan/)│                   │
│              │   CGO ggml.h   │       │   (future)     │                   │
│              └─────────────────┘       └─────────────────┘                   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      Device Management                                │   │
│  ├──────────────────────────────────────────────────────────────────────┤   │
│  │                                                                       │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐   │   │
│  │  │  DeviceInfo    │  │  GPULayersList  │  │ Heterogeneous       │   │   │
│  │  │  (device.go)   │  │   (device.go)  │  │ Scheduler           │   │   │
│  │  ├─────────────────┤  ├─────────────────┤  │(device_capability.go)│  │   │
│  │  │ - ID            │  │ - []GPULayers  │  ├─────────────────────┤   │   │
│  │  │ - Library       │  │ - Sum()        │  │ - deviceCapabilities│   │   │
│  │  │ - TotalMemory   │  │ - Hash()       │  │ - layerCosts       │   │   │
│  │  │ - FreeMemory    │  │                │  │ + ComputeOffload   │   │   │
│  │  └─────────────────┘  └─────────────────┘  │   Policy()         │   │   │
│  │                                              └─────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Request Flow Class Collaboration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         REQUEST LIFECYCLE                                    │
│                                                                              │
│    Client              Server              Scheduler          Runner/Llama   │
│       │                  │                    │                  │            │
│       │  POST /generate  │                    │                  │            │
│       │─────────────────>│                    │                  │            │
│       │                  │                    │                  │            │
│       │         ┌────────┴────────┐            │                  │            │
│       │         │scheduleRunner()│            │                  │            │
│       │         │ (routes.go:129) │            │                  │            │
│       │         └────────┬────────┘            │                  │            │
│       │                  │                    │                  │            │
│       │                  │   GetRunner()      │                  │            │
│       │                  │──────────────────>│                  │            │
│       │                  │                    │                  │            │
│       │                  │         ┌──────────┴──────────┐       │            │
│       │                  │         │   load() or        │       │            │
│       │                  │         │   findRunnerTo     │       │            │
│       │                  │         │   Unload()         │       │            │
│       │                  │         │  (sched.go:415)    │       │            │
│       │                  │         └──────────┬──────────┘       │            │
│       │                  │                    │                  │            │
│       │                  │   runnerCh         │                  │            │
│       │                  │<──────────────────│                  │            │
│       │                  │                    │                  │            │
│       │                  │         ┌──────────┴──────────┐       │            │
│       │                  │         │ llama.Load(ctx,   │       │            │
│       │                  │         │ systemInfo,gpus,  │       │            │
│       │                  │         │ requireFull)     │       │            │
│       │                  │         │ (llm/server.go)  │       │            │
│       │                  │         └──────────┬──────────┘       │            │
│       │                  │                    │       ┌───────────┴────────┐ │
│       │                  │                    │       │  StartRunner()     │ │
│       │                  │                    │       │  (llm/server.go)   │ │
│       │                  │                    │       │  spawns subprocess │ │
│       │                  │                    │       └───────────┬────────┘ │
│       │                  │                    │                  │            │
│       │                  │                    │                  │  Runner    │
│       │                  │                    │                  │  Process   │
│       │                  │                    │                  │            │
│       │                  │                    │   Completion()   │            │
│       │                  │                    │<─────────────────│            │
│       │                  │                    │                  │            │
│       │   SSE Response   │   SSE Response      │                  │            │
│       │<─────────────────│<───────────────────│                  │            │
│       │                  │                    │                  │            │
│       │                  │                    │                  │            │
└───────┴──────────────────┴────────────────────┴──────────────────┴────────────┘
```

---

## Timestep Diagrams

### Model Loading Timestep (Potato Machine with Limited VRAM)

```
┌────────────────────────────────────────────────────────────────────────────────┐
│               POTATO MACHINE MODEL LOADING (Limited VRAM)                      │
│                                                                                 │
│  Time    Component              Action                                         │
│  ────    ───────────            ─────                                         │
│                                                                                 │
│    0ms   Client                  POST /api/generate {model: "qwen2.5:3b-q4"}   │
│          │                                                                   │
│          ▼                                                                   │
│   +5ms   Server                 scheduleRunner()                               │
│          │                                                                   │
│          ▼                                                                   │
│  +10ms   Scheduler              Check: model "qwen2.5:3b-q4" in loaded?        │
│          │                                    NO                              │
│          ▼                                                                   │
│  +15ms   Scheduler              load() called                                 │
│          │                                                                   │
│          ▼                                                                   │
│  +20ms   GPU Detection          discover.GPUDevices()                        │
│          │                                                                   │
│          ▼                                                                   │
│  +25ms   Memory Estimation      GGML.Decode() - estimate layer sizes          │
│          │                                                                   │
│          ▼                                                                   │
│  +30ms   VRAM Check             GPU Free: 2GB, Model needs: 1.8GB              │
│          │                                                                   │
│          ▼                                                                   │
│  +35ms   Layer Calculation      GPULayersList: 28 layers on GPU, 4 on CPU    │
│          │                                                                   │
│          ▼                                                                   │
│  +40ms   StartRunner()          Spawn llama runner subprocess                │
│          │                                                                   │
│          ▼                                                                   │
│  +50ms   Runner Process          ggml_backend_init() - Vulkan/CPU fallback   │
│          │                                                                   │
│          ▼                                                                   │
│  +80ms   Weight Loading         mmap GGUF file - stream quantized weights     │
│          │                                                                   │
│          ▼                                                                   │
│ +200ms   KV Cache Alloc         backend.Alloc() - 512MB for context          │
│          │                                                                   │
│          ▼                                                                   │
│ +220ms   Runner Ready            Ping() succeeds                              │
│          │                                                                   │
│          ▼                                                                   │
│ +250ms   RunnerRef returned     Success channel sent to scheduler              │
│          │                                                                   │
│          ▼                                                                   │
│ +255ms   Scheduler              runnerRef added to loaded{} map              │
│          │                                                                   │
│          ▼                                                                   │
│ +260ms   Completion()           First token generated                        │
│          │                                                                   │
│          ▼                                                                   │
│ +280ms   SSE Response           {"content": "Hello"} stream to client        │
│          │                                                                   │
│          ▼                                                                   │
│ +500ms   Done                   {"done": true}                               │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Multi-Model Eviction Timestep (VRAM Pressure)

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-MODEL EVICTION TIMESTEP                               │
│                                                                                 │
│  Time    Component              Action                                         │
│  ────    ───────────            ─────                                         │
│                                                                                 │
│    0ms   Request B              POST /api/generate {model: "llama3:8b-q4"}     │
│          │                                                                   │
│          ▼                                                                   │
│   +5ms   ScheduleRunner B      Check VRAM: 2GB free, model needs 2.5GB        │
│          │                                                                   │
│          ▼                                                                   │
│  +10ms   findRunnerToUnload()   Scan loaded{} - find idle runner            │
│          │                                                                   │
│          ▼                                                                   │
│  +15ms   Runner A located       refCount = 0 (idle)                          │
│          │                                                                   │
│          ▼                                                                   │
│  +20ms   waitForVRAMRecovery()  Establish baseline: free=2GB, total=8GB    │
│          │                                                                   │
│          ▼                                                                   │
│  +25ms   Runner A.unload()      llama.Close() - release VRAM                │
│          │                                                                   │
│          ▼                                                                   │
│  +30ms   Poll GPU              free=3GB (not enough yet)                      │
│          │                                                                   │
│          ▼                                                                   │
│  +35ms   Poll GPU              free=4.5GB (~75% recovered)                   │
│          │                                                                   │
│          ▼                                                                   │
│  +37ms   Recovery complete     waitForVRAMRecovery returns                  │
│          │                                                                   │
│          ▼                                                                   │
│  +40ms   Load Model B          llama.Load() with full GPU offload           │
│          │                                                                   │
│          ▼                                                                   │
│  +80ms   Runner B Ready         Completion() begins                          │
│          │                                                                   │
│          ▼                                                                   │
│ +100ms   SSE Stream             Tokens flowing to client                     │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

### AirLLM Weight Streaming Timestep (Large Model)

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                    AIRLLM WEIGHT STREAMING TIMESTEP                            │
│                    (143GB Model on 24GB VRAM)                                  │
│                                                                                 │
│  Time    Component              Action                                         │
│  ────    ───────────            ─────                                         │
│                                                                                 │
│    0ms   Request                POST /api/generate {model: "minimax2.5:q4"}     │
│          │                                                                   │
│          ▼                                                                   │
│   +5ms   Runner Selection       airLLMModelAndReason() = true                  │
│          │                        (multipart GGUF detected)                  │
│          ▼                                                                   │
│  +10ms   AirLLMRunner.Start()   Spawn airllm_runner.py on port+1             │
│          │                                                                   │
│          ▼                                                                   │
│  +20ms   Python Runner          from_pretrained(AirLLM)                      │
│          │                                                                   │
│          ▼                                                                   │
│ +500ms   Layer 0 Load           torch.load(shard_0) → GPU                     │
│          │                                                                   │
│          ▼                                                                   │
│  +30ms   Forward Pass          Layer 0 computation                           │
│          │                                                                   │
│          ▼                                                                   │
│  +40ms   Layer 0 → Meta         layer.to("meta") + clean_memory()            │
│          │                                                                   │
│          ▼                                                                   │
│  +50ms   Layer 1 Load           torch.load(shard_1) → GPU                     │
│          │                                                                   │
│          ▼                                                                   │
│  +60ms   Forward Pass          Layer 1 computation                           │
│          │                                                                   │
│          ▼                                                                   │
│  +70ms   Layer 1 → Meta         layer.to("meta") + clean_memory()            │
│          │                                                                   │
│          ▼                                                                   │
│         ...                  (repeat for all 80 layers)                      │
│          │                                                                   │
│          ▼                                                                   │
│+2000ms   Final Layer           Layer 79 → output                              │
│          │                                                                   │
│          ▼                                                                   │
│+2100ms   Sampling               token = sample(logits)                         │
│          │                                                                   │
│          ▼                                                                   │
│+2110ms   SSE Token              {"content": "token"} stream                   │
│          │                                                                   │
│          ▼                                                                   │
│+2200ms   finalize_inference     torch.cuda.empty_cache()                     │
│          │                        gc.collect()                               │
│          ▼                                                                   │
│         ...                  (next token iteration)                           │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Hardware-Level Operations

### GPU Memory Hierarchy (Potato Machine Target)

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                    POTATO MACHINE MEMORY LAYOUT                               │
│                                                                                 │
│  ╔═══════════════════════════════════════════════════════════════════════════╗ │
│  ║                         SYSTEM RAM (16GB)                                   ║ │
│  ║  ┌─────────────────────────────────────────────────────────────────────┐   ║ │
│  ║  │ OS + Applications                              │  ~4GB              │   ║ │
│  ║  ├─────────────────────────────────────────────────────────────────────┤   ║ │
│  ║  │ GGML Backend (CPU fallback layers)            │  ~2GB              │   ║ │
│  ║  ├─────────────────────────────────────────────────────────────────────┤   ║ │
│  ║  │ Input/Output Buffers                          │  ~1GB              │   ║ │
│  ║  ├─────────────────────────────────────────────────────────────────────┤   ║ │
│  ║  │ KV Cache (CPU portion, if needed)             │  ~2GB              │   ║ │
│  ║  ├─────────────────────────────────────────────────────────────────────┤   ║ │
│  ║  │ Model Quantized Weights (CPU offload)         │  ~7GB              │   ║ │
│  ║  └─────────────────────────────────────────────────────────────────────┘   ║ │
│  ╚═══════════════════════════════════════════════════════════════════════════╝ │
│                                                                                 │
│  ╔═══════════════════════════════════════════════════════════════════════════╗ │
│  ║                         VRAM (4-8GB typical)                               ║ │
│  ║  ┌─────────────────────────────────────────────────────────────────────┐   ║ │
│  ║  │ GPU Overhead (driver, context)                   │  ~500MB         │   ║ │
│  ║  ├─────────────────────────────────────────────────────────────────────┤   ║ │
│  ║  │ Attention KV Cache                                 │  ~1-2GB       │   ║ │
│  ║  ├─────────────────────────────────────────────────────────────────────┤   ║ │
│  ║  │ Active Layer Weights (current forward pass)        │  ~500MB-1GB   │   ║ │
│  ║  ├─────────────────────────────────────────────────────────────────────┤   ║ │
│  ║  │ Partial Model Weights (GPU-offloaded layers)     │  ~2-4GB       │   ║ │
│  ║  └─────────────────────────────────────────────────────────────────────┘   ║ │
│  ╚═══════════════════════════════════════════════════════════════════════════╝ │
│                                                                                 │
│  ╔═══════════════════════════════════════════════════════════════════════════╗ │
│  ║                         NVME/STORAGE                                       ║ │
│  ║  ┌─────────────────────────────────────────────────────────────────────┐   ║ │
│  ║  │ GGUF Model File (quantized)                        │  4-8GB       │   ║ │
│  ║  ├─────────────────────────────────────────────────────────────────────┤   ║ │
│  ║  │ mmap'd sections streamed on demand                                   │   ║ │
│  ║  └─────────────────────────────────────────────────────────────────────┘   ║ │
│  ╚═══════════════════════════════════════════════════════════════════════════╝ │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Vulkan Compute Pipeline (Cross-Vendor GPU)

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                    VULKAN COMPUTE PIPELINE                                     │
│                         (Hardware-Agnostic)                                    │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                         APPLICATION LAYER                                 │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐             │  │
│  │  │  Tensor::Mulmat │  │ Attention::SDPA │  │  Layer::Forward │             │  │
│  │  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘             │  │
│  └──────────┼──────────────────┼──────────────────┼──────────────────────┘  │
│             │                  │                  │                          │
│             ▼                  ▼                  ▼                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                      GGML VULKAN BACKEND                                 │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐             │  │
│  │  │ vkCreateShaderModule│ │ vkAllocateBuffer │ │  vkQueueSubmit │             │  │
│  │  │ (compute.spv)  │  │ (weight tensor) │  │  (compute cmd) │             │  │
│  │  └────────────────┘  └────────────────┘  └────────────────┘             │  │
│  │                                                                          │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐ │  │
│  │  │                    VULKAN DEVICE                                   │ │  │
│  │  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐   │ │  │
│  │  │  │  NVIDIA   │  │    AMD     │  │   INTEL    │  │    MESA    │   │ │  │
│  │  │  │  (PROP)   │  │   (RADV)   │  │  (ANV)     │  │  (Lavapi)  │   │ │  │
│  │  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘   │ │  │
│  │  │       │               │               │               │          │ │  │
│  │  │       └───────────────┴───────────────┴───────────────┘          │ │  │
│  │  │                           │                                      │ │  │
│  │  │                    ┌──────┴──────┐                               │ │  │
│  │  │                    │  Compute    │                               │ │  │
│  │  │                    │  Queue      │                               │ │  │
│  │  │                    └─────────────┘                               │ │  │
│  │  └────────────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Weight Streaming Data Flow (AirLLM NVME → GPU)

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                    AIRLLM WEIGHT STREAMING DATA FLOW                           │
│                    (Layer-by-Layer NVME → GPU Transfer)                        │
│                                                                                 │
│  NVME Storage                    VRAM                        GPU Compute       │
│  ───────────                     ────                        ───────────       │
│                                                                                 │
│  ┌─────────────────┐            ┌─────────────────┐      ┌────────────────┐  │
│  │ model-00001.gguf│            │                  │      │                │  │
│  │ ├─ Layer 0 (w0) │───────────▶│  Layer 0 Weights │      │                │  │
│  │ ├─ Layer 1 (w1) │            │  [ACTIVE]        │──────│──▶ Forward()   │  │
│  │ ├─ Layer 2 (w2) │            │                  │      │                │  │
│  │ ├─ Layer 3 (w3) │            ├─────────────────┤      └────────────────┘  │
│  │ ├─ ...          │            │  Layer 1 Weights │                           │
│  │ ├─ Layer N      │───────────▶│  [LOADING...]    │                           │
│  │ └─ KV Cache     │            ├─────────────────┤      ┌────────────────┐  │
│  └─────────────────┘            │  Layer 2 Weights │      │                │  │
│         │                      │  [META/DISCARD]  │      │   Compute      │  │
│         │                      ├─────────────────┤      │   in flight    │  │
│         │                      │                  │      │                │  │
│         │                      │  KV Cache        │◀─────│── Layer Output │  │
│         │                      │  (persistent)    │      └────────────────┘  │
│         │                      └─────────────────┘                           │
│         │                                                                     │
│         │                      ┌─────────────────┐      ┌────────────────┐  │
│         │                      │                 │      │                │  │
│         └─────────────────────▶│  PCIe Transfer  │──────│  Bandwidth    │  │
│           (mmap on demand)     │  ~10-20 GB/s    │      │  ~10-20 GB/s  │  │
│                                │  (PCIe 4.0 x4)  │      │                │  │
│                                └─────────────────┘      └────────────────┘  │
│                                                                                │
│  TIMELINE:                                                                   │
│  ────────                                                                   │
│                                                                                │
│  Layer N:                                                                    │
│  ├─ T+0ms:    Layer N weights loaded via mmap                                │
│  ├─ T+5ms:    torch.load() to GPU (async)                                   │
│  ├─ T+10ms:   Forward pass compute                                          │
│  ├─ T+25ms:   Layer output → KV cache                                       │
│  ├─ T+26ms:   Layer N weights → "meta" (offload)                            │
│  ├─ T+27ms:   clean_memory()                                                │
│  └─ T+28ms:   Layer N+1 weights loading...                                  │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Class Inflection Points

### Critical Polymorphism Points

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                         CLASS INFLECTION POINTS                                 │
│                    (Where Behavior Changes Dramatically)                         │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  1. Runner Selection (runner/runner.go:17-72)                                 │
│     ┌────────────────────────────────────────────────────────────────────────┐ │
│     │ func airLLMModelAndReason(modelPath string) (ok bool, reason string)  │ │
│     │                                                                        │ │
│     │   // INFLECTION: Single boolean return controls entire execution path  │ │
│     │   // - true  → AirLLMRunner (Python subprocess, NVME streaming)       │ │
│     │   // - false → LlamaRunner (CGO llama.cpp, full VRAM load)            │ │
│     │                                                                        │ │
│     │   if os.Getenv("OLLAMA_USE_AIRLLM") == "0" {                          │ │
│     │       return false, ""  // ← Path divergence point                   │ │
│     │   }                                                                    │ │
│     │   // ... heuristics ...                                               │ │
│     │   return true, reason   // ← Path divergence point                   │ │
│     └────────────────────────────────────────────────────────────────────────┘ │
│                                                                                │
│  2. Backend Instantiation (ml/backend.go:86-92)                               │
│     ┌────────────────────────────────────────────────────────────────────────┐ │
│     │ func NewBackend(modelPath string, params BackendParams) Backend {     │ │
│     │                                                                        │ │
│     │   // INFLECTION: Backend registry lookup                              │ │
│     │   // Currently hardcoded to "ggml" - future: Vulkan, CUDA, ROCm       │ │
│     │                                                                        │ │
│     │   if backend, ok := backends["ggml"]; ok {                            │ │
│     │       return backend(modelPath, params)  // ← Gateway to all ML       │ │
│     │   }                                                                    │ │
│     │   return nil, fmt.Errorf("unsupported backend")                       │ │
│     └────────────────────────────────────────────────────────────────────────┘ │
│                                                                                │
│  3. Scheduler Load Decision (server/sched.go:414-514)                         │
│     ┌────────────────────────────────────────────────────────────────────────┐ │
│     │ func (s *Scheduler) load(req *LlmRequest, ...) bool {                  │ │
│     │                                                                        │ │
│     │   // INFLECTION: requireFull determines GPU-only vs CPU fallback      │ │
│     │                                                                        │ │
│     │   gpuIDs, err := llama.Load(ctx, systemInfo, gpus, requireFull)      │ │
│     │   if err != nil {                                                    │ │
│     │       if errors.Is(err, ErrLoadRequiredFull) {                       │ │
│     │           // ← CRITICAL: Model too large, needs eviction             │ │
│     │           s.activeLoading.Close()                                    │ │
│     │           s.activeLoading = nil                                      │ │
│     │           return true  // Signal: need to evict first               │ │
│     │       }                                                               │ │
│     │   }                                                                   │ │
│     │   return false  // Load successful                                    │ │
│     └────────────────────────────────────────────────────────────────────────┘ │
│                                                                                │
│  4. Device Priority Selection (ml/device.go)                                  │
│     ┌────────────────────────────────────────────────────────────────────────┐ │
│     │ Device Priority Order:                                                │ │
│     │   1. CUDA    → NVIDIA GPUs (preferred)                               │ │
│     │   2. ROCm    → AMD GPUs (preferred)                                  │ │
│     │   3. Vulkan  → Cross-vendor fallback (potato machine hero!)          │ │
│     │   4. Metal   → Apple GPUs                                            │ │
│     │   5. CPU     → Universal fallback                                    │ │
│     │                                                                        │ │
│     │ // INFLECTION: First available GPU wins                               │ │
│     │ // Potato machine with Intel iGPU + AMD dGPU:                        │ │
│     │ //   - ROCm finds AMD dGPU → selected                               │ │
│     │ //   - Intel iGPU ignored (Vulkan could use either)                  │ │
│     └────────────────────────────────────────────────────────────────────────┘ │
│                                                                                │
│  5. VRAM Recovery Threshold (server/sched.go:802-803)                        │
│     ┌────────────────────────────────────────────────────────────────────────┐ │
│     │ // INFLECTION: 75% recovery threshold                               │ │
│     │ // Too low: subsequent model loads may fail                          │ │
│     │ // Too high: waiting for last 25% wastes time                        │ │
│     │                                                                        │ │
│     │ if float32(freeMemoryNow-freeMemoryBefore) > float32(runner.vramSize)*0.75 { │
│     │     finished <- struct{}{}  // ← Proceed with new load               │ │
│     │ }                                                                      │ │
│     └────────────────────────────────────────────────────────────────────────┘ │
│                                                                                │
│  6. AirLLM Memory Cleanup Trigger (airllm_runner.py)                          │
│     ┌────────────────────────────────────────────────────────────────────────┐ │
│     │ # Per-layer cleanup (AirLLM native)                                   │ │
│     │ layer.to("meta")          # Move to meta device (tiny memory)        │ │
│     │ clean_memory()            # AirLLM's memory cleanup                   │ │
│     │                                                                        │ │
│     │ # Post-request cleanup (Prismalama addition)                         │ │
│     │ if os.getenv("AIRLLM_POST_INFER_CLEANUP", "1") != "0":              │ │
│     │     torch.cuda.empty_cache()  # ← Trigger explicit VRAM return     │ │
│     │     gc.collect()             # ← Python GC sweep                     │ │
│     │     malloc_trim()            # ← glibc heap trim                      │ │
│     └────────────────────────────────────────────────────────────────────────┘ │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Critical Code Paths for Potato Machine

### Path 1: Request → Inference (Happy Path)

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                    HAPPY PATH: Inference on Potato                              │
│                                                                                 │
│  Entry: POST /api/generate {model: "qwen2.5:3b-q4", prompt: "..."}            │
│                                                                                 │
│  server/routes.go:174           GenerateHandler()                              │
│       │                                                                     │
│       ▼                                                                     │
│  server/routes.go:129           scheduleRunner()                              │
│       │                                                                     │
│       ├─► Check model exists                                                   │
│       │                                                                     │
│       ▼                                                                     │
│  server/sched.go:86             GetRunner()                                  │
│       │                                                                     │
│       ├─► Check if model already loaded                                       │
│       │        │                                                              │
│       │        ├─ YES: runnerRef returned immediately                         │
│       │        │                                                              │
│       │        └─ NO: proceed to load                                         │
│       │                                                                     │
│       ▼                                                                     │
│  server/sched.go:415           load()                                        │
│       │                                                                     │
│       ├─► GGML.Decode() - parse GGUF metadata                                │
│       │                                                                     │
│       ├─► GPU detection (discover.GPUDevices)                                 │
│       │                                                                     │
│       ├─► Memory estimation (GPULayersList)                                   │
│       │        │                                                              │
│       │        └─► "3B Q4 model needs 1.8GB, 4GB VRAM available"             │
│       │                                                                     │
│       ▼                                                                     │
│  llm/server.go:270             StartRunner()                                   │
│       │                                                                     │
│       ├─► Set GPU_VISIBLE_DEVICES env                                        │
│       ├─► Spawn llama runner subprocess                                       │
│       └─► Wait for runner to become ready                                    │
│       │                                                                     │
│       ▼                                                                     │
│  llm/server.go:496             llama.Load()                                   │
│       │                                                                     │
│       ├─► Allocate layer buffers                                              │
│       ├─► mmap GGUF weights                                                   │
│       └─► Initialize KV cache                                                 │
│       │                                                                     │
│       ▼                                                                     │
│  runner/llamarunner/           Completion()                                    │
│       │                                                                     │
│       ├─► Tokenize prompt                                                     │
│       ├─► Forward pass (GPU layers)                                           │
│       ├─► Forward pass (CPU fallback layers)                                 │
│       ├─► Sample next token                                                  │
│       └─► Stream token via SSE                                               │
│       │                                                                     │
│       ▼                                                                     │
│  Exit: SSE stream {"content": "Hello"} ... {"done": true}                    │
│                                                                                 │
│  POTATO OPTIMIZATIONS:                                                         │
│  ├─ Quantized weights (Q4_0) reduce memory 4x                                 │
│  ├─ Partial GPU offload (28 layers GPU, 4 CPU)                               │
│  ├─ Vulkan backend for Intel iGPU support                                    │
│  └─ Paged attention for efficient KV cache                                   │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Path 2: Model Too Large → Eviction → Reload

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                    EVICTION PATH: Model Doesn't Fit                            │
│                                                                                 │
│  Entry: POST /api/generate {model: "llama3:8b", prompt: "..."}                │
│        (llama3:8b needs 5GB, only 2GB VRAM free)                              │
│                                                                                 │
│  server/sched.go:415           load()                                         │
│       │                                                                     │
│       ├─► Memory estimation: 5GB needed, 2GB available                        │
│       │                                                                     │
│       ▼                                                                     │
│  llm/server.go:496             llama.Load() returns ErrLoadRequiredFull       │
│       │                                                                     │
│       ▼                                                                     │
│  server/sched.go:498           if errors.Is(err, ErrLoadRequiredFull)        │
│       │                                                                     │
│       ├─► s.activeLoading.Close() - close current model                       │
│       │                                                                     │
│       ├─► s.activeLoading = nil                                             │
│       │                                                                     │
│       └─► return true - signal need to evict                                  │
│       │                                                                     │
│       ▼                                                                     │
│  server/sched.go:375           waitForVRAMRecovery()                          │
│       │                                                                     │
│       ├─► Establish baseline: free=2GB                                        │
│       │                                                                     │
│       ├─► Poll every 250ms                                                   │
│       │                                                                     │
│       ├─► Wait until 75% of model VRAM recovered                             │
│       │                                                                     │
│       └─► Timeout after 5s (waitForRecovery default)                         │
│       │                                                                     │
│       ▼                                                                     │
│  server/sched.go:220           Load new model (llama3:8b)                     │
│       │                                                                     │
│       ├─► VRAM now available                                                 │
│       │                                                                     │
│       └─► Load succeeds                                                      │
│       │                                                                     │
│       ▼                                                                     │
│  Completion() → SSE stream                                                   │
│                                                                                 │
│  POTATO INSIGHT:                                                               │
│  ├─ Potato machines will trigger this path frequently                         │
│  ├─ waitForVRAMRecovery prevents "VRAM thrashing"                             │
│  └─ 5s timeout prevents indefinite waiting                                   │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Path 3: AirLLM Long Context (Streaming Large Models)

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                    AIRLLM PATH: Large Model Weight Streaming                   │
│                                                                                │
│  Entry: POST /api/generate {model: "minimax2.5:q4", prompt: "..."}            │
│        (MiniMax 2.5 has 222 layers, 143GB total, Q4 quantized)                │
│                                                                                │
│  runner/runner.go:107          airLLMModelAndReason() = true                   │
│       │                        (multipart GGUF detected)                        │
│       │                                                                     │
│       ▼                                                                     │
│  runner/airllmrunner/runner.go  airllmrunner.Execute()                        │
│       │                                                                     │
│       ├─► Start Python subprocess                                             │
│       ├─► pythonPort = port + 1                                               │
│       ├─► httpClient to localhost:pythonPort                                   │
│       │                                                                     │
│       ▼                                                                     │
│  airllm_runner.py             LoadRequest → AirLLM                            │
│       │                                                                     │
│       ├─► AutoModel.from_pretrained(model_path, ...)                        │
│       │                                                                     │
│       ├─► Layer 0 loaded to GPU                                               │
│       │                                                                     │
│       ▼                                                                     │
│  airllm_runner.py             Completion (token generation)                   │
│       │                                                                     │
│       │   for each token:                                                    │
│       │   ┌─────────────────────────────────────────────────────────────┐   │
│       │   │ Layer 0: layer.to("cuda"); output = layer(input)            │   │
│       │   │ Layer 0: layer.to("meta"); clean_memory()                   │   │
│       │   │ Layer 1: layer.to("cuda"); output = layer(input)            │   │
│       │   │ Layer 1: layer.to("meta"); clean_memory()                   │   │
│       │   │ ... (repeat for all 222 layers)                            │   │
│       │   │                                                              │   │
│       │   │ KV cache persists across layers (stays on GPU)             │   │
│       │   └─────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  Output: SSE stream of tokens (slower due to layer streaming)                  │
│                                                                                │
│  POTATO INSIGHT:                                                               │
│  ├─ AirLLM is THE path for models larger than VRAM                          │
│  ├─ Each token requires full layer-by-layer traversal                         │
│  ├─ KV cache stays on GPU (doesn't stream)                                    │
│  ├─ Throughput: ~5-10 tokens/sec on fast NVME                               │
│  └─ Without AirLLM: OOM on first forward pass                                │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Path 4: Quantized Model Fallback (Minimum Memory)

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                    FALLBACK PATH: CPU + Minimal VRAM                            │
│                                                                                │
│  Entry: POST /api/generate {model: "phi3:3.8b-q4"}                           │
│        (phi3 needs 2.2GB, system has 4GB VRAM with other allocations)         │
│                                                                                │
│  llm/server.go:496             llama.Load()                                   │
│       │                                                                     │
│       ├─► GPU available: 500MB (too small for full model)                    │
│       │                                                                     │
│       ├─► GPU layers = 0 (fallback to CPU)                                  │
│       │                                                                     │
│       ├─► Check: can model fit in CPU + remaining VRAM?                      │
│       │                                                                     │
│       ▼                                                                     │
│  llama runner (subprocess)     Full CPU execution                             │
│       │                                                                     │
│       ├─► GGML with CPU backend                                             │
│       ├─► Threads = CPU cores (envconfig.ThreadCount)                         │
│       │                                                                     │
│       └─► Weights stay in system RAM (via mmap)                             │
│       │                                                                     │
│       ▼                                                                     │
│  Performance: ~1-3 tokens/sec (CPU-only)                                     │
│                                                                                │
│  POTATO INSIGHT:                                                               │
│  ├─ Q4 quantization essential for potato machines                            │
│  ├─ phi3:3.8b-q4 = 2.2GB vs 7.9GB fp16 (72% reduction)                      │
│  ├─ CPU fallback works but is slow                                           │
│  └─ Vulkan backend helps iGPU systems                                        │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## File Structure Summary

```
prismalama/
├── cmd/              # CLI entry points
├── llm/              # Ollama-compatible server
├── runner/           # Runner dispatch + implementations
│   ├── runner.go              # Dispatch logic
│   ├── llamarunner/           # llama.cpp GGUF
│   ├── ollamarunner/         # Ollama native
│   └── airllmrunner/         # AirLLM proxy (Go→Python)
├── server/           # HTTP routes, scheduler
├── ml/               # ML backend interfaces
│   ├── backend.go            # Backend, Tensor, Context interfaces
│   ├── backend/ggml/         # GGML/CGO bindings
│   ├── device.go             # GPU discovery, memory
│   └── nn/                   # Neural network ops
├── airllm_runner.py  # Python AirLLM HTTP server
├── src/airllm/       # Vendored AirLLM Python package
├── llama/            # prismallama.cpp (GGUF engine)
├── integration/      # Test suite
├── docs/             # Developer docs
├── PKGBUILD          # Arch package definition
└── Makefile.sync     # Engine sync from piotroxp/prismallama.cpp
```

---

## Engine Source

**prismallama.cpp** (https://github.com/piotroxp/prismallama.cpp) is the maintained GGUF/llama.cpp fork. Sync into this repo via `Makefile.sync` (sets `FETCH_HEAD` commit).

---

## References
- `docs/DEVELOPER.md` - Developer guide, runner selection, env vars
- `docs/RUNTIME_DISPATCH.md` - Which runner handles which model
- `docs/WEIGHT_STREAMING_STRATEGY.md` - Streaming implementation details
- `README-PKGBUILD.md` - Package build instructions
- `ARCHITECTURE.md` - Visual diagrams
