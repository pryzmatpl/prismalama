# Prismalama Architecture

A heterogeneous AI inference system enabling streaming from NVME on any GPU/CPU through Ollama + AirLLM + Vulkan integration.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PRISMALAMA                                     │
│  Unified Inference Platform: Ollama + AirLLM (Streaming) + Vulkan (Any GPU)│
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLIENT LAYER                                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│  │   REST API      │  │   CLI (cmd/)   │  │   Interactive Terminal    │ │
│  │   (:11434)      │  │   ollama run    │  │   chat interface          │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SERVER LAYER (llm/)                                │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      Server (llm/server.go)                         │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │  │
│  │  │  Scheduler  │  │  Model      │  │  Response   │               │  │
│  │  │  (gpu/cpu   │  │  Loader     │  │  Streamer   │               │  │
│  │  │   alloc)    │  │             │  │             │               │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘               │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐
│   CUDA BACKEND          │ │   ROCR/ROCm BACKEND    │ │   VULKAN BACKEND        │
│   (NVIDIA GPUs)         │ │   (AMD GPUs)           │ │   (ANY VULKAN-COMPAT)   │
│   ml/backend.go         │ │   ml/backend.go        │ │   ml/backend.go          │
│   - cuBLAS              │ │   - rocBLAS            │ │   - Vulkan Compute       │
│   - Flash Attention     │ │   - Flash Attention    │ │   - Cross-vendor         │
└─────────────────────────┘ └─────────────────────────┘ └─────────────────────────┘
              │                       │                       │
              └───────────────────────┼───────────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      RUNNER LAYER (runner/)                                 │
│                                                                             │
│  ┌───────────────────┐  ┌───────────────────┐  ┌────────────────────────┐  │
│  │ OllamaRunner      │  │ AirLLMRunner      │  │ LlamaRunner            │  │
│  │ (llama.cpp)       │  │ (airllm)          │  │ (legacy)               │  │
│  │ - Standard LLM    │  │ - Streaming       │  │                        │  │
│  │   inference       │  │   layers from    │  │                        │  │
│  │ - Full VRAM       │  │   NVME           │  │                        │  │
│  │   loading         │  │ - Minimal VRAM   │  │                        │  │
│  └───────────────────┘  └───────────────────┘  └────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPUTE BACKEND (ml/)                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      ml.Backend Interface                            │   │
│  │  - Tensor operations (ml/nn/)                                        │   │
│  │  - Memory management (ml/device.go)                                  │   │
│  │  - Device discovery (discover/)                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Supported Devices:                                                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │  CUDA  │  │  ROCm  │  │ Vulkan  │  │  Metal  │  │   CPU   │        │
│  │ NVIDIA │  │  AMD   │  │  Any    │  │ Apple   │  │ x86/ARM │        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Model Storage                    │  KV Cache / Working Memory      │   │
│  │ ─────────────────               │  ──────────────────────────────  │   │
│  │ • GGUF files (llama.cpp)        │  • GPU VRAM allocation            │   │
│  │ • Safetensors (HF transformers) │  • CPU RAM fallback              │   │
│  │ • AirLLM (NVME streaming)      │  • Layer-wise loading             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Server Layer (`llm/server.go`)

The server coordinates model loading, scheduling, and response streaming.

```
┌─────────────────────────────────────────────────────────────┐
│                    SERVER WORKFLOW                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Receive Request                                        │
│     └─> POST /api/generate or /api/chat                    │
│                         │                                   │
│                         ▼                                   │
│  2. Model Discovery                                        │
│     └─> Check if model loaded                               │
│                         │                                   │
│                         ▼                                   │
│  3. GPU Allocation (Scheduler)                              │
│     └─> ml/device.go:GPULayersList                         │
│     └─> Calculate memory per device                        │
│     └─> Distribute layers across GPUs                      │
│                         │                                   │
│                         ▼                                   │
│  4. Runner Selection                                       │
│     ├─> OllamaRunner (standard llama.cpp)                  │
│     ├─> AirLLMRunner (streaming from NVME)                 │
│     └─> LlamaRunner (legacy)                                │
│                         │                                   │
│                         ▼                                   │
│  5. Inference + Streaming                                   │
│     └─> SSE (Server-Sent Events) response                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2. Device Management (`ml/device.go`)

Manages heterogeneous compute device discovery and memory allocation.

```go
// Core Types
type DeviceInfo struct {
    ID           string   // Device identifier
    Library      string   // "CUDA", "ROCm", "Vulkan", "Metal", "CPU"
    Name         string   // Human-readable name
    TotalMemory  uint64   // Total VRAM/RAM
    FreeMemory   uint64   // Available for model loading
    ComputeMajor int      // Compute capability major version
    ComputeMinor int      // Compute capability minor version
}

type GPULayers struct {
    DeviceID
    Layers []int  // Layer indices to offload
}

type BackendMemory struct {
    InputWeights uint64     // Always on CPU
    CPU          DeviceMemory
    GPUs         []DeviceMemory
}
```

**Device Priority** (from `device.go:549-559`):
1. CUDA (NVIDIA) - Preferred
2. ROCm (AMD) - Preferred  
3. Vulkan - Fallback for any GPU
4. Metal - Apple GPUs
5. CPU - Universal fallback

### 3. Backend Interface (`ml/backend.go`)

Abstract tensor compute backend supporting multiple hardware vendors.

```go
type Backend interface {
    Close()
    Load(ctx context.Context, progress func(float32)) error
    BackendMemory() BackendMemory
    Config() fs.Config
    Get(name string) Tensor
    NewContext() Context
    BackendDevices() []DeviceInfo
}

type Context interface {
    Empty(dtype DType, shape ...int) Tensor
    Forward(...Tensor) Context
    Compute(...Tensor)
    // ... tensor operations
}
```

### 4. Runner Implementations

#### OllamaRunner (`runner/ollamarunner/`)
- Standard llama.cpp inference
- Full model loaded in VRAM
- Supports all GGUF formats
- Flash Attention support

#### AirLLMRunner (`runner/airllmrunner/`)
```go
type Server struct {
    modelPath  string
    port       int
    pythonCmd  *exec.Cmd      // Python airllm process
    httpClient *http.Client
}

// Streaming workflow:
// 1. Start Python airllm_runner.py
// 2. Proxy HTTP requests to Python backend
// 3. Stream tokens via SSE
```
- **Key Feature**: Streams model layers from NVME on-demand
- Minimal VRAM footprint (layers loaded dynamically)
- Python-based (air_llm library)
- Works with any backend (CUDA/ROCm/Vulkan via Python)

#### LlamaRunner (`runner/llamarunner/`)
- Legacy implementation
- Same as OllamaRunner

## Memory Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MEMORY HIERARCHY                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  VRAM (GPU)              RAM (System)          NVME (Storage)   │
│  ════════════            ═══════════           ═════════════   │
│                                                                 │
│  ┌─────────────┐        ┌─────────────┐      ┌────────────┐  │
│  │ KV Cache    │        │ Input       │      │ Full Model│  │
│  │ - Keys      │        │ Embeddings  │      │ (GGUF/     │  │
│  │ - Values    │        │             │      │  Safetensor│  │
│  └─────────────┘        └─────────────┘      └────────────┘  │
│                                                                 │
│  ┌─────────────┐        ┌─────────────┐                       │
│  │ Active      │        │ Input       │                       │
│  │ Layers      │        │ Weights     │                       │
│  │ (selected)  │        │ (CPU offload│                       │
│  └─────────────┘        └─────────────┘                       │
│                                                                 │
│  AirLLM Mode:                                                    │
│  ┌─────────────┐        ┌─────────────┐      ┌────────────┐  │
│  │ 1-2 Layers │───────▶│ Layer       │─────▶│ Model      │  │
│  │ in VRAM    │        │ Swap        │      │ Streaming  │  │
│  └─────────────┘        └─────────────┘      └────────────┘  │
│                         (active layer                            │
│                          fetching)                               │
└─────────────────────────────────────────────────────────────────┘
```

## Layer Distribution

When multiple GPUs available:

```
Model: 32 layers

GPU 0 (8GB):    [Layers 0-15]   ████████████████
GPU 1 (8GB):    [Layers 16-31]  ████████████████
CPU (fallback): [Input weights]  ████

Total: 32 layers distributed
```

From `device.go:72-102`:
```go
type GPULayersList []GPULayers

func (l GPULayersList) Sum() int  // Total layers across all GPUs
func (l GPULayersList) Hash() uint64  // Cache key for layer assignment
```

## Inference Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                     INFERENCE TIMESTEP DIAGRAM                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Time ─────────────────────────────────────────────────────────────▶  │
│                                                                      │
│  ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐             │
│  │Token N │    │Token N+1│   │Token N+2│   │Token N+3│  ...        │
│  └───┬────┘    └───┬────┘    └───┬────┘    └───┬────┘             │
│      │            │            │            │                      │
│      ▼            ▼            ▼            ▼                      │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │                    FORWARD PASS                            │    │
│  │  1. Input Embedding                                         │    │
│  │  2. For each layer:                                         │    │
│  │     ┌─────────────────────────────────────┐                │    │
│  │     │ • Attention (Q, K, V)                │                │    │
│  │     │ • RoPE positional encoding          │                │    │
│  │     │ • MLP (FFN)                          │                │    │
│  │     │ • RMS/LayerNorm                      │                │    │
│  │     └─────────────────────────────────────┘                │    │
│  │  3. Output LM Head                                          │    │
│  └────────────────────────────────────────────────────────────┘    │
│      │            │            │            │                      │
│      ▼            ▼            ▼            ▼                      │
│  ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐             │
│  │ Sampler│    │ Sampler│    │ Sampler│    │ Sampler│             │
│  │        │    │        │    │        │    │        │             │
│  │ • Temp │    │ • Temp │    │ • Temp │    │ • Temp │             │
│  │ • Top-K│    │ • Top-K│    │ • Top-K│    │ • Top-K│             │
│  │ • Top-P│    │ • Top-P│    │ • Top-P│    │ • Top-P│             │
│  └────────┘    └────────┘    └────────┘    └────────┘             │
│      │            │            │            │                      │
│      ▼            ▼            ▼            ▼                      │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │              STREAM RESPONSE (SSE)                        │    │
│  │  {"content": "word", "done": false}                       │    │
│  │  {"content": " more", "done": false}                      │    │
│  │  {"content": " words", "done": true}                       │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

## Vulkan Integration

Vulkan provides cross-vendor GPU support via compute shaders.

```
┌─────────────────────────────────────────────────────────────────┐
│                   VULKAN BACKEND PATH                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  OLLAMA_VULKAN=1                                                │
│        │                                                        │
│        ▼                                                        │
│  discover/runner.go                                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Load: libggmlvulkan.so                                   │   │
│  │ - Enumerate Vulkan devices                               │   │
│  │ - Query device capabilities                               │   │
│  │ - Set GGML_VK_VISIBLE_DEVICES                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│        │                                                        │
│        ▼                                                        │
│  ml/device.go                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ FlashAttentionSupported()                                 │   │
│  │ • Vulkan: Supported (via ggml-vulkan)                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Vendor Support:                                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ NVIDIA   │  │   AMD    │  │  Intel   │  │    Apple     │   │
│  │ (via     │  │  (via    │  │  (Xe)    │  │   (MoltenVK) │   │
│  │ Vulkan)  │  │ Vulkan)  │  │          │  │              │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Configuration** (from `envconfig/config.go`):
```go
EnableVulkan = Bool("OLLAMA_VULKAN")           // Enable/disable
VkVisibleDevices = String("GGML_VK_VISIBLE_DEVICES")  // Device selection
```

## Class Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CLASS RELATIONSHIPS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                            ┌──────────────────┐                             │
│                            │   llm.Server     │                             │
│                            │   (server.go)    │                             │
│                            └────────┬─────────┘                             │
│                                     │                                       │
│         ┌───────────────────────────┼───────────────────────────┐          │
│         │                           │                           │          │
│         ▼                           ▼                           ▼          │
│  ┌─────────────┐          ┌─────────────────┐         ┌─────────────┐      │
│  │ Scheduler  │          │  RunnerFactory  │         │  GpuAlloc  │      │
│  │             │          │                 │         │             │      │
│  │ • Select() │          │ • NewOllama()   │         │ • Calc()    │      │
│  │ • Reserve()│          │ • NewAirLLM()   │         │ • Distribute│      │
│  └─────────────┘          └────────┬────────┘         └─────────────┘      │
│                                     │                                       │
│         ┌───────────────────────────┼───────────────────────────┐          │
│         │                           │                           │          │
│         ▼                           ▼                           ▼          │
│  ┌─────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐     │
│  │ OllamaRunner    │  │   AirLLMRunner      │  │  LlamaRunner        │     │
│  │ (llama.cpp)    │  │  (airllm Python)    │  │  (legacy)          │     │
│  │                 │  │                     │  │                     │     │
│  │ • Load()        │  │ • pythonCmd         │  │ • Load()            │     │
│  │ • Predict()     │  │ • HTTP proxy        │  │ • Predict()         │     │
│  │ • Embedding()   │  │ • NVME streaming   │  │ • Embedding()       │     │
│  └────────┬────────┘  └──────────┬──────────┘  └──────────┬────────┘      │
│           │                       │                        │               │
│           └───────────────────────┼───────────────────────┘               │
│                                   ▼                                         │
│                    ┌──────────────────────────┐                             │
│                    │   ml.Backend (interface)│                             │
│                    └────────────┬─────────────┘                             │
│                                   │                                         │
│         ┌─────────────────────────┼─────────────────────────┐              │
│         │                         │                         │              │
│         ▼                         ▼                         ▼              │
│  ┌─────────────┐          ┌─────────────┐          ┌─────────────┐        │
│  │ CUDA Backend│          │ ROCm Backend│          │Vulkan Backend│       │
│  │             │          │             │          │              │        │
│  │ • cuBLAS    │          │ • rocBLAS   │          │• Vulkan SDK  │        │
│  │ • cuDNN     │          │ • FlashAttn │          │• Compute     │        │
│  └─────────────┘          └─────────────┘          │  Shaders     │        │
│                                                      └─────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## API Endpoints

```
┌─────────────────────────────────────────────────────────────────┐
│                      REST API REFERENCE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Base URL: http://localhost:11434                               │
│                                                                 │
│  POST /api/generate                                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ {                                                          │   │
│  │   "model": "llama3.2",                                    │   │
│  │   "prompt": "Explain quantum computing",                  │   │
│  │   "stream": true,                                         │   │
│  │   "options": {                                            │   │
│  │     "temperature": 0.7,                                  │   │
│  │     "num_gpu_layers": 32                                 │   │
│  │   }                                                       │   │
│  │ }                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  POST /api/chat                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ {                                                          │   │
│  │   "model": "llama3.2",                                    │   │
│  │   "messages": [                                           │   │
│  │     {"role": "user", "content": "Hello"}                 │   │
│  │   ]                                                       │   │
│  │ }                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  GET /api/tags          - List available models                 │
│  GET /api/version       - Server version                        │
│  POST /api/embeddings   - Generate embeddings                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_VULKAN` | `0` | Enable Vulkan backend |
| `GGML_VK_VISIBLE_DEVICES` | `all` | Vulkan device selection |
| `CUDA_VISIBLE_DEVICES` | `all` | CUDA device selection |
| `ROCR_VISIBLE_DEVICES` | `all` | ROCm device selection |
| `OLLAMA_GPU_LAYERS` | `auto` | Layers to offload to GPU |
| `OLLAMA_NUM_THREADS` | `auto` | CPU threads for inference |

### Model Loading Options

```go
type BackendParams struct {
    AllocMemory    bool            // Allocate or just measure
    NumThreads     int             // CPU threads
    GPULayers      GPULayersList   // GPU layer distribution
    FlashAttention FlashAttentionType  // Enable/disable
}
```

## Adding New Backends

To add a new compute backend (e.g., OpenCL):

1. **Register Backend** (`ml/backend.go:76-84`):
```go
var backends = make(map[string]func(string, BackendParams) (Backend, error))

func RegisterBackend(name string, f func(string, BackendParams) (Backend, error)) {
    backends[name] = f
}
```

2. **Implement Backend Interface** (`ml/backend.go:16-32`):
```go
type Backend interface {
    Close()
    Load(ctx context.Context, progress func(float32)) error
    BackendMemory() BackendMemory
    Config() fs.Config
    Get(name string) Tensor
    NewContext() Context
    BackendDevices() []DeviceInfo
}
```

3. **Device Discovery** (`discover/runner.go`):
```go
func (r *Runner) GetDeviceInfos(ctx context.Context) []ml.DeviceInfo {
    // Enumerate devices
    // Query memory and capabilities
}
```

4. **Build with Backend** (`CMakeLists.txt`):
```cmake
add_library(ggml${BACKEND} ...)
target_link_libraries(llama ${BACKEND}_LIBRARY)
```

## Troubleshooting

### GPU Not Detected
```bash
# Check CUDA
nvidia-smi
# Check ROCm
rocm-smi
# Check Vulkan
vulkaninfo | grep GPU
```

### Memory Issues
```bash
# Monitor VRAM
nvidia-smi -l 1  # NVIDIA
rocm-smi -d 0   # AMD

# Adjust layers
OLLAMA_GPU_LAYERS=16 ollama run model
```

### Vulkan Issues
```bash
# Enable explicitly
OLLAMA_VULKAN=1 ollama serve

# List devices
vulkaninfo
```

## Development

### Running Tests
```bash
go test ./...              # All tests
go test ./runner/...      # Runner tests
go test ./ml/...          # ML library tests
```

### Building
```bash
go build .                 # Build ollama
go build -tags airllm .    # Include AirLLM support
```

### Architecture Notes

- **AirLLM requires Python**: The runner spawns a Python subprocess (`airllm_runner.py`)
- **Vulkan is experimental**: Set `OLLAMA_VULKAN=1` to enable
- **Layer streaming**: Only AirLLMRunner supports NVME streaming
- **Cross-vendor**: Vulkan enables AMD/Intel/Apple GPUs via single codebase
