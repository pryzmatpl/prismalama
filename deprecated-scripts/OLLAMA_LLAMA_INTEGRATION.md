# Kimi K2.5 Integration: Ollama + prismalama + llama.cpp

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Layer                               │
│  opencode → Ollama API → HTTP requests                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Ollama Server                               │
│  - API endpoints (/api/generate, /api/chat)                     │
│  - Model management                                            │
│  - Routes requests to appropriate runner                       │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
    ┌─────────────────┐ ┌──────────┐ ┌─────────────────┐
    │  llamarunner    │ │airllm-   │ │ ollamarunner    │
    │  (llama.cpp)    │ │ runner   │ │ (default)       │
    └────────┬────────┘ └────┬─────┘ └────────┬────────┘
             │               │                │
             ▼               ▼                ▼
    ┌──────────────────────────────────────────────────────┐
    │              llama.cpp Engine                       │
    │  - GGUF format support                             │
    │  - ROCm GPU acceleration                           │
    │  - Layer offloading                                │
    │  - Multi-file GGUF (shards)                        │
    └──────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Model Storage                                 │
│  - Standard: /var/lib/ollama/blobs/ (copied)                   │
│  - Kimi: /nvme3/AI Models/Kimi/ (579GB, symlinks)              │
│  - MiniMax: /sda2/MiniMaxAi_MiniMax-M2.1-GGUF/ (138GB)         │
└─────────────────────────────────────────────────────────────────┘
```

## How prismalama Extends Ollama

### 1. Custom Runners
Located in `/sda2/prismalama/runner/`:
- **`llamarunner/`** - Standard llama.cpp (what Ollama uses)
- **`airllmrunner/`** - AirLLM integration for huge models
- **`ollamarunner/`** - Default Ollama runner

### 2. Model Detection (`runner/runner.go`)
```go
func isAirLLMModel(modelPath string) bool {
    // Detects safetensors or transformers format
    // Falls back to OLLAMA_USE_AIRLLM env var
}
```

### 3. Integration Points
- **prismalama** adds AirLLM support to Ollama's runner selection
- **Ollama** provides the API server and model management
- **llama.cpp** does the actual inference (always used)

## Three Ways to Use Kimi

### Option 1: Symlink Method (Recommended) ✅
**Leverages Ollama + prismalama + llama.cpp without copying**

```bash
# Run this with sudo:
sudo bash /sda2/add-kimi-to-ollama.sh

# Then use normally:
ollama run kimi-k2.5 'Hello!'
opencode run -m ollama/kimi-k2.5
```

**How it works:**
1. Calculate sha256 hash of Kimi GGUF files
2. Create symlinks in `/var/lib/ollama/blobs/`
3. Create manifest pointing to symlinks
4. Ollama's llamarunner loads via llama.cpp
5. Model stays on NVMe, layers loaded to GPU

**Pros:**
- ✅ Uses existing Ollama infrastructure
- ✅ Works with opencode
- ✅ No 579GB copy needed
- ✅ ROCm GPU acceleration
- ✅ API server included

### Option 2: Direct llama.cpp (Simplest)
**Bypass Ollama, use llama.cpp directly**

```bash
# Install llama-cpp-python with ROCm
bash /sda2/install-llama-cpp.sh

# Run directly
python3 /sda2/kimi-direct.py 'Hello!'
```

**How it works:**
- Python bindings directly call llama.cpp
- No Ollama overhead
- Direct file access to GGUF

**Pros:**
- ✅ Fastest setup
- ✅ No manifest/symlink management
- ✅ Full control over parameters

**Cons:**
- ❌ No API server
- ❌ No opencode integration
- ❌ Manual model management

### Option 3: AirLLM Integration (Memory Optimized)
**Use prismalama's AirLLM runner for layer offloading**

```bash
# Set environment
export OLLAMA_USE_AIRLLM=1
export AIRLLM_DEVICE=cuda:0

# Would work if dependencies installed:
# ollama run /sda2/airllm/kimi-k2.5 'Hello!'
```

**How it works:**
- airllmrunner loads model with 4-bit compression
- Layers loaded on-demand
- Minimal GPU memory usage

**Pros:**
- ✅ Best for very limited GPU memory
- ✅ 579GB → ~8GB GPU usage

**Cons:**
- ❌ Requires Python dependencies
- ❌ Slower inference
- ❌ Complex setup

## Recommended Approach: Option 1 (Symlinks)

This is the **best of all worlds**:

1. **piggybacks on existing software**:
   - Ollama (API, management)
   - prismalama (ROCm support, runners)
   - llama.cpp (inference engine)

2. **zero disk space overhead**:
   - Only symlinks created (~100 bytes)
   - 579GB stays on NVMe

3. **full feature set**:
   - HTTP API
   - Model management
   - opencode integration
   - ROCm acceleration
   - Multi-user support

## Quick Setup

```bash
# 1. Run the setup script (creates symlinks + manifest)
sudo bash /sda2/add-kimi-to-ollama.sh

# 2. Verify model is available
ollama list | grep kimi

# 3. Run inference
ollama run kimi-k2.5 'Explain quantum computing'

# 4. Use with opencode
opencode run -m ollama/kimi-k2.5
```

## Technical Details

### Blob Structure
```
/var/lib/ollama/blobs/
├── sha256-abc123...  → /nvme3/AI Models/Kimi/Kimi-K2.5-Q4_K_M-00001-of-00013.gguf (SYMLINK)
├── sha256-def456...  → /sda2/MiniMaxAi_MiniMax-M2.1-GGUF/... (SYMLINK)
└── sha256-ghi789...  → /var/lib/ollama/models/qwen2.5-coder/ (COPIED)
```

### Manifest Structure
```json
{
  "schemaVersion": 2,
  "layers": [
    {
      "mediaType": "application/vnd.ollama.image.model",
      "digest": "sha256-abc123...",
      "size": 45000000000,
      "from": "/nvme3/AI Models/Kimi/..."
    }
  ]
}
```

### Runner Selection Flow
```
ollama run kimi-k2.5
    ↓
Ollama Server
    ↓
Detect model type → llamarunner (GGUF)
    ↓
llamarunner.Execute()
    ↓
llama.cpp llama_load_model()
    ↓
Load GGUF from symlink location
    ↓
ROCm GPU inference
```

## Summary

**Yes!** We can absolutely piggyback on existing software:

1. **Ollama** provides the API and management layer
2. **prismalama** adds ROCm support and custom runners
3. **llama.cpp** (included in both) does the inference

The **symlink method** lets us add Kimi to this stack without copying 579GB, giving us:
- Full Ollama API compatibility
- opencode integration
- ROCm GPU acceleration
- Zero additional disk usage

Run `sudo bash /sda2/add-kimi-to-ollama.sh` to set it up!
