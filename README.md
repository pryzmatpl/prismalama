# Prismalama

Prismalama is an **Ollama-compatible** server built on Vulkan-accelerated GGML (prismallama.cpp) with **layer streaming** enabled by default. It runs GGUF models efficiently on AMD/NVIDIA/Intel GPUs and handles models larger than VRAM via AirLLM-style NVMe weight streaming.

## Key Defaults

| Setting | Default | Purpose |
|---------|---------|---------|
| `OLLAMA_LAYER_STREAMING` | `1` | Layer-by-layer GGUF loading from NVMe (saves RAM) |
| `OLLAMA_KEEP_ALIVE` | `5m` | Models unload after 5 minutes idle (no auto-loading) |
| Backend | GGML/Vulkan | GPU-accelerated inference via Vulkan compute |

## Quick Start

```bash
# Install (Arch Linux)
sudo pacman -U prismalama-ollama-*.pkg.tar.zst

# Start service
sudo systemctl enable --now ollama

# Run a model (loads on-demand, unloads after 5m idle)
ollama run qwen2.5:3b "Hello"

# Or use the API directly
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:3b",
  "prompt": "Hello"
}'
```

## Configuration

### Environment Variables

Edit `/etc/default/ollama`:

```bash
# Model storage (default: /nvme3/models)
OLLAMA_MODELS=/path/to/models

# Layer streaming: load GGUF blocks from NVMe on-demand (default: 1 = enabled)
OLLAMA_LAYER_STREAMING=1

# Keep models loaded for N seconds after last request (default: 5m)
OLLAMA_KEEP_ALIVE=5m

# GPU settings
HIP_VISIBLE_DEVICES=0                    # AMD GPU selection
OLLAMA_VULKAN=1                          # Enable Vulkan backend
OLLAMA_LIBRARY_PATH=/usr/lib/ollama/rocm  # ROCm libraries

# AirLLM (HF safetensors, multi-part GGUF) - opt-in
OLLAMA_USE_AIRLLM=0
```

### Systemd Service Override

For runtime changes that persist across package upgrades, create:
`/etc/systemd/system/ollama.service.d/override.conf`

```ini
[Service]
Environment=OLLAMA_KEEP_ALIVE=5m
Environment=OLLAMA_LAYER_STREAMING=1
```

Then reload: `sudo systemctl daemon-reload && sudo systemctl restart ollama`

## Architecture

Prismalama has **two inference engines**:

### 1. GGML (Default)
- Vulkan-accelerated GGUF inference via prismallama.cpp
- Memory-mapped file access with partial GPU layer offload
- **Layer streaming** (`OLLAMA_LAYER_STREAMING=1`): loads GGUF blocks from NVMe on-demand during inference, evicting previous blocks to stay within budget
- Best for: models that fit (or nearly fit) in VRAM+RAM

### 2. AirLLM (Opt-in)
- True layer-wise weight streaming for models exceeding GPU memory
- Python-based runner for Hugging Face safetensors or multi-part GGUF
- Enable with: `OLLAMA_USE_AIRLLM=1` + `python-pytorch-rocm` + `transformers`

### Engine Selection

| Model Format | Engine | Enable |
|--------------|--------|--------|
| GGUF (single/multi-part) | GGML | Default (layer streaming on) |
| Hugging Face safetensors | AirLLM | `OLLAMA_USE_AIRLLM=1` |

See [docs/RUNTIME_DISPATCH.md](docs/RUNTIME_DISPATCH.md) for details on how the runner selects the engine.

## Memory Behavior

With `OLLAMA_LAYER_STREAMING=1` and `OLLAMA_KEEP_ALIVE=5m`:

1. **On request**: Model loads blocks from NVMe as needed, within streaming budget
2. **After 5m idle**: Model unloads entirely, freeing RAM/VRAM
3. **No model pre-loading**: Prismalama never loads a model at startup

**Note**: Large models (143GB+ MiniMax) may still fail to load if they exceed total system memory even with streaming, or if GPU VRAM cannot accommodate the active working set.

## Troubleshooting

### Model fails to load with "model requires more system memory"
- The model is too large even for layer streaming
- Try a quantized variant (Q4, Q5) or smaller model

### Vulkan buffer allocation errors
- GPU memory exhausted by other models or processes
- Reduce `OLLAMA_KEEP_ALIVE` or unload other models
- Check `journalctl -u ollama` for details

### Layer streaming not working
- Ensure `OLLAMA_LAYER_STREAMING=1` is set
- Verify NVMe path is correct in `OLLAMA_MODELS`

## Development

See [docs/DEVELOPER.md](docs/DEVELOPER.md) for:
- Build instructions
- Code architecture
- Test suite
- Runner implementation details

### Building from Source

```bash
git clone https://github.com/piotroxp/prismalama.git
cd prismalama
makepkg -sfi
```

### Running Tests

```bash
# Unit tests
go test ./...

# Integration tests
go test -tags=integration ./integration/...

# Layer streaming tests
go test -tags=integration,minimax ./integration/...
```

## Model Support

GGUF format models (Llama, Qwen, MiniMax, Mistral, Phi, Gemma, etc.) work out of the box.
Hugging Face safetensors require `OLLAMA_USE_AIRLLM=1` and the PyTorch stack.

## History

Prismalama evolved from upstream Ollama with these key changes:
- Vulkan backend as primary GPU path (not CUDA/ROCm-only)
- Layer streaming via `OLLAMA_LAYER_STREAMING` for GGUF
- AirLLM integration for true NVMe weight streaming
- Arch Linux packaging via PKGBUILD

## License

MIT - See [LICENSE](LICENSE)

## Acknowledgments

- [Ollama](https://github.com/ollama/ollama) - Base server
- [prismallama.cpp](https://github.com/piotroxp/prismallama.cpp) - GGUF/Vulkan fork of llama.cpp
- [AirLLM](https://github.com/AIR-ML/AirLLM) - Layer streaming concepts
