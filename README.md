# Prismalama

![Prismalama Logo](logo.jpg)

Prismalama is an **Ollama-compatible** server built on Vulkan-capable GGML (prismallama.cpp). **Layer streaming** for GGUF is **enabled in the Arch package** (`OLLAMA_LAYER_STREAMING=1` in **`/etc/default/ollama`**); the bare Go binary defaults **`OLLAMA_LAYER_STREAMING` to off when unset** (see **`docs/GOAL-GAPS.md`**). It targets GGUF workloads on AMD/NVIDIA/Intel GPUs; behavior for larger-than-VRAM models depends on mmap/offload policy, backend support for streaming hooks, and whether AirLLM is enabled — see **`docs/RUNTIME_DISPATCH.md`**.

## Key Defaults

| Setting | Default | Purpose |
|---------|---------|---------|
| `OLLAMA_LAYER_STREAMING` | **`1`** in **`/etc/default/ollama`** (package); **unset ⇒ off** in raw `envconfig` | Layer-by-layer GGUF load / streaming path when supported |
| `OLLAMA_KEEP_ALIVE` | `5m` | Models unload after idle window (no startup preload) |
| Compute stack | GGML (HIP / Vulkan / CPU per build) | **`OLLAMA_VULKAN=1`** needed for Vulkan backends on Linux — see RUNTIME_DISPATCH |
| `OLLAMA_USE_AIRLLM` | **`0`** (Arch package) | **`0`/`false`/`no`** disables **all** AirLLM routing; set **`1`** for HF / forced AirLLM |

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

Dispatch is implemented in **`runner/dispatch.go`** (`DecideEngine`). Summary:

| On-disk layout | Engine when `OLLAMA_USE_AIRLLM` is **not** `0`/`false`/`no` |
|----------------|---------------------------------------------------------------|
| HF hints (`*.safetensors`, `model.safetensors.index.json`, typical `config.json`) | AirLLM |
| Multi-part GGUF (`*-00001-of-*.gguf`) | AirLLM |
| Single-file GGUF | GGML |

If **`OLLAMA_USE_AIRLLM=0`** (Arch package default), **every** path stays on **GGML**, including safetensors and multipart heuristics — opt in with **`OLLAMA_USE_AIRLLM=1`** when you need AirLLM.

See [docs/RUNTIME_DISPATCH.md](docs/RUNTIME_DISPATCH.md) for logs, mmap, and GPU stacks.

## Memory Behavior

With `OLLAMA_LAYER_STREAMING=1` and `OLLAMA_KEEP_ALIVE=5m`:

1. **On request**: Model loads blocks from NVMe as needed, within streaming budget
2. **After 5m idle**: Model unloads entirely, freeing RAM/VRAM
3. **No model pre-loading by default**: Prismalama does not preload models at startup in standard package/runtime configuration

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
# Unit tests (large packages such as ./server may take minutes on first compile)
go test ./...

# Integration tests
go test -tags=integration ./integration/...

# Layer streaming tests
go test -tags=integration,minimax ./integration/...
```

Use **`-short`**, **`-timeout`**, or narrow packages (e.g. **`./envconfig`**, **`./runner -run DecideEngine`**) for fast feedback; **`go test ./server/...`** can exceed a few minutes when CGO builds cold cache.

## Model Support

GGUF models are the default path. In practice, compatibility depends on architecture support,
quantization variant, available memory, and backend/runtime setup.
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
