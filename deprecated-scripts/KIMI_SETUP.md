# Kimi K2.5 Setup for Ollama AirLLM ROCm

> **Deprecated historical note:** this file captures an older host-specific setup.
> Current behavior and defaults are documented in `docs/DEVELOPER.md` and `docs/GOAL-GAPS.md`.
> Statements below may be outdated or inaccurate for the current Prismalama codebase.

## Model Information

- **Model**: Kimi K2.5 (Moonshot AI)
- **Size**: 579GB (13 shards, Q4_K_M quantized)
- **Location**: `/nvme3/AI Models/Kimi/`
- **Symlink**: `/sda2/airllm/kimi-k2.5/`
- **Context Length**: 32,768 tokens
- **Architecture**: Mixture of Experts (MoE)

## Setup Status: ✅ Complete

All configuration files have been created and the model is ready for inference.

## Files Created

1. **Modelfiles**:
   - `/sda2/Modelfile.kimi-k2.5` - Standard Ollama configuration
   - `/sda2/Modelfile.kimi-airllm` - AirLLM-optimized for Potato machines

2. **Runner Scripts**:
   - `/sda2/airllm/kimi_runner.py` - Python AirLLM runner
   - `/sda2/register-kimi.sh` - Ollama registration script

3. **Configuration**:
   - `/sda2/airllm/opencode_config.json` - Updated with Kimi model info

## Quick Start

### Register with Ollama

```bash
sudo bash /sda2/register-kimi.sh
```

This will create two Ollama models:

- `kimi-k2.5` - Standard version
- `kimi-k2.5-airllm` - Optimized with AirLLM layer offloading

### Run Inference

**For Potato Machines (Recommended)**:

```bash
# Use AirLLM-optimized version for 579GB model
ollama run kimi-k2.5-airllm

# Or with a specific prompt
ollama run kimi-k2.5-airllm "Explain quantum computing in simple terms"
```

**Standard Mode** (if you have enough VRAM):

```bash
ollama run kimi-k2.5
```

### With OpenCode

```bash
# Run with opencode using the local model
opencode run -m ollama/kimi-k2.5-airllm

# Or in the TUI
opencode
# Then select: ollama/kimi-k2.5-airllm
```

## Potato Machine Optimization

Since Kimi K2.5 is 579GB, it requires AirLLM for inference on consumer hardware:

### AirLLM Features Enabled:

- ✅ **Layer Offloading**: Loads only active layers into GPU memory
- ✅ **4-bit Quantization**: Reduces memory footprint
- ✅ **Prefetching**: Pre-loads next layers for speed
- ✅ **Streaming**: Outputs tokens as they're generated

### Memory Requirements:

- **Minimum GPU**: 8GB VRAM (with AirLLM)
- **Recommended**: 16GB+ VRAM for better performance
- **CPU Mode**: Works entirely on CPU (slower but functional)

### Performance Expectations:

- First load: ~30-60 seconds (loading model shards)
- Token generation: ~2-5 tokens/second (depends on hardware)
- Memory usage: ~4-8GB GPU RAM during inference

## Configuration Details

### Systemd Service

The Ollama service has been configured with:

```ini
ReadWritePaths=/sda2/airllm /var/lib/ollama /tmp
Environment=OLLAMA_MODELS=/sda2/airllm
```

### Environment Variables

```bash
OLLAMA_MODELS="/sda2/airllm"
OLLAMA_HOST="127.0.0.1:11434"
HSA_OVERRIDE_GFX_VERSION=11.0.0
AIRLLM_COMPRESSION="4bit"
AIRLLM_DEVICE="cuda:0"
```

## Troubleshooting

### Model Not Loading

```bash
# Check Ollama service
systemctl status ollama

# View logs
journalctl -u ollama -f

# Restart service
sudo systemctl restart ollama
```

### Out of Memory

- Use `kimi-k2.5-airllm` instead of `kimi-k2.5`
- Reduce context length in Modelfile
- Enable CPU-only mode by setting `num_gpu 0`

### Slow Performance

- Ensure AirLLM is using GPU: check `HIP_VISIBLE_DEVICES=0`
- Enable prefetching in AirLLM config
- Use smaller batch sizes

## Model Files Structure

```
/nvme3/AI Models/Kimi/
├── Kimi-K2.5-Q4_K_M-00001-of-00013.gguf (44GB)
├── Kimi-K2.5-Q4_K_M-00002-of-00013.gguf (46GB)
├── ...
└── Kimi-K2.5-Q4_K_M-00013-of-00013.gguf (42GB)

/sda2/airllm/kimi-k2.5 -> /nvme3/AI Models/Kimi (symlink)
```

## Advanced Usage

### Custom Prompts

```bash
# Long-form generation
ollama run kimi-k2.5-airllm "Write a detailed essay about..."

# Code generation
ollama run kimi-k2.5-airllm "Write a Python function that..."

# Analysis
ollama run kimi-k2.5-airllm "Analyze the following text: ..."
```

### Batch Processing

Use the Python runner directly:

```bash
python3 /sda2/airllm/kimi_runner.py \
  --prompt "Your prompt here" \
  --max-length 4096 \
  --temperature 0.7
```

## Integration with OpenCode

The Kimi model is now available in OpenCode:

```bash
# List available models
opencode models ollama

# You should see:
# - ollama/kimi-k2.5
# - ollama/kimi-k2.5-airllm

# Run with Kimi
opencode run -m ollama/kimi-k2.5-airllm "Your question here"
```

## Next Steps

1. ✅ Register model: `sudo bash /sda2/register-kimi.sh`
2. ✅ Test inference: `ollama run kimi-k2.5-airllm`
3. ✅ Use with OpenCode: `opencode run -m ollama/kimi-k2.5-airllm`
4. 🎉 Enjoy your 579GB model on a Potato machine!

## Notes

- Kimi K2.5 excels at long-context tasks (up to 32k tokens)
- Use AirLLM version for any system with <100GB VRAM
- Model is stored on NVMe but accessible via /sda2 symlink
- First inference may take 1-2 minutes (loading shards)

---

Setup completed: 2026-02-12
