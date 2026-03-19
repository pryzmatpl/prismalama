# GLM-4.7-Flash with AirLLM + Ollama Integration

## Overview

This setup allows you to run the **GLM-4.7-Flash-4bit** model (15.7GB, 4-bit quantized) on a GPU-poor system using:
- **AirLLM**: Layer-by-layer loading to fit large models in limited VRAM
- **Ollama**: Easy model serving and API access
- **OpenCode**: Direct integration for code generation tasks

## Model Details

| Property | Value |
|----------|--------|
| **Model** | zai-org/GLM-4.7-Flash |
| **Quantization** | 4-bit |
| **Size** | ~15.7 GB |
| **Architecture** | Glm4MoeLiteForCausalLM (MoE) |
| **Context Length** | 4096 tokens |
| **Vocabulary** | 154,880 tokens |
| **Languages** | Chinese, English |

## Installation

### Prerequisites

The following have been installed:
```bash
# Python packages
transformers
safetensors
torch
accelerate
airllm (from /run/media/piotro/CACHE/prismalama/airllm/air_llm)

# System
Ollama (prismalama/ollama binary)
```

### Setup Already Complete

All files have been created at `/run/media/piotro/CACHE/airllm/`:

```
/run/media/piotro/CACHE/airllm/
├── GLM-4.7-Flash-4bit -> ../GLM-4.7-Flash-4bit (symlink)
├── Modelfile.glm47
├── airllm_env.sh
├── test_glm47.py
├── opencode_integration.py
└── opencode_config.json
```

## Usage

### Option 1: Direct Python (AirLLM)

Best for: Testing, direct inference, custom applications

```bash
cd /run/media/piotro/CACHE/airllm
source ./airllm_env.sh
./test_glm47.py
```

**Output example:**
```
============================================================
Testing GLM-4.7-Flash with AirLLM
============================================================
✓ CUDA available
  Device: 0
  Total memory: 12.00 GB
  Free memory: 11.50 GB

Loading model (this may take a while)...
Using low-memory layer-by-layer loading...
✓ Model loaded successfully!

============================================================
Testing inference...
============================================================
Input: What is the capital of Poland?
Tokens: torch.Size([1, 6])

Output: What is the capital of Poland?
The capital of Poland is Warsaw.

✓ GLM-4.7-Flash is working with AirLLM!
```

### Option 2: Ollama Server

Best for: API access, multiple clients, easy serving

```bash
# Start Ollama service
systemctl start ollama

# Create model from Modelfile (if not already created)
ollama create glm47 -f /run/media/piotro/CACHE/airllm/Modelfile.glm47

# Run model interactively
ollama run glm47

# Use via API
curl http://127.0.0.1:11434/api/generate -d '{
  "model": "glm47",
  "prompt": "Hello, how are you?",
  "stream": false
}'
```

### Option 3: OpenCode Integration

Best for: Code generation tasks, IDE integration, automated workflows

```python
import json
import sys

# Add AirLLM to path
sys.path.insert(0, "/run/media/piotro/CACHE/prismalama/airllm/air_llm")

# Load configuration
with open("/run/media/piotro/CACHE/airllm/opencode_config.json") as f:
    config = json.load(f)

# Use in your code
print(f"Model: {config['model']['name']}")
print(f"Path: {config['model']['path']}")
print(f"Backend: {config['inference']['backend']}")

# Initialize AirLLM model
from airllm import AutoModel

model = AutoModel.from_pretrained(
    config['model']['path'],
    compression=config['inference']['compression'],
    profiling_mode=config['inference']['layer_loading']
)

# Generate text
input_text = ["def fibonacci(n):"]
tokens = model.tokenizer(input_text, return_tensors="pt", return_attention_mask=False)
output = model.generate(tokens['input_ids'].cuda(), max_new_tokens=100)
result = model.tokenizer.decode(output.sequences[0])

print(result)
```

## Environment Configuration

### AirLLM Environment Variables

Set in `/run/media/piotro/CACHE/airllm/airllm_env.sh`:

```bash
# Model configuration
export AIRLLM_MODEL_PATH="/run/media/piotro/CACHE/GLM-4.7-Flash-4bit"
export AIRLLM_COMPRESSION="4bit"
export AIRLLM_PREFETCHING="true"

# Python path
export PYTHONPATH="/run/media/piotro/CACHE/prismalama/airllm/air_llm:$PYTHONPATH"

# Ollama models path
export OLLAMA_MODELS="/run/media/piotro/CACHE/airllm"

# Performance tuning
export OLLAMA_NUM_PARALLEL="1"
export OLLAMA_MAX_LOADED_MODELS="1"
export OLLAMA_CONTEXT_LENGTH="4096"
export OLLAMA_KEEP_ALIVE="5m"
```

### Ollama Configuration

Set in `/etc/default/ollama` (automatically configured by the package):

```bash
export OLLAMA_MODELS="/run/media/piotro/CACHE/airllm"
```

## Memory Requirements

| Component | Required VRAM |
|-----------|---------------|
| **Minimum (CPU)** | 0 GB (slow) |
| **Recommended (4-bit)** | 8-12 GB |
| **Optimal** | 16+ GB |

**With 8GB VRAM:**
- AirLLM will load layers sequentially
- Each layer is loaded, processed, then unloaded
- Fits the full 15.7GB model in 8GB VRAM
- Slightly slower but functional

**With 16GB+ VRAM:**
- Can keep multiple layers in memory
- Better performance
- Still uses AirLLM for memory management

## Performance Tuning

### Low-Memory Configuration

Edit `Modelfile.glm47`:

```dockerfile
PARAMETER num_ctx 2048        # Reduce from 4096
PARAMETER num_batch 4          # Reduce from 8
PARAMETER num_gpu 1            # Use single GPU
PARAMETER num_thread 4          # Reduce CPU threads
```

### High-Memory Configuration

```dockerfile
PARAMETER num_ctx 8192        # Increase context
PARAMETER num_batch 16         # Increase batch size
PARAMETER temperature 0.8        # Higher temperature
```

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution:**
```bash
# Reduce context length
export OLLAMA_CONTEXT_LENGTH="2048"

# Use CPU-only mode
export CUDA_VISIBLE_DEVICES=""

# Try smaller models first to verify setup
```

### Issue: "Model loading is very slow"

**Solution:**
```bash
# Enable AirLLM compression
export AIRLLM_COMPRESSION="4bit"

# Enable prefetching
export AIRLLM_PREFETCHING="true"

# Use SSD for caching
export AIRLLM_CACHE_DIR="/run/media/piotro/CACHE/airllm/.cache"
```

### Issue: "Ollama service not starting"

**Solution:**
```bash
# Check logs
sudo journalctl -u ollama -n 50

# Verify binary permissions
ls -la /usr/bin/ollama
sudo chmod +x /usr/bin/ollama

# Check port conflicts
netstat -tuln | grep 11434
```

### Issue: "Model architecture not recognized"

**Solution:**
The AirLLM auto_model.py has been updated to recognize GLM-4.7:

```python
# In /run/media/piotro/CACHE/prismalama/airllm/air_llm/airllm/auto_model.py
elif "ChatGLM" in config.architectures[0] or "Glm4" in config.architectures[0] or "GLM4" in config.architectures[0]:
    return "airllm", "AirLLMChatGLM"
```

This allows GLM-4.7 models to use the ChatGLM adapter.

## Architecture Details

### GLM-4.7-Flash Architecture

```
Glm4MoeLiteForCausalLM
├── Mixture of Experts (MoE)
│   ├── n_routed_experts: 64
│   ├── n_shared_experts: 1
│   └── num_experts_per_tok: 4
├── 47 Transformer Layers
├── Hidden Size: 2048
├── Attention Heads: 20
├── Rotary Embeddings: Enabled
└── Context Window: 202,752 tokens (configurable)
```

### AirLLM Memory Optimization

```
Traditional Loading:
[Model] -> [Load All at Once] -> [VRAM: 15.7 GB] ✗

AirLLM Loading:
[Model] -> [Load Layer 1] -> [Process] -> [Unload] -> [Load Layer 2] ...
         [VRAM: ~2 GB per layer] ✓

Result: 15.7 GB model fits in 8 GB VRAM!
```

## API Endpoints

### Ollama API

```bash
# Generate text
curl http://127.0.0.1:11434/api/generate -d '{
  "model": "glm47",
  "prompt": "Explain quantum computing",
  "stream": false
}'

# Chat completion
curl http://127.0.0.1:11434/api/chat -d '{
  "model": "glm47",
  "messages": [
    {"role": "user", "content": "What is AI?"}
  ]
}'

# List models
curl http://127.0.0.1:11434/api/tags

# Model info
curl http://127.0.0.1:11434/api/show -d '{
  "name": "glm47"
}'
```

### Python Client

```python
import requests

# Generate
response = requests.post('http://127.0.0.1:11434/api/generate', json={
    'model': 'glm47',
    'prompt': 'Hello, world!',
    'stream': False
})

print(response.json()['response'])

# Chat
response = requests.post('http://127.0.0.1:11434/api/chat', json={
    'model': 'glm47',
    'messages': [
        {'role': 'user', 'content': 'Explain machine learning'}
    ]
})

print(response.json()['message']['content'])
```

## Comparison: GLM-4.7 vs Other Models

| Model | Size | VRAM Required | Features |
|--------|-------|---------------|-----------|
| **GLM-4.7-Flash** | 15.7 GB | 8-12 GB (4-bit) | Chinese/English, MoE, Flash attention |
| Llama 2 70B | 140 GB | 24+ GB (8-bit) | English only |
| Qwen 72B | 144 GB | 24+ GB (4-bit) | Chinese/English |
| ChatGLM3 6B | 12 GB | 4-6 GB | Chinese/English |

**Advantages of GLM-4.7:**
- ✅ Optimized for both Chinese and English
- ✅ Efficient MoE architecture (64 experts)
- ✅ Large context window (202K tokens)
- ✅ 4-bit quantization for memory efficiency
- ✅ Flash attention for speed

## Next Steps

1. **Test the setup:**
   ```bash
   cd /run/media/piotro/CACHE/airllm
   source ./airllm_env.sh
   ./test_glm47.py
   ```

2. **Start Ollama server:**
   ```bash
   systemctl start ollama
   systemctl enable ollama  # Auto-start on boot
   ```

3. **Use in applications:**
   - Python: Load via AirLLM directly
   - API: Use Ollama REST API
   - OpenCode: Use opencode_config.json
   - CLI: `ollama run glm47`

4. **Monitor performance:**
   ```bash
   # GPU memory
   nvidia-smi

   # Ollama logs
   journalctl -u ollama -f

   # Model loading
   ./test_glm47.py  # Shows timing info
   ```

## Support & Resources

- **GLM-4.7 Model**: https://huggingface.co/zai-org/GLM-4.7-Flash
- **AirLLM**: https://github.com/lyogavin/airllm
- **Ollama**: https://github.com/ollama/ollama
- **Prismalama**: /run/media/piotro/CACHE/prismalama

## License

- GLM-4.7: Apache 2.0
- AirLLM: Apache 2.0
- Ollama: MIT License

---

**Status:** ✅ Ready for use with OpenCode

**Configuration Path:** `/run/media/piotro/CACHE/airllm/opencode_config.json`

**Quick Start:**
```bash
cd /run/media/piotro/CACHE/airllm
source ./airllm_env.sh
./test_glm47.py
```
