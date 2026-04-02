# Using Kimi K2.5 "As Is" - Direct Inference Guide

## Model Location
- **Path**: `/nvme3/AI Models/Kimi/`
- **Format**: GGUF (13 shards, Q4_K_M quantized)
- **Size**: 579GB
- **Files**: 
  - `Kimi-K2.5-Q4_K_M-00001-of-00013.gguf` through `Kimi-K2.5-Q4_K_M-00013-of-00013.gguf`

## Option 1: llama-cpp-python with ROCm (Recommended)

### Installation
```bash
bash /sda2/install-llama-cpp.sh
```

This installs llama-cpp-python with ROCm support for your RX 7900 XTX.

### Usage
```bash
python3 /sda2/kimi-direct.py "Your prompt here"
```

**Example:**
```bash
python3 /sda2/kimi-direct.py "Explain quantum computing in simple terms"
python3 /sda2/kimi-direct.py "Write a Python function to sort a list"
```

### Features
- ✅ Direct GGUF loading (no copying)
- ✅ ROCm GPU acceleration
- ✅ 579GB model stays on NVMe
- ✅ Layer offloading to GPU
- ✅ Fast token generation

## Option 2: llama.cpp CLI (Build Required)

If you prefer the C++ llama.cpp directly:

```bash
# Build llama.cpp with ROCm
cd /sda2/prismalama
cmake -B build -DGGML_HIPBLAS=ON -DAMDGPU_TARGETS=gfx1100
cmake --build build --config Release -j$(nproc)

# Run inference
./build/bin/llama-cli \
  -m "/nvme3/AI Models/Kimi/Kimi-K2.5-Q4_K_M-00001-of-00013.gguf" \
  -p "Hello, how are you?" \
  -n 512 \
  --temp 0.6 \
  -ngl 999
```

## Option 3: Using Existing Ollama (Copy Required)

**⚠️ Warning**: Requires 579GB free space

```bash
# Create model (will copy 579GB)
ollama create kimi-k2.5 -f /sda2/Modelfile.kimi

# Run
ollama run kimi-k2.5
```

## Quick Reference

### Direct Python Script
```bash
# One-line usage after installation
python3 /sda2/kimi-direct.py "What is the meaning of life?"
```

### With Custom Parameters
```python
from llama_cpp import Llama

llm = Llama(
    model_path="/nvme3/AI Models/Kimi/Kimi-K2.5-Q4_K_M-00001-of-00013.gguf",
    n_gpu_layers=-1,  # All layers on GPU
    n_ctx=32768       # 32K context
)

output = llm(
    "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n",
    max_tokens=1000,
    temperature=0.6
)
```

### Performance Expectations
- **Loading**: 1-2 minutes (first time)
- **GPU Memory**: ~4-8GB (with layer offloading)
- **Speed**: 5-20 tokens/second (depends on context length)
- **Model**: Stays on NVMe, layers loaded on-demand

## Why This Works

**llama.cpp** natively supports:
1. ✅ Multi-file GGUF (the 13 shards)
2. ✅ ROCm GPU acceleration
3. ✅ Layer offloading (load only what's needed)
4. ✅ Direct file access (no copying required)

## Comparison

| Method | Disk Space | Setup | Speed | Recommendation |
|--------|-----------|-------|-------|----------------|
| llama-cpp-python | 0GB extra | Easy | Fast | ⭐⭐⭐⭐⭐ |
| llama.cpp CLI | 0GB extra | Medium | Fast | ⭐⭐⭐⭐ |
| Ollama | 579GB copy | Easy | Fast | ⭐⭐ (disk limited) |
| AirLLM | 0GB extra | Complex | Medium | ⭐⭐ (dependencies) |

## Next Steps

1. **Install llama-cpp-python**:
   ```bash
   bash /sda2/install-llama-cpp.sh
   ```

2. **Test Kimi**:
   ```bash
   python3 /sda2/kimi-direct.py "Hello!"
   ```

3. **Enjoy your 579GB model** without copying! 🎉

---
**Note**: The installation script will compile llama-cpp-python with ROCm support for your RX 7900 XTX. This takes 5-10 minutes but is a one-time setup.
