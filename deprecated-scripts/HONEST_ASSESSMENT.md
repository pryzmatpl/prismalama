# KIMI ON POTATO HARDWARE - Honest Assessment

> **Deprecated historical note:** this file is machine-specific legacy guidance.
> Current supported behavior is documented in `docs/DEVELOPER.md`,
> `docs/RUNTIME_DISPATCH.md`, and `README-PKGBUILD.md`.
> Statements below may be outdated or inaccurate for the current Prismalama codebase.

## The Real Problem

We've been fighting Ollama's architecture. Here's what's actually happening:

1. **Multi-file GGUF Issue**: Kimi is 13 separate GGUF files (00001-of-00013 to 00013-of-00013)
2. **Ollama's Design**: Ollama expects single files or properly imported multi-file models
3. **The Error**: "invalid split file name" means Ollama's llama.cpp can't find the other 12 shards

## What Actually Works

### Option 1: Let Ollama Copy Properly (RECOMMENDED)

Since you now have 1.2TB free on /nvme3, let Ollama do its thing:

```bash
# Set environment for your user session
export OLLAMA_MODELS=/nvme3/ollama-models

# Create Kimi model (Ollama will copy and properly index all 13 shards)
ollama create kimi -f /sda2/Modelfile.kimi

# Wait for it... (579GB copy will take ~10-20 minutes)
# Then run:
ollama run kimi "Hello!"
```

**This WILL work** because:
- Ollama knows how to handle multi-file GGUF during `create`
- It has space on /nvme3 to copy
- The model will be properly indexed

### Option 2: Use llama.cpp Directly (ALREADY WORKS)

I created this earlier:

```bash
# Install llama-cpp-python with ROCm
pip3 install llama-cpp-python --force-reinstall --no-cache-dir

# Run Kimi directly
python3 /sda2/kimi-direct.py "Hello!"
```

**Pros:** No Ollama complexity, works immediately
**Cons:** No API server, no opencode integration

### Option 3: llama.cpp Server (BEST OF BOTH WORLDS)

```bash
# Build llama.cpp server
cd /sda2/prismalama
cmake -B build -DGGML_HIPBLAS=ON
make -C build llama-server

# Run server pointing to Kimi
./build/bin/llama-server \
  -m "/nvme3/AI Models/Kimi/Kimi-K2.5-Q4_K_M-00001-of-00013.gguf" \
  --port 11434 \
  -ngl 999
```

**This gives you:**
- HTTP API (like Ollama)
- ROCm GPU acceleration
- Direct file access (no copying)
- Can be used with anything that speaks HTTP

## What Doesn't Work (And Why)

❌ **Symlinks in Ollama blobs**: Ollama validates and copies files anyway
❌ **Manual manifest creation**: Ollama regenerates manifests on `create`
❌ **Trying to avoid the copy**: Ollama's architecture requires importing models

## My Recommendation

**Use Option 1** (Let Ollama copy) because:
1. ✅ You have space now (1.2TB on /nvme3)
2. ✅ It integrates with opencode
3. ✅ Full API support
4. ✅ One-time wait in that environment, then repeatable there (not a universal guarantee)

Run this:
```bash
export OLLAMA_MODELS=/nvme3/ollama-models
ollama create kimi -f /sda2/Modelfile.kimi
ollama run kimi "Let's save humanity!"
```

**Alternative**: Use Option 3 (llama.cpp server) if you want immediate results without waiting for the copy.

## The Bottom Line

The "potato hardware" solution works - you CAN run 579GB Kimi with 24GB VRAM using layer offloading. The issue was never the hardware, it was trying to bypass Ollama's import process.

**Stop fighting the tool. Let Ollama copy the model to /nvme3. It will work.**
