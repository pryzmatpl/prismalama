# Prismalama Sync Report - /sda2 Migration

> **Deprecated historical note:** one-time migration report.
> Keep for audit/history; do not treat as current operational guidance.

## Current Status

### Models Found on /sda2
1. **Qwen2.5-Coder-32B-Instruct-Q5_K_S.gguf** (22GB) - Main model
2. **GLM-4.7-Flash** (59GB) - Full precision safetensors
3. **GLM-4.7-Flash-4bit** (16GB) - Quantized safetensors
4. **Step3-VL-10B-Base** (12GB) - Vision model
5. **MiniMax-M2.1** (0 bytes) - EMPTY FILE (needs attention)

### Updated Modelfiles
- ✅ `/sda2/Modelfile` - Updated path from `/run/media/piotro/CACHE/` to `/sda2/`
- ✅ `/sda2/Modelfile.qwen25-coder` - New file for Qwen model
- ✅ `/sda2/Modelfile.qwen25-coder-airllm` - AirLLM variant
- ✅ `/sda2/Modelfile.glm47` - GLM-4.7 configuration updated

### Prismalama Build Status
- **Version**: 0.5.7-1 (ollama-airllm-rocm)
- **Location**: `/sda2/prismalama/`
- **Status**: Built and ready
- **Issue**: Configuration files still reference old mount point

## Required Actions

### 1. Complete System Installation
Run the sync script with sudo:
```bash
sudo bash /sda2/prismalama-sync.sh
```

This will:
- Update systemd service to use /sda2 paths
- Update environment configuration
- Install latest binaries
- Create necessary directories
- Restart the service

### 2. Register Models with Ollama
After the service is running:
```bash
# Register Qwen2.5 Coder
ollama create qwen25-coder -f /sda2/Modelfile

# Register GLM-4.7 Flash
ollama create glm47-flash -f /sda2/Modelfile.glm47

# Verify models
ollama list
```

### 3. Sync with OpenCode
OpenCode (v1.1.60) is already installed and has the following local models configured:
- `ollama/hf.co/bigatuna/NousCoder-14B-GGUF:Q4_K_M`
- `ollama/qwen2.5-coder:32b`

After registering models with Ollama, they will be available in OpenCode.

## OpenCode Status

### Current Version
- **Installed**: v1.1.60
- **Location**: `/home/developer/.opencode/bin/opencode`

### Available Models in OpenCode
```
opencode/big-pickle
opencode/claude-3-5-haiku
opencode/claude-opus-4-1
opencode/claude-opus-4-5
opencode/claude-opus-4-6
opencode/gemini-3-flash
opencode/gemini-3-pro
opencode/glm-4.6
opencode/glm-4.7
opencode/gpt-5
opencode/gpt-5-codex
opencode/kimi-k2.5
opencode/kimi-k2.5-free
opencode/minimax-m2.1
opencode/minimax-m2.1-free
ollama/hf.co/bigatuna/NousCoder-14B-GGUF:Q4_K_M
ollama/qwen2.5-coder:32b
```

### Local Models Status
The local models are configured in OpenCode but Ollama service needs to be running for them to work.

## Issues Found

1. **MiniMax Model**: `/sda2/MiniMaxAI_MiniMax-M2.1-Q5_K_S.gguf` is empty (0 bytes)
   - This appears to be a broken symlink or incomplete copy
   - Located at: `/sda2/airllm/nouscoder-14b-q4_k_m.gguf` (symlink to old cache)
   
2. **Ollama Service**: Currently failing to start (status=226/NAMESPACE)
   - Needs path updates in systemd service file
   - The sync script will fix this

3. **Old Path References**: 
   - `/sda2/airllm/nouscoder-14b-q4_k_m.gguf` -> points to `/run/media/piotro/CACHE/`
   - `/sda2/airllm/Qwen2.5-Coder-32B-Instruct-Q5_K_S.gguf` -> points to old cache
   - These need to be updated to point to `/sda2/`

## Files Created/Updated

1. `/sda2/Modelfile` - Updated paths
2. `/sda2/Modelfile.qwen25-coder` - New
3. `/sda2/Modelfile.qwen25-coder-airllm` - New  
4. `/sda2/Modelfile.glm47` - Updated paths
5. `/sda2/prismalama-sync.sh` - Installation script
6. `/sda2/opencode-sync-report.md` - This report

## Next Steps

1. Run: `sudo bash /sda2/prismalama-sync.sh`
2. Register models with Ollama
3. Test with: `opencode run -m ollama/qwen2.5-coder:32b`
4. Update symlinks in /sda2/airllm/ to point to actual model files

## Verification Commands

```bash
# Check Ollama status
systemctl status ollama

# List registered models
ollama list

# Test model
ollama run qwen25-coder "Hello"

# Check OpenCode models
opencode models ollama

# Run OpenCode with local model
opencode run -m ollama/qwen2.5-coder:32b
```

---
Generated: 2026-02-12
Migration: Dynamic mount (/run/media/piotro/CACHE/) -> Static mount (/sda2)
