# Fix for Ollama Service - Read-only File System Error

## Problem
The ollama service is failing with:
```
Error: remove /sda2/airllm/manifests: read-only file system
```

## Root Cause
Two conflicting configurations:
1. `/etc/default/ollama` sets `OLLAMA_MODELS=/sda2/airllm`
2. `/etc/systemd/system/ollama.service.d/override.conf` sets `OLLAMA_MODELS=/nvme3/models`
3. `ProtectSystem=strict` makes filesystem read-only except `ReadWritePaths`
4. `/sda2/airllm` is NOT in ReadWritePaths, but `/nvme3/models` is

## Solution

### 1. Update /etc/default/ollama
Run as root:
```bash
sudo cat > /etc/default/ollama << 'EOF'
OLLAMA_MODELS=/nvme3/models
OLLAMA_HOST="127.0.0.1:11434"
HSA_OVERRIDE_GFX_VERSION=11.0.0
AIRLLM_COMPRESSION="4bit"
AIRLLM_DEVICE="cuda:0"
PYTHONPATH="/usr/share/ollama/airllm:/usr/share/ollama/airllm/air_llm:/usr/share/ollama/airllm/air_llm"

# ROCm specific
HIP_VISIBLE_DEVICES=0
EOF
```

### 2. Restart the service
```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

### 3. Verify
```bash
systemctl status ollama
journalctl -u ollama -f
```

## Additional Fix Needed in Build Script

The build script `build-rocm.sh` line 115 should be updated to use `/nvme3/models` instead of `/sda2/airllm` to prevent this issue in future builds.
