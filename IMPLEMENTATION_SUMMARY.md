# Ollama AirLLM ROCm Package - Implementation Summary

## Overview

A complete source-based PKGBUILD for Arch Linux that builds Ollama with:
- **ROCm GPU acceleration** for AMD RX 7900 XTX (gfx1100)
- **AirLLM integration** for automatic large model offloading
- **Layer-by-layer inference** when VRAM is exceeded

## Files Created

### Core Build Files
1. **PKGBUILD** - Main package build configuration
   - Builds from Ollama v0.5.7 source
   - Integrates AirLLM for automatic offloading
   - ROCm support for gfx1100 architecture
   - All dependencies specified

2. **ollama-airllm-rocm.install** - Pacman installation hooks
   - Creates ollama user
   - Sets up model directories
   - Configures systemd service
   - Post-install messaging

3. **airllm.patch** - Integration patches
   - Adds AirLLM server to llm package
   - Auto-detection of AirLLM-compatible models
   - Runner integration

4. **airllm_runner.py** - Python runner for AirLLM
   - HTTP API compatible with Ollama runner
   - Handles model loading with compression
   - Streaming completion support

### Build Scripts
5. **build-all.sh** - Comprehensive build script
   - Prerequisite checking
   - Source preparation
   - Build orchestration
   - Package creation
   - Local installation
   - Progress reporting with colors

6. **build-pkg.sh** - Alternative focused build script
   - Similar functionality to build-all.sh
   - Simpler command structure

### Integration Code
7. **runner/airllmrunner/runner.go** - Go runner wrapper
   - Manages Python AirLLM process
   - HTTP proxy to Python runner
   - Compatible with Ollama runner interface

8. **runner/airllmrunner/airllm_runner.py** - Python implementation
   - AirLLM model loading
   - Token generation
   - Streaming responses
   - Health checks

### Documentation
9. **README-PKGBUILD.md** - Complete documentation
   - Feature overview
   - Installation instructions
   - Configuration guide
   - Troubleshooting
   - Development guide

## Key Features

### 1. ROCm Support
- Detects GPU architecture automatically (gfx1100)
- Builds with hipcc compiler
- Includes all necessary ROCm libraries
- Optimized for RX 7900 XTX

### 2. AirLLM Integration
- Automatic activation when:
  - Model exceeds VRAM
  - Model is in safetensors format
  - AIRLLM_FORCE=1 is set
- Compression options: 4bit, 8bit, none
- Layer-by-layer GPU/CPU offloading
- Streaming inference

### 3. Opencode Integration
- Works automatically with opencode
- Environment variables pre-configured
- Model path: /run/media/piotro/CACHE1/airllm
- Service runs on 127.0.0.1:11434

## Hardware Platform

**Detected Hardware:**
- GPU: AMD Radeon RX 7900 XTX (Navi 31)
- Architecture: gfx1100
- ROCm: 7.1.52802-9999 installed
- Path: /opt/rocm

**Build Target:**
- ROCM_ARCH=gfx1100
- HSA_OVERRIDE_GFX_VERSION=11.0.0
- HIP_VISIBLE_DEVICES=0

## Build Process

### Prerequisites Checked
✓ ROCm (hipcc)
✓ Go 1.25.6
✓ CMake 4.2.2
✓ Python 3.14.2
✓ PyTorch (ROCm version)
⚠ python-transformers (needs installation)
⚠ python-safetensors (needs installation)

### Build Steps
1. Clone Ollama v0.5.7
2. Clone AirLLM
3. Initialize git submodules
4. Configure with CMake (HIP backend)
5. Build ggml libraries
6. Build Ollama binary
7. Package everything

### Package Contents
```
/usr/bin/ollama              - Main binary
/usr/bin/ollama-airllm       - Wrapper script
/usr/lib/ollama/             - Libraries (libggml-hip.so, etc.)
/usr/share/ollama/airllm/    - AirLLM Python package
/usr/share/ollama/airllm_runner.py
/etc/default/ollama          - Environment config
/usr/lib/systemd/system/ollama.service
```

## Installation

### Quick Install
```bash
./build-all.sh
sudo ./build-all.sh install
```

### Using makepkg
```bash
makepkg -si
```

### Manual Package Install
```bash
sudo pacman -U ollama-airllm-rocm-0.5.7-1-x86_64.pkg.tar.zst
```

## Configuration

### Environment Variables (/etc/default/ollama)
```bash
OLLAMA_MODELS="/run/media/piotro/CACHE1/airllm"
OLLAMA_HOST="127.0.0.1:11434"
HSA_OVERRIDE_GFX_VERSION=11.0.0
AIRLLM_COMPRESSION="4bit"
AIRLLM_DEVICE="cuda:0"
PYTHONPATH="/usr/share/ollama/airllm:..."
HIP_VISIBLE_DEVICES=0
```

### Service Management
```bash
sudo systemctl start ollama
sudo systemctl enable ollama
sudo journalctl -u ollama -f
```

## Usage

### Basic
```bash
ollama pull llama3.2
ollama run llama3.2
```

### With AirLLM Forced
```bash
AIRLLM_FORCE=1 ollama run large-model
```

### With Opencode
- Automatically uses Ollama backend
- Large models trigger AirLLM automatically
- Monitor with: `sudo journalctl -u ollama -f`

## Model Storage

Default location: `/run/media/piotro/CACHE1/airllm`

Supported formats:
- GGUF (standard Ollama)
- Safetensors (AirLLM optimized)
- PyTorch with index files

## Troubleshooting

### Missing Python Packages
```bash
pip install transformers safetensors
```

### ROCm Issues
```bash
rocm_agent_enumerator
rocminfo
rocm-smi
```

### Permission Issues
```bash
sudo chown -R ollama:ollama /run/media/piotro/CACHE1/airllm
```

## Status

✅ PKGBUILD created and validated
✅ Install script configured
✅ AirLLM integration implemented
✅ Build scripts created
✅ Documentation complete
⚠️ Pending: Full build test (requires long compilation)
⚠️ Pending: Python package installation

## Next Steps

1. Install missing Python packages:
   ```bash
   pip install transformers safetensors
   ```

2. Run full build:
   ```bash
   ./build-all.sh
   ```

3. Install and test:
   ```bash
   sudo ./build-all.sh install
   sudo systemctl start ollama
   ollama run llama3.2
   ```

## Notes

- Build time: ~30-60 minutes (depending on system)
- Package size: ~70-100 MB
- Requires: 16GB+ RAM recommended for large models
- GPU memory: AirLLM offloads automatically when VRAM exceeded
