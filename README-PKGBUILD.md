# Ollama with AirLLM and ROCm Support

This package provides Ollama built from source with AirLLM integration and ROCm GPU acceleration for AMD RX 7900 XTX (gfx1100).

## Features

- **ROCm Support**: Full GPU acceleration for AMD RX 7900 XTX (gfx1100)
- **AirLLM Integration**: Automatic offloading of large models when VRAM is insufficient
- **Layer-by-Layer Inference**: Only loads necessary layers into GPU memory
- **4-bit Compression**: Default compression for reduced memory usage
- **Safetensors Support**: Native support for safetensors format models
- **Opencode Integration**: Works seamlessly with opencode for AI assistance

## Hardware Requirements

- AMD RX 7900 XTX or compatible GPU (gfx1100 architecture)
- ROCm 6.x or later
- 16GB+ VRAM recommended
- 32GB+ RAM for large model offloading

## Quick Start

### Option 1: Use the Build Script (Recommended)

```bash
# Full build process
./build-all.sh

# Or step by step
./build-all.sh check      # Check prerequisites
./build-all.sh prepare    # Clone sources
./build-all.sh build      # Build from source
./build-all.sh package    # Create package
./build-all.sh pkgbuild   # Build pacman package
./build-all.sh install    # Install locally
```

### Option 2: Use makepkg Directly

```bash
# Install dependencies first
sudo pacman -S rocm-hip-sdk go cmake git python-pytorch-rocm python-transformers python-safetensors

# Build with makepkg
makepkg -si
```

### Option 3: Manual Installation

```bash
# Build
cd src/ollama
mkdir build && cd build
cmake .. -DLLAMA_HIPBLAS=ON -DAMDGPU_TARGETS=gfx1100
cmake --build . -j$(nproc)
cd ..
go build -o ollama .

# Install manually
sudo cp ollama /usr/bin/
sudo cp -r build/lib/ollama/* /usr/lib/ollama/
sudo systemctl daemon-reload
```

## Configuration

### Environment Variables

Edit `/etc/default/ollama` to configure:

```bash
# Model storage location
export OLLAMA_MODELS="/run/media/piotro/CACHE1/airllm"

# ROCm configuration
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0

# AirLLM configuration
export AIRLLM_COMPRESSION="4bit"  # Options: 4bit, 8bit, none
export AIRLLM_DEVICE="cuda:0"

# Force AirLLM for all compatible models
export AIRLLM_FORCE=1
```

### Systemd Service

```bash
# Start service
sudo systemctl start ollama

# Enable on boot
sudo systemctl enable ollama

# View logs
sudo journalctl -u ollama -f
```

## Usage

### Basic Usage

```bash
# List available models
ollama list

# Pull a model
ollama pull llama3.2

# Run a model
ollama run llama3.2

# Run with AirLLM forced
USE_AIRLLM=1 ollama run llama3.2
```

### With Opencode

For opencode integration, the package automatically handles large model offloading:

1. Set the Ollama endpoint in opencode config:
   ```json
   {
     "ollama_host": "http://127.0.0.1:11434"
   }
   ```

2. Large models (exceeding VRAM) will automatically use AirLLM

3. Check logs to see AirLLM activation:
   ```bash
   sudo journalctl -u ollama -f | grep -i airllm
   ```

### Model Storage

Models are stored in `/run/media/piotro/CACHE1/airllm` by default. This can be changed in `/etc/default/ollama`.

Supported formats:
- **GGUF**: Standard Ollama format
- **Safetensors**: AirLLM optimized format
- **PyTorch**: With index files

## AirLLM Features

### Automatic Offloading

AirLLM automatically activates when:
- Model size exceeds available VRAM
- Model is in safetensors format
- `AIRLLM_FORCE=1` is set

### Compression Options

- **4bit** (default): Maximum compression, ~75% memory savings
- **8bit**: Balanced compression and quality
- **none**: No compression, full precision

Set via environment variable:
```bash
export AIRLLM_COMPRESSION="8bit"
```

### Layer Streaming

AirLLM loads model layers on-demand:
1. Only active layers are in GPU memory
2. Inactive layers are offloaded to RAM
3. Minimal latency with efficient caching

## Troubleshooting

### ROCm Not Detected

```bash
# Check ROCm installation
rocm_agent_enumerator
rocminfo | grep "Name:"

# Verify HIP
hipcc --version
```

### Models Not Loading

```bash
# Check model path
ls -la /run/media/piotro/CACHE1/airllm

# Verify permissions
sudo chown -R ollama:ollama /run/media/piotro/CACHE1/airllm

# Check logs
sudo journalctl -u ollama -n 100
```

### AirLLM Not Activating

```bash
# Force AirLLM mode
export AIRLLM_FORCE=1
ollama run <model>

# Check Python dependencies
python3 -c "from airllm import AutoModel; print('OK')"

# Verify AirLLM installation
ls -la /usr/share/ollama/airllm/
```

### Performance Issues

1. **Check GPU usage**: `rocm-smi`
2. **Monitor memory**: `free -h` and check VRAM
3. **Adjust compression**: Try different `AIRLLM_COMPRESSION` values
4. **Reduce batch size**: In model options

## File Structure

```
/usr/bin/ollama              # Main binary
/usr/bin/ollama-airllm       # Wrapper script
/usr/lib/ollama/             # Libraries
  ├── libggml-base.so
  ├── libggml-cpu-*.so
  └── libggml-hip.so         # ROCm backend
/usr/share/ollama/
  ├── airllm/                # AirLLM Python package
  │   └── air_llm/
  └── airllm_runner.py       # AirLLM runner
/etc/default/ollama          # Environment config
/usr/lib/systemd/system/ollama.service
```

## Building from Source

### Prerequisites

```bash
sudo pacman -S \
    rocm-hip-sdk \
    rocm-cmake \
    go \
    cmake \
    git \
    python-pytorch-rocm \
    python-transformers \
    python-safetensors \
    python-numpy \
    python-accelerate
```

### Build Steps

1. **Clone and prepare**:
   ```bash
   ./build-all.sh prepare
   ```

2. **Build Ollama**:
   ```bash
   ./build-all.sh build
   ```

3. **Create package**:
   ```bash
   ./build-all.sh package
   ./build-all.sh pkgbuild
   ```

4. **Install**:
   ```bash
   sudo ./build-all.sh install
   ```

## Uninstall

```bash
# Remove package
sudo pacman -R ollama-airllm-rocm

# Clean up (optional)
sudo rm -rf /run/media/piotro/CACHE1/airllm
sudo rm -rf /var/lib/ollama
sudo userdel ollama
```

## Development

### Project Structure

```
.
├── PKGBUILD                    # Arch Linux package build
├── ollama-airllm-rocm.install  # Installation hooks
├── airllm.patch               # AirLLM integration patches
├── build-all.sh               # Build script
├── build-pkg.sh               # Alternative build script
├── runner/
│   └── airllmrunner/
│       ├── runner.go          # Go AirLLM runner
│       └── airllm_runner.py   # Python AirLLM runner
└── src/                       # Source directory (created during build)
    ├── ollama/               # Ollama source
    └── airllm/               # AirLLM source
```

### Adding New Features

1. Modify `runner/airllmrunner/` for AirLLM changes
2. Update `airllm.patch` for Ollama integration
3. Test with `./build-all.sh build`
4. Update version in PKGBUILD

## License

MIT License - See LICENSE file for details.

## Support

- Ollama: https://github.com/ollama/ollama
- AirLLM: https://github.com/lyogavin/AirLLM
- ROCm: https://rocm.docs.amd.com/
