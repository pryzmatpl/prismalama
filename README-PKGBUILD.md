# Prismalama Arch package (`PKGBUILD`)

This tree ships **`PKGBUILD`** at the repository root that builds **this** codebase (prismallama.cpp / GGML via CMake, Go `ollama` binary, AirLLM runner assets)—**not** upstream release tarballs.

## What you get

- **GGML**: CPU + **ROCm HIP** + **Vulkan** shared backends installed under `/usr/lib/ollama/rocm` (see `OLLAMA_RUNNER_DIR` in `CMakeLists.txt`).
- **Go binary**: `ollama` with `version.Version` stamped as `prismalama`.
- **AirLLM**: `airllm_runner.py` and `src/airllm/air_llm` → `/usr/share/ollama/` for weight streaming / multi-part GGUF paths (install `python-pytorch-rocm` etc. as needed).
- **systemd** + **`/etc/default/ollama`**: `OLLAMA_MODELS=/nvme3/models`, `OLLAMA_LIBRARY_PATH=/usr/lib/ollama/rocm`, `OLLAMA_USE_AIRLLM=1`, ROCm env vars.

## Quick start (recommended)

```bash
# Optional: refresh vendored llama.cpp + ggml (pin FETCH_HEAD in Makefile.sync)
make -f Makefile.sync sync

# Install build deps (Arch)
sudo pacman -S --needed rocm-hip-sdk rocm-hip-runtime go cmake ninja gcc vulkan-headers vulkan-icd-loader

# Faster link for a single GPU ISA (example: RX 7900 class)
export PRISMALAMA_AMDGPU_TARGETS=gfx1100

./build-rocm.sh
# or: makepkg -sf
sudo pacman -U prismalama-ollama-*.pkg.tar.zst
sudo systemctl enable --now ollama
```

Point models at your disk (default `/nvme3/models`); edit `/etc/default/ollama` if needed.

**When to build:** after each **delivered Prismalama feature** once **integration tests** cover it, and **whenever you change Go or runner code** you intend to run system-wide (otherwise **`/usr/bin/ollama`** stays old until you reinstall). Run **`make ship-check`** (integration + `./build-rocm.sh`) or **`make ship-check-fast`** ( **`TestBlueSky` only**, no package). See **`docs/DEVELOPER.md` § Ship gate**. Bump **`pkgrel`** in **`PKGBUILD`** when releasing a new installable snapshot.

```bash
# Typical refresh after git pull
makepkg -sf
sudo pacman -U prismalama-ollama-*.pkg.tar.zst
sudo systemctl restart ollama
```

### Versioning (pacman vs upstream Ollama)

- **`pkgver` in this `PKGBUILD`** is the **Prismalama package** version (currently `0.4.1`). It does **not** automatically track [upstream Ollama](https://github.com/ollama/ollama) release numbers.
- If you (or an older recipe) previously installed **`prismalama-ollama` with `pkgver` like `0.18.x`**, pacman compares versions numerically: **`0.4.1 < 0.18.2`**, so a fresh `makepkg` build looks like a **downgrade** even though you built “the latest” from **this** repo.
- The package sets **`epoch=1`** so installs sort **after** those legacy `0.18.*` builds. The running binary still reports **`0.4.1-rN-prismalama`** via `-ldflags` (see `build()` in `PKGBUILD`).
- When you intentionally align with an upstream Ollama tag, bump **`pkgver`** (and document it); use **`epoch`** only when pacman ordering needs a reset.

## Hardware

- AMD GPU with ROCm (override `PRISMALAMA_AMDGPU_TARGETS` before `makepkg` if not `gfx1100`).
- Large quantized GGUF or AirLLM-capable layouts: see `docs/DEVELOPER.md`.

---

## Legacy / alternate notes (older scripts)

The sections below describe older `build-all.sh` flows and manual `src/ollama` cmake paths; **prefer `makepkg` + root `PKGBUILD`** above.

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

- Prismalama GGUF engine (llama.cpp fork): https://github.com/piotroxp/prismallama.cpp
- Ollama (upstream): https://github.com/ollama/ollama
- AirLLM: https://github.com/lyogavin/AirLLM
- ROCm: https://rocm.docs.amd.com/

Source builds should sync the vendored engine via `make -f Makefile.sync …` (see `llama/README.md`) and pin `FETCH_HEAD` for reproducible packages.
