# Prismalama Arch package (`PKGBUILD`)

This tree ships **`PKGBUILD`** at the repository root that builds **this** codebase (prismallama.cpp / GGML via CMake, Go `ollama` binary, AirLLM runner assets)—**not** upstream release tarballs.

## What you get

- **GGML**: CPU + **ROCm HIP** + **Vulkan** shared backends installed under `/usr/lib/ollama/rocm` (see `OLLAMA_RUNNER_DIR` in `CMakeLists.txt`).
- **Go binary**: `ollama` with `version.Version` stamped as `prismalama`.
- **AirLLM (optional)**: `airllm_runner.py` and `src/airllm/air_llm` → `/usr/share/ollama/` for Hugging Face / PyTorch-heavy layouts when you **opt in** (`OLLAMA_USE_AIRLLM=1` + PyTorch stack). **Default install uses GGML only** — no `transformers` / heavy Python deps.
- **systemd** + **`/etc/default/ollama`**: `OLLAMA_MODELS=/nvme3/models`, `OLLAMA_LIBRARY_PATH=/usr/lib/ollama/rocm`, **`OLLAMA_USE_AIRLLM=0`** (GGUF + GGML GPU out of the box), ROCm env vars.

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

Point models at your disk (default `/nvme3/models`); edit `/etc/default/ollama` if needed. **GGUF models** match the default (no extra Python). **Hugging Face / safetensors** layouts need **`OLLAMA_USE_AIRLLM=1`** plus the PyTorch stack below — that path is explicitly opt-in.

**If you turn AirLLM on** (`OLLAMA_USE_AIRLLM=1`) **and** the runner selects it (HF-style trees, or you force it), you need **`transformers`** and **`safetensors`** in the same **`python3`** the service uses. **`python-pytorch-rocm`** is in the official repos; **`python-transformers`** / **`python-safetensors`** may or may not be — use **one** of:

```bash
# A) PyPI (most reliable on Arch; PEP 668 requires --break-system-packages for system python)
sudo python3 -m pip install --break-system-packages transformers safetensors
```

```bash
# B) Official repos first (names vary; use search — do not assume an AUR package exists)
sudo pacman -Ss safetensors
sudo pacman -Ss transformers
# Example if both exist in extra/community:
# sudo pacman -S --needed python-pytorch-rocm python-safetensors python-transformers
```

`yay -S python-safetensors` often fails because **there is no AUR package with that exact name**; PyPI packages are **`safetensors`** and **`transformers`**, not `python-safetensors` as a **yay** target unless someone publishes it.

Then **`sudo systemctl restart ollama`**. If `airllm_runner.py` still fails with **`No module named 'transformers'`**, run **`sudo -u ollama python3 -c "import transformers"`** to verify the **`ollama`** user sees the install.

To **avoid** AirLLM entirely (GGUF-only via llama.cpp), set **`OLLAMA_USE_AIRLLM=0`** — **including** multi-part GGUF, AirLLM routing is disabled (see **`docs/RUNTIME_DISPATCH.md`**). No **`transformers`** install is required for that path.

**Docker:** **`docker/gpu`** builds **`prismalama-gpu`** (Go + GGML HIP/Vulkan only). It does **not** ship Python **`transformers`**; use it when you want **GPU GGUF** without the AirLLM stack. **A separate “AirLLM Docker”** is not wired into the runner today (the subprocess expects **`python3`** on the host). If you need AirLLM in a container, you would extend the image with **`pip install transformers safetensors`**, **`python-pytorch-rocm`** (or CUDA), and run **`ollama serve`** there — still simpler than building a second Python-only image unless you are orchestrating at scale.

**When to build:** after each **delivered Prismalama feature** once **integration tests** cover it, and **whenever you change Go or runner code** you intend to run system-wide (otherwise **`/usr/bin/ollama`** stays old until you reinstall). Run **`make ship-check`** (integration + `./build-rocm.sh`) or **`make ship-check-fast`** ( **`TestBlueSky` only**, no package). See **`docs/DEVELOPER.md` § Ship gate**. Bump **`pkgrel`** in **`PKGBUILD`** when releasing a new installable snapshot.

```bash
# Typical refresh after git pull (build, install, pull deps; -f forces rebuild)
sudo makepkg -sfi
sudo systemctl restart ollama
```

Same idea as **`makepkg -sf`** then **`pacman -U`**, but **`-i`** installs the built package in one step; **`-f`** avoids “already built” skips when **`pkgrel`** was not bumped.

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

### `ollama run` spins forever (CLI spinner)

The client shows a spinner until the **first** `/api/generate` response. For an **empty prompt** (interactive mode), the server still **loads the model** first (`scheduleRunner` → runner process). That step can take **minutes** for very large GGUF, slow disks, or first-time GPU init — it is not necessarily a deadlock.

1. **Do not `Ctrl+Z`** — that **suspends** the CLI; the server may still be loading. Use another terminal, or `fg` to resume.
2. **Watch the server** while loading: `sudo journalctl -u ollama -f`. Look for **`runner dispatch engine=llama`** vs **`engine=airllm`**, **`error loading llama server`**, **`ModuleNotFoundError`**, or OOM lines.
3. **Inspect the model**: `ollama show qwopus:latest` — note blob paths / size. Multi-hundred-GB trees take time to mmap even when healthy.
4. **Environment**: Package default is **`OLLAMA_USE_AIRLLM=0`** (GGML only). A **systemd drop-in** (`/etc/systemd/system/ollama.service.d/override.conf`) can still set **`OLLAMA_USE_AIRLLM=1`**; if AirLLM runs but **`transformers`** is missing, loads can fail or stall — either **`pip install …`** as documented above or set **`OLLAMA_USE_AIRLLM=0`** and rely on GGUF + GGML.
5. **GPU discovery**: Warnings about **user overrode visible device** mean an env var hid GPUs; check **`HIP_VISIBLE_DEVICES`** / **`CUDA_VISIBLE_DEVICES`** in `/etc/default/ollama` and overrides. **Vulkan** backends stay off unless **`OLLAMA_VULKAN=1`** (`discover/runner.go`).

### AirLLM not used (or you want to force it)

There is **no** `AIRLLM_FORCE` in the Go runner — use **`OLLAMA_USE_AIRLLM=1`** in `/etc/default/ollama` (and restart) to prefer the Python path when heuristics allow. For **opt-out**, use **`OLLAMA_USE_AIRLLM=0`**.

```bash
# Check Python deps (same user as the service if possible)
sudo -u ollama python3 -c "import transformers, safetensors; print('OK')"

# Verify packaged AirLLM tree
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
