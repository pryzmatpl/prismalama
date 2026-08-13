# Prismalama Arch Package (`PKGBUILD`)

![Prismalama Logo](logo.jpg)

Builds **this** Prismalama tree (prismallama.cpp/GGML via CMake, Go `ollama` binary, AirLLM assets) into an Arch Linux package — **not** upstream Ollama tarballs.

## What's Installed

| Path                                     | Contents                                                                           |
| ---------------------------------------- | ---------------------------------------------------------------------------------- |
| `/usr/bin/ollama`                        | Prismalama binary (Go + GGML/Vulkan)                                               |
| `/usr/lib/ollama/rocm/`                  | GGML backends (CPU + Vulkan; HIP/CUDA per **`PRISMALAMA_BACKENDS`** at build time) |
| `/usr/share/ollama/airllm_runner.py`     | AirLLM Python runner                                                               |
| `/usr/share/ollama/airllm/`              | AirLLM Python package (if `src/airllm/air_llm` exists)                             |
| `/etc/default/ollama`                    | Environment variables                                                              |
| `/usr/lib/systemd/system/ollama.service` | Systemd service                                                                    |

## Defaults

The package sets these defaults in `/etc/default/ollama`:

```bash
OLLAMA_MODELS=/nvme3/models
OLLAMA_HOST=127.0.0.1:11434
OLLAMA_NUM_PARALLEL=1
OLLAMA_KEEP_ALIVE=5m                    # Models unload after 5 min idle
OLLAMA_LIBRARY_PATH=/usr/lib/ollama/rocm
OLLAMA_LAYER_STREAMING=1                # GGUF layer streaming enabled
OLLAMA_USE_AIRLLM=0                    # AirLLM opt-in only
# AMD / all profiles only:
# HIP_VISIBLE_DEVICES=0
# HSA_OVERRIDE_GFX_VERSION=11.0.0
```

## Build

### Backend profile (**pacman** `makedepends`)

Set **`PRISMALAMA_BACKENDS`** **before** `makepkg` so only the stacks you need are installed:

| Value                | Pulls (typical)                                                                                                                                                                                             | CMake                   |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| **`amd`**            | `rocm-hip-sdk`, HIP runtime                                                                                                                                                                                 | HIP + Vulkan            |
| **`nvidia`**         | `cuda`                                                                                                                                                                                                      | CUDA + Vulkan (no ROCm) |
| **`all`**            | ROCm + CUDA                                                                                                                                                                                                 | HIP + CUDA + Vulkan     |
| **`minimal`**        | neither ROCm nor CUDA toolkits                                                                                                                                                                              | CPU + Vulkan only       |
| **`auto`** (default) | Uses **`pacman -Q`** for `rocm-hip-sdk` / `cuda`, else **`lspci`** PCI vendor (**`1002`** AMD, **`10de`** NVIDIA, **`8086`** Intel iGPU → minimal), else **`minimal`** or **`PRISMALAMA_BACKENDS_DEFAULT`** |

**NVIDIA-only host (avoid ROCm):**

```bash
export PRISMALAMA_BACKENDS=nvidia
makepkg -sfi
```

```bash
# Build and install
makepkg -sfi

# Or just build
makepkg -sf

# Install pre-built package
sudo pacman -U prismalama-ollama-*.pkg.tar.zst

# Restart after install/upgrade
sudo systemctl restart ollama
```

### GPU ISA (HIP / ROCm)

HIP kernels are compiled for a specific **AMDGPU_TARGETS** ISA string.

- **Auto (recommended on the target PC):** leave **`PRISMALAMA_AMDGPU_TARGETS` unset** and run **`makepkg -sfi`** on a machine where **`rocminfo`** sees your GPU. The `PKGBUILD` runs **`scripts/detect-prismalama-amdgpu-target.sh`** and prints the chosen gfx via `warning()` during the build.
- **Fallback:** if detection fails (e.g. build server without AMD GPU), the package still builds with **`gfx1100`** and prints a warning — override explicitly for your ISA.
- **Manual override:**

```bash
export PRISMALAMA_AMDGPU_TARGETS=gfx1030  # RDNA2 (RX 6xxx/7xxx)
makepkg -sfi
```

- **Disable auto-detect** (use fallback default only): `PRISMALAMA_AMDGPU_AUTO=0 makepkg -sfi`

### CUDA (NVIDIA GGML)

With **`PRISMALAMA_BACKENDS=nvidia`** or **`all`**, if **`nvcc`** is on **`PATH`** when you run **`makepkg`**, the PKGBUILD sets **`LLAMA_CUDA=ON`** and builds **`ggml-cuda`**. For **`amd`** / **`minimal`**, CUDA is not enabled (use **`all`** or **`nvidia`**).

- **Architectures:** export **`PRISMALAMA_CUDA_ARCHITECTURES`** before building (default **`native`** via CMake when unset in PKGBUILD — same idea as upstream ggml-cuda).
- **Disable CUDA even when nvcc exists:** **`PRISMALAMA_CUDA_AUTO=0 makepkg -sfi`**
- **Host C++ compiler for CUDA:** GCC **16** ships libstdc++ headers that **nvcc** cannot parse (errors in `<functional>` around `operator()(this …)`). The PKGBUILD automatically uses **`g++-14`**, then **`g++-13`**, then **`g++-12`** if found on `PATH`. Install **`extra/gcc14`** on Arch for `/usr/bin/g++-14`, or set explicitly:
  ```bash
  export PRISMALAMA_CUDA_HOST_CXX=/usr/bin/g++-14
  makepkg -sfi
  ```
  After a failed CUDA compile, remove the **`build/`** directory so CMake does not reuse a bad **`CMAKE_CUDA_HOST_COMPILER`** cache entry.
- **Runtime:** if the package was built with CUDA, install **`cuda`** (or equivalent libcudart provider) on the target machine.

### Vulkan (built unconditionally by this PKGBUILD)

**`ggml-vulkan`** is built and installed **unconditionally** (no silent skip). **`makedepends`** includes **`glslang`** so CMake can find **`glslc`** (`ggml-vulkan` requires it). Vulkan is independent of **`AMDGPU_TARGETS`** and serves as a **fallback compute backend** when HIP/CUDA are unavailable or not selected at runtime.

## Layer Streaming (Default)

`OLLAMA_LAYER_STREAMING=1` (default) enables GGUF layer streaming:

- Streaming-capable backends can load/evict GGUF blocks under budget during inference
- If streaming interfaces are unavailable for a backend/model path, load falls back to standard behavior
- Models larger than RAM can run without loading entirely into memory

Disable by setting `OLLAMA_LAYER_STREAMING=0` if you want traditional mmap behavior.

## AirLLM (Opt-in)

With **`OLLAMA_USE_AIRLLM=0`** (this package’s default), **`runner/dispatch.go`** keeps **all** models on the native GGML runner — including layouts that would otherwise match Hugging Face or multipart heuristics. Set **`OLLAMA_USE_AIRLLM=1`** when you intentionally want the Python stack.

AirLLM handles Hugging Face safetensors and multi-part GGUF with layer-wise NVMe streaming where supported. To enable:

```bash
# In /etc/default/ollama:
OLLAMA_USE_AIRLLM=1

# Install Python deps for ollama user
sudo python3 -m pip install --break-system-packages transformers safetensors
sudo systemctl restart ollama
```

Check it works:

```bash
sudo -u ollama python3 -c "import transformers; print('OK')"
```

## Systemd Override (Persistent Across Upgrades)

Package upgrades overwrite `/etc/default/ollama` and `/usr/lib/systemd/system/ollama.service`. For persistent runtime config, use a drop-in:

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf << 'EOF'
[Service]
Environment=OLLAMA_KEEP_ALIVE=5m
Environment=OLLAMA_LAYER_STREAMING=1
EOF

sudo systemctl daemon-reload
sudo systemctl restart ollama
```

## Uninstall

```bash
sudo pacman -R prismalama-ollama
sudo userdel ollama 2>/dev/null || true
# Optional: rm -rf /nvme3/models /var/lib/ollama
```

## Versioning

- `pkgver` is **Prismalama's** version, not upstream Ollama's
- `epoch=1` ensures packages sort after legacy `0.18.x` builds
- Binary reports version as `X.Y.Z-rN-prismalama` via `-ldflags`

## Build Dependencies

```bash
sudo pacman -S --needed \
  rocm-hip-sdk \
  rocm-hip-runtime \
  go \
  cmake \
  ninja \
  gcc \
  vulkan-headers \
  vulkan-icd-loader
```
