# Prismalama Arch Package (`PKGBUILD`)

Builds **this** Prismalama tree (prismallama.cpp/GGML via CMake, Go `ollama` binary, AirLLM assets) into an Arch Linux package — **not** upstream Ollama tarballs.

## What's Installed

| Path | Contents |
|------|----------|
| `/usr/bin/ollama` | Prismalama binary (Go + GGML/Vulkan) |
| `/usr/lib/ollama/rocm/` | GGML backends: CPU, HIP (ROCm), Vulkan |
| `/usr/share/ollama/airllm_runner.py` | AirLLM Python runner |
| `/usr/share/ollama/airllm/` | AirLLM Python package (if `src/airllm/air_llm` exists) |
| `/etc/default/ollama` | Environment variables |
| `/usr/lib/systemd/system/ollama.service` | Systemd service |

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
HIP_VISIBLE_DEVICES=0
HSA_OVERRIDE_GFX_VERSION=11.0.0
```

## Build

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

### GPU ISA

Default is `gfx1100` (RX 7900 class). Override before build:

```bash
export PRISMALAMA_AMDGPU_TARGETS=gfx1030  # RDNA2 (RX 6xxx/7xxx)
makepkg -sfi
```

## Layer Streaming (Default)

`OLLAMA_LAYER_STREAMING=1` (default) enables GGUF layer streaming:

- GGUF blocks are loaded from NVMe on-demand during inference
- Previous blocks are evicted to stay within the 4 GiB streaming budget
- Models larger than RAM can run without loading entirely into memory

Disable by setting `OLLAMA_LAYER_STREAMING=0` if you want traditional mmap behavior.

## AirLLM (Opt-in)

AirLLM handles Hugging Face safetensors and multi-part GGUF with true layer-wise NVMe streaming. To enable:

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
