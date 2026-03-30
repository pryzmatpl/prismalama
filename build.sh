#!/bin/bash
# Simple build script for ollama-airllm package

set -e

BUILD_DIR="build_ollama_airllm"
PKG_VERSION="v0.4.1.r5053.4b15df6b"
PKG_NAME="ollama-airllm-rocm"
PKG_REL="1"

if [ -f PKGBUILD ]; then
    PKG_NAME=$(awk -F= '/^pkgname=/{print $2; exit}' PKGBUILD)
    PKG_VERSION=$(awk -F= '/^pkgver=/{print $2; exit}' PKGBUILD)
    PKG_REL=$(awk -F= '/^pkgrel=/{print $2; exit}' PKGBUILD)
fi

PKG_FILE="${PKG_NAME}-${PKG_VERSION}-${PKG_REL}-x86_64.pkg.tar.zst"

echo "Building ${PKG_NAME}..."

# Create build directory
mkdir -p "$BUILD_DIR"

# Use local mlx-c headers when available (offline)
MLX_C_PATH="${MLX_C_PATH:-$(pwd)/build/_deps/mlx-c-src}"
if [ -d "$MLX_C_PATH" ]; then
    echo "Using local mlx-c headers at: $MLX_C_PATH"
    MLX_CFLAGS="-I$MLX_C_PATH"
else
    echo "Warning: mlx-c headers not found at $MLX_C_PATH; building without MLX headers"
    MLX_CFLAGS=""
fi

# Build ollama binary
echo "Building ollama binary..."
export GOFLAGS="-trimpath -buildmode=pie"
export CGO_ENABLED=1
export CGO_CFLAGS="$MLX_CFLAGS"
export CGO_CPPFLAGS="-DMLX_ENGINE=OFF -DGGML_HIP=ON"
export LDFLAGS="-w -s -X=github.com/ollama/ollama/version.Version=${PKG_VERSION}"
go build -tags="" -buildvcs=false -o "$BUILD_DIR/ollama-bin" -ldflags="-w -s -X=github.com/ollama/ollama/version.Version=${PKG_VERSION}" .

# Create package structure
echo "Creating package structure..."
mkdir -p "$BUILD_DIR/usr/bin"
mkdir -p "$BUILD_DIR/usr/lib/systemd/system"
mkdir -p "$BUILD_DIR/usr/lib/sysusers.d"
mkdir -p "$BUILD_DIR/etc/default"
mkdir -p "$BUILD_DIR/usr/share/ollama"
mkdir -p "$BUILD_DIR/sda2/airllm"
mkdir -p "$BUILD_DIR/usr/share/licenses/$PKG_NAME"

# Copy files
echo "Copying files..."
cp "$BUILD_DIR/ollama-bin" "$BUILD_DIR/usr/bin/ollama"
chmod 755 "$BUILD_DIR/usr/bin/ollama"

# Create systemd service
cat > "$BUILD_DIR/usr/lib/systemd/system/ollama.service" << 'EOF'
[Unit]
Description=Ollama Server with AirLLM Integration
Documentation=https://github.com/ollama/ollama
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=ollama
EnvironmentFile=/etc/default/ollama
ExecStart=/usr/bin/ollama serve
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
Environment="OLLAMA_NUM_PARALLEL=1"
Restart=always
RestartSec=3

NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/sda2/airllm /var/lib/ollama
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

# Create sysusers config
cat > "$BUILD_DIR/usr/lib/sysusers.d/ollama.conf" << 'EOF'
u ollama - "Ollama service user" -
EOF

# Create environment config
cat > "$BUILD_DIR/etc/default/ollama" << 'EOF'
export OLLAMA_MODELS="/sda2/airllm"
EOF

# AirLLM — two variants exist in this repo:
#   src/airllm/air_llm/  (git submodule) — NVME weight streaming, CUDA/ROCm, used by PKGBUILD
#   airllm-clean/air_llm/                    — MLX for Apple Silicon ONLY, NOT for Linux/ROCm
# Use src/airllm for the Linux/ROCm packaged build.
if [ -d "src/airllm/air_llm" ]; then
    cp -r src/airllm/air_llm "$BUILD_DIR/usr/share/ollama/airllm"
    echo "AirLLM: using src/airllm/air_llm (NVME weight streaming, CUDA/ROCm)"
else
    echo "ERROR: src/airllm/air_llm not found. Cannot proceed — airllm-clean is NOT compatible with Linux/ROCm."
    exit 1
fi

# Copy license
cp LICENSE "$BUILD_DIR/usr/share/licenses/$PKG_NAME/"


# Create install script
cat > "$BUILD_DIR/${PKG_NAME}.install" << 'EOF'
post_install() {
  systemd-sysusers ollama.conf
  chown -R ollama:ollama /sda2/airllm 2>/dev/null || true
  
  echo ""
  echo "Ollama with AirLLM integration has been installed!"
  echo ""
  echo "Models directory: /sda2/airllm"
  echo ""
  echo "To start the service:"
  echo "  sudo systemctl start ollama"
  echo ""
  echo "To enable on boot:"
  echo "  sudo systemctl enable ollama"
  echo ""
  echo "Configuration file: /etc/default/ollama"
  echo ""
  echo "AirLLM integration is available at: /usr/share/ollama/airllm"
  echo ""
}

post_upgrade() {
  post_install
}

pre_remove() {
  systemctl disable --now ollama 2>/dev/null || true
}
EOF

# Create package
  echo "Creating package: $PKG_FILE"
  cd "$BUILD_DIR"
  cp ../PKGBUILD .
  if [ -f "../airllm_runner.py" ]; then
    cp ../airllm_runner.py .
  fi
  if [ -f "../airllm.patch" ]; then
    cp ../airllm.patch .
  fi
  rm -rf "$BUILD_DIR/ollama" "$BUILD_DIR/src" "$BUILD_DIR/pkg"

  ROOT_DIR="$(pwd)/.."
  LOCAL_SOURCES="source=(\"ollama::file://${ROOT_DIR}/src/ollama\" \"airllm::file://${ROOT_DIR}/src/airllm\" \"ollama-airllm-rocm.install\" \"airllm_runner.py\" \"airllm.patch\")"
  LOCAL_SUMS="sha256sums=('SKIP' 'SKIP' 'SKIP' 'SKIP' 'SKIP')"
  awk -v src="$LOCAL_SOURCES" -v sums="$LOCAL_SUMS" '
    BEGIN {skip=0; skip2=0}
    /^source=\(/ {print src; skip=1; next}
    skip && /^\)/ {skip=0; next}
    skip {next}
    /^sha256sums=\(/ {print sums; skip2=1; next}
    skip2 && /^\)/ {skip2=0; next}
    skip2 {next}
    {print}
  ' PKGBUILD > PKGBUILD.local

  LANG=C makepkg -C -c -f -s -p PKGBUILD.local

echo "Package created: $PKG_FILE"
echo ""
echo "To install:"
echo "  sudo pacman -U $PKG_FILE"
echo ""
echo "After installation:"
echo "  sudo systemctl start ollama"
echo "  sudo systemctl enable ollama"
