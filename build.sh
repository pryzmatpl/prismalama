#!/bin/bash
# Simple build script for ollama-airllm package

set -e

BUILD_DIR="build_ollama_airllm"
PKG_VERSION="v0.4.1.r5053.4b15df6b"
PKG_NAME="ollama-airllm"
PKG_FILE="${PKG_NAME}-${PKG_VERSION}-1-x86_64.pkg.tar.zst"

echo "Building ${PKG_NAME}..."

# Create build directory
mkdir -p "$BUILD_DIR"

# Clone mlx-c headers
echo "Cloning mlx-c headers..."
git clone --depth 1 --branch "$(cat MLX_VERSION)" https://github.com/ml-explore/mlx-c.git build/_deps/mlx-c-src

# Build ollama binary
echo "Building ollama binary..."
export GOFLAGS="-trimpath -buildmode=pie"
export CGO_ENABLED=1
export CGO_CFLAGS="-I$(pwd)/build/_deps/mlx-c-src"
export CGO_CPPFLAGS="-DMLX_ENGINE=OFF -DGGML_HIP=ON"
export LDFLAGS="-w -s -X=github.com/ollama/ollama/version.Version=${PKG_VERSION}"
go build -tags="" -o "$BUILD_DIR/ollama" -ldflags="-w -s -X=github.com/ollama/ollama/version.Version=${PKG_VERSION}" .

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
cp "$BUILD_DIR/ollama" "$BUILD_DIR/usr/bin/"
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

 # Copy AirLLM
 cp -r airllm-clean/air_llm "$BUILD_DIR/usr/share/ollama/airllm"

# Copy license
cp LICENSE "$BUILD_DIR/usr/share/licenses/$PKG_NAME/"


# Create install script
cat > "$BUILD_DIR/ollama-airllm.install" << 'EOF'
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
  LANG=C makepkg -C -c -f -g -s --packagelist

echo "Package created: $PKG_FILE"
echo ""
echo "To install:"
echo "  sudo pacman -U $PKG_FILE"
echo ""
echo "After installation:"
echo "  sudo systemctl start ollama"
echo "  sudo systemctl enable ollama"
