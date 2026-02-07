#!/bin/bash
# Simple build script for ollama-airllm package

set -e

BUILD_DIR="build_ollama_airllm"
PKG_VERSION="v0.4.1.r5052.e102207f"
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
export LDFLAGS="-w -s -X=github.com/ollama/ollama/version.Version=${PKG_VERSION}"
go build -tags mlx -o "$BUILD_DIR/ollama" .

# Create package structure
echo "Creating package structure..."
mkdir -p "$BUILD_DIR/usr/bin"
mkdir -p "$BUILD_DIR/usr/lib/systemd/system"
mkdir -p "$BUILD_DIR/usr/lib/sysusers.d"
mkdir -p "$BUILD_DIR/etc/default"
mkdir -p "$BUILD_DIR/usr/share/ollama"
mkdir -p "$BUILD_DIR/run/media/piotro/CACHE/airllm"
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
ReadWritePaths=/run/media/piotro/CACHE/airllm /var/lib/ollama
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
export OLLAMA_MODELS="/run/media/piotro/CACHE/airllm"
EOF

# Copy AirLLM
cp -r airllm "$BUILD_DIR/usr/share/ollama/"

# Copy license
cp LICENSE "$BUILD_DIR/usr/share/licenses/$PKG_NAME/"

# Create .MTREE file
echo "Creating package metadata..."
cd "$BUILD_DIR"
LANG=C bsdtar -czf .MTREE --format=mtree \
    --options='!all,use-set,type,uid,gid,mode,time,size,sha256,link' \
    usr etc run

# Create package
echo "Creating package: $PKG_FILE"
LANG=C bsdtar -czf "../$PKG_FILE" .MTREE usr etc run

cd ..

echo "Package created: $PKG_FILE"
echo ""
echo "To install:"
echo "  sudo pacman -U $PKG_FILE"
echo ""
echo "After installation:"
echo "  sudo systemctl start ollama"
echo "  sudo systemctl enable ollama"
