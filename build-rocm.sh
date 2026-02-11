#!/bin/bash
set -e

BUILD_DIR="build_ollama_airllm_rocm"
PKG_VERSION="v0.4.1.r5053.4b15df6b"
PKG_NAME="ollama-airllm-rocm"
PKG_FILE="${PKG_NAME}-${PKG_VERSION}-1-x86_64.pkg.tar.zst"

echo "Building ${PKG_NAME} with ROCm support..."

# Check for ROCm/HIP
if ! command -v hipcc &> /dev/null; then
    echo "ERROR: hipcc not found. Please install ROCm first."
    exit 1
fi

# Detect GPU architecture
echo "Detecting GPU architecture..."
ROCM_ARCH=$(rocm_agent_enumerator 2>/dev/null | head -1)
if [ -z "$ROCM_ARCH" ]; then
    echo "WARNING: Could not detect ROCm architecture, using gfx1100"
    ROCM_ARCH="gfx1100"
fi
echo "Using ROCm architecture: $ROCM_ARCH"

# Create build directory
mkdir -p "$BUILD_DIR"

# Build ggml backend with ROCm using cmake
echo "Building ggml backend with ROCm support..."
rm -rf build
mkdir -p build
cd build

export HIP_PATH="$(hipconfig -R)"
export HIPCXX="$(hipconfig -l)/clang"
export AMDGPU_TARGETS="$ROCM_ARCH"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLX_ENGINE=OFF \
    -DLLAMA_CURL=ON \
    -DLLAMA_HIPBLAS=ON \
    -DLLAMA_CUDA=OFF \
    -DCMAKE_HIP_COMPILER_ROCM_ROOT="$HIP_PATH" \
    -DCMAKE_INSTALL_PREFIX=$(pwd)/../"$BUILD_DIR"/usr

cmake --build . --config Release -j$(nproc)

cd ..

# Build ollama binary
echo "Building ollama binary..."
export GOFLAGS="-trimpath -buildmode=pie"
export CGO_ENABLED=1
export LDFLAGS="-w -s -X=github.com/ollama/ollama/version.Version=${PKG_VERSION}"
go build -tags="" -o "$BUILD_DIR/ollama" -ldflags="-w -s -X=github.com/ollama/ollama/version.Version=${PKG_VERSION}" .

# Create package structure
echo "Creating package structure..."
mkdir -p "$BUILD_DIR/usr/bin"
mkdir -p "$BUILD_DIR/usr/lib/ollama"
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

# Copy ROCm libraries from build
if [ -d "build/lib/ollama" ]; then
    cp -r build/lib/ollama/* "$BUILD_DIR/usr/lib/ollama/" 2>/dev/null || true
fi

# Create systemd service
cat > "$BUILD_DIR/usr/lib/systemd/system/ollama.service" << 'EOF'
[Unit]
Description=Ollama Server with AirLLM Integration (ROCm)
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
ReadWritePaths=/run/media/piotro/CACHE1/airllm /run/media/piotro/CACHE/airllm /var/lib/ollama
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

# Create sysusers config
cat > "$BUILD_DIR/usr/lib/sysusers.d/ollama.conf" << 'EOF'
u ollama - "Ollama service user" -
EOF

# Create environment config with ROCm settings
cat > "$BUILD_DIR/etc/default/ollama" << 'EOF'
export OLLAMA_MODELS="/run/media/piotro/CACHE1/airllm"
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export AIRLLM_COMPRESSION="4bit"
export PYTHONPATH="/usr/share/ollama/airllm:$PYTHONPATH"
EOF

# Copy AirLLM
if [ -d "airllm-clean/air_llm" ]; then
    cp -r airllm-clean/air_llm "$BUILD_DIR/usr/share/ollama/airllm"
fi

# Also copy from submodule if available
if [ -d "airllm/air_llm" ]; then
    cp -r airllm/air_llm/airllm "$BUILD_DIR/usr/share/ollama/airllm/" 2>/dev/null || true
fi

# Copy AirLLM Python runner
cp runner/airllmrunner/airllm_runner.py "$BUILD_DIR/usr/share/ollama/"
chmod 755 "$BUILD_DIR/usr/share/ollama/airllm_runner.py"

# Copy license
cp LICENSE "$BUILD_DIR/usr/share/licenses/$PKG_NAME/"

# Create install script
cat > "$BUILD_DIR/ollama-airllm-rocm.install" << 'EOF'
post_install() {
  systemd-sysusers ollama.conf
  chown -R ollama:ollama /run/media/piotro/CACHE1/airllm 2>/dev/null || true
  chown -R ollama:ollama /run/media/piotro/CACHE/airllm 2>/dev/null || true
  
  echo ""
  echo "Ollama with AirLLM and ROCm integration has been installed!"
  echo ""
  echo "Models directory: /run/media/piotro/CACHE1/airllm"
  echo ""
  echo "AirLLM automatically handles large models by loading layers on-demand."
  echo "Models with safetensors format (like GLM-4.7) will use AirLLM automatically."
  echo ""
  echo "To start the service:"
  echo "  sudo systemctl start ollama"
  echo ""
  echo "To enable on boot:"
  echo "  sudo systemctl enable ollama"
  echo ""
  echo "Configuration file: /etc/default/ollama"
  echo ""
  echo "AirLLM Python package: /usr/share/ollama/airllm"
  echo "AirLLM Runner: /usr/share/ollama/airllm_runner.py"
  echo ""
  echo "ROCm GPU acceleration is enabled for gfx1100 (7900 XTX)."
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
sed -i 's/pkgname=ollama-airllm/pkgname=ollama-airllm-rocm/' PKGBUILD

# Create .SRCINFO for makepkg
makepkg --printsrcinfo > .SRCINFO

# Actually build the package (remove --packagelist flag)
LANG=C makepkg -f

# Move package to parent directory
mv *.pkg.tar.zst ../ 2>/dev/null || true

cd ..
echo "Package created: $PKG_FILE"
echo ""
echo "To install:"
echo "  sudo pacman -U $PKG_FILE"
echo ""
echo "After installation:"
echo "  sudo systemctl start ollama"
echo "  sudo systemctl enable ollama"
