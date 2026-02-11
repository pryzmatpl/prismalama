#!/bin/bash
# Build script for Ollama with AirLLM and ROCm support
# For AMD RX 7900 XTX (gfx1100)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
PKG_NAME="ollama-airllm-rocm"
PKG_VERSION="0.5.7"
PKG_REL="1"
ROCM_ARCH="gfx1100"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check for ROCm
    if ! command -v hipcc &> /dev/null; then
        log_error "ROCm hipcc not found. Please install rocm-hip-sdk"
        exit 1
    fi
    
    # Check for Go
    if ! command -v go &> /dev/null; then
        log_error "Go not found. Please install Go"
        exit 1
    fi
    
    # Check for cmake
    if ! command -v cmake &> /dev/null; then
        log_error "cmake not found. Please install cmake"
        exit 1
    fi
    
    # Detect GPU architecture
    DETECTED_ARCH=$(rocm_agent_enumerator 2>/dev/null | grep -v "gfx000" | head -1)
    if [ -n "$DETECTED_ARCH" ]; then
        log_info "Detected ROCm architecture: $DETECTED_ARCH"
        if [ "$DETECTED_ARCH" != "$ROCM_ARCH" ]; then
            log_warn "Detected architecture ($DETECTED_ARCH) differs from expected ($ROCM_ARCH)"
            log_warn "Adjusting ROCM_ARCH to $DETECTED_ARCH"
            ROCM_ARCH="$DETECTED_ARCH"
        fi
    else
        log_warn "Could not detect GPU architecture, using default: $ROCM_ARCH"
    fi
    
    # Check Python dependencies
    log_info "Checking Python dependencies..."
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" || log_warn "PyTorch not installed"
    python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')" || log_warn "transformers not installed"
    
    log_info "All prerequisites checked"
}

# Prepare sources
prepare_sources() {
    log_info "Preparing sources..."
    
    # Clone Ollama if not present
    if [ ! -d "src/ollama" ]; then
        log_info "Cloning Ollama repository..."
        mkdir -p src
        git clone --depth 1 --branch "v${PKG_VERSION}" https://github.com/ollama/ollama.git src/ollama
    fi
    
    # Clone AirLLM if not present
    if [ ! -d "src/airllm" ]; then
        log_info "Cloning AirLLM repository..."
        git clone --depth 1 https://github.com/lyogavin/AirLLM.git src/airllm
    fi
    
    # Initialize submodules
    cd src/ollama
    git submodule update --init --recursive
    cd "$SCRIPT_DIR"
    
    log_info "Sources prepared"
}

# Build Ollama
build_ollama() {
    log_info "Building Ollama with ROCm support..."
    
    cd src/ollama
    
    # Set build environment
    export HIP_PATH="/opt/rocm"
    export HIPCXX="/opt/rocm/llvm/bin/clang++"
    export AMDGPU_TARGETS="${ROCM_ARCH}"
    export HSA_OVERRIDE_GFX_VERSION=11.0.0
    export GOFLAGS="-trimpath -buildmode=pie"
    export CGO_ENABLED=1
    
    # Build ggml with ROCm
    log_info "Building ggml backend..."
    rm -rf build
    mkdir -p build
    cd build
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -DLLAMA_CURL=ON \
        -DLLAMA_HIPBLAS=ON \
        -DLLAMA_CUDA=OFF \
        -DCMAKE_HIP_COMPILER_ROCM_ROOT="/opt/rocm" \
        -DAMDGPU_TARGETS="${ROCM_ARCH}" \
        -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
    
    cmake --build . --config Release -j$(nproc)
    
    cd ..  # Back to src/ollama
    
    # Build Ollama binary
    log_info "Building Ollama binary..."
    go build \
        -tags="rocm" \
        -o "${SCRIPT_DIR}/build/ollama-bin" \
        -ldflags="-w -s -X=github.com/ollama/ollama/version.Version=${PKG_VERSION}" \
        .
    
    cd "$SCRIPT_DIR"
    log_info "Ollama build complete"
}

# Prepare AirLLM
prepare_airllm() {
    log_info "Preparing AirLLM..."
    
    # Copy AirLLM to build directory
    if [ -d "src/airllm/air_llm" ]; then
        rm -rf build/airllm
        cp -r src/airllm/air_llm build/airllm
        log_info "AirLLM copied to build directory"
    else
        log_error "AirLLM source not found"
        exit 1
    fi
}

# Create package structure
create_package() {
    log_info "Creating package structure..."
    
    PKG_DIR="build/pkg"
    rm -rf "$PKG_DIR"
    mkdir -p "$PKG_DIR"
    
    # Create directories
    install -dm755 "$PKG_DIR/usr/bin"
    install -dm755 "$PKG_DIR/usr/lib/ollama"
    install -dm755 "$PKG_DIR/usr/lib/systemd/system"
    install -dm755 "$PKG_DIR/usr/lib/sysusers.d"
    install -dm755 "$PKG_DIR/etc/default"
    install -dm755 "$PKG_DIR/usr/share/ollama"
    install -dm755 "$PKG_DIR/usr/share/licenses/${PKG_NAME}"
    
    # Install binary
    install -Dm755 build/ollama-bin "$PKG_DIR/usr/bin/ollama"
    
    # Install libraries
    if [ -d "src/ollama/build/lib/ollama" ]; then
        cp -r src/ollama/build/lib/ollama/* "$PKG_DIR/usr/lib/ollama/"
    fi
    
    # Install AirLLM
    cp -r build/airllm "$PKG_DIR/usr/share/ollama/"
    
    # Install AirLLM runner
    install -Dm755 runner/airllmrunner/airllm_runner.py "$PKG_DIR/usr/share/ollama/airllm_runner.py"
    
    # Install systemd service
    cat > "$PKG_DIR/usr/lib/systemd/system/ollama.service" << EOF
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

# Security settings
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/run/media/piotro/CACHE1/airllm /run/media/piotro/CACHE/airllm /var/lib/ollama /tmp
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

    # Install sysusers config
    cat > "$PKG_DIR/usr/lib/sysusers.d/ollama.conf" << EOF
u ollama - "Ollama service user" -
EOF

    # Install environment config
    cat > "$PKG_DIR/etc/default/ollama" << EOF
# Ollama configuration for ROCm with AirLLM
export OLLAMA_MODELS="/run/media/piotro/CACHE1/airllm"
export OLLAMA_HOST="127.0.0.1:11434"
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export AIRLLM_COMPRESSION="4bit"
export AIRLLM_DEVICE="cuda:0"
export PYTHONPATH="/usr/share/ollama/airllm:/usr/share/ollama/airllm/air_llm:\$PYTHONPATH"

# ROCm specific
export HIP_VISIBLE_DEVICES=0
EOF

    # Install license
    install -Dm644 src/ollama/LICENSE "$PKG_DIR/usr/share/licenses/${PKG_NAME}/LICENSE"
    
    log_info "Package structure created"
}

# Build pacman package
build_pkg() {
    log_info "Building pacman package..."
    
    # Copy PKGBUILD to build directory
    cp PKGBUILD build/
    cp ollama-airllm-rocm.install build/
    
    # Update PKGBUILD with correct version
    sed -i "s/^pkgver=.*/pkgver=${PKG_VERSION}/" build/PKGBUILD
    
    cd build
    
    # Create src tarball
    mkdir -p src
    cp -r pkg/* src/ 2>/dev/null || true
    
    # Generate SRCINFO
    makepkg --printsrcinfo > .SRCINFO 2>/dev/null || true
    
    # Build package
    if command -v makepkg &> /dev/null; then
        makepkg -f 2>/dev/null || {
            log_warn "makepkg failed, creating manual package..."
            create_manual_package
        }
    else
        log_warn "makepkg not found, creating manual package..."
        create_manual_package
    fi
    
    cd "$SCRIPT_DIR"
}

# Create manual package (without makepkg)
create_manual_package() {
    log_info "Creating manual pacman package..."
    
    PKG_FILE="${PKG_NAME}-${PKG_VERSION}-${PKG_REL}-x86_64.pkg.tar.zst"
    
    # Create .MTREE
    cd pkg
    find . -type f -o -type l | sort | while read file; do
        file="${file#.}"
        [ -n "$file" ] && echo "$file"
    done > ../.files
    
    # Create tarball
    tar -czf "../${PKG_FILE%.zst}.gz" -C . .
    
    cd "$SCRIPT_DIR/build"
    
    # Compress with zstd
    if command -v zstd &> /dev/null; then
        zstd -19 -T0 "${PKG_FILE%.zst}.gz" -o "$SCRIPT_DIR/${PKG_FILE}"
        rm -f "${PKG_FILE%.zst}.gz"
    else
        mv "${PKG_FILE%.zst}.gz" "$SCRIPT_DIR/${PKG_FILE%.zst}.gz"
        PKG_FILE="${PKG_FILE%.zst}.gz"
    fi
    
    log_info "Package created: $PKG_FILE"
}

# Install package locally
install_local() {
    log_info "Installing package locally..."
    
    # Check if we have a built package
    if ls *.pkg.tar.* 1> /dev/null 2>&1; then
        sudo pacman -U *.pkg.tar.*
    else
        log_warn "No package found, copying files directly..."
        
        # Copy files manually
        sudo cp -r build/pkg/* /
        sudo systemd-sysusers ollama.conf 2>/dev/null || true
        sudo systemctl daemon-reload
        
        # Create directories
        sudo install -dm755 -o ollama -g ollama /run/media/piotro/CACHE1/airllm 2>/dev/null || true
        sudo install -dm755 -o ollama -g ollama /var/lib/ollama 2>/dev/null || true
    fi
    
    log_info "Installation complete"
}

# Main
main() {
    case "${1:-}" in
        check)
            check_prerequisites
            ;;
        prepare)
            check_prerequisites
            prepare_sources
            ;;
        build)
            check_prerequisites
            prepare_sources
            build_ollama
            prepare_airllm
            create_package
            ;;
        package)
            create_package
            build_pkg
            ;;
        install)
            install_local
            ;;
        all|"")
            check_prerequisites
            prepare_sources
            build_ollama
            prepare_airllm
            create_package
            build_pkg
            log_info "Build complete! Package is ready."
            echo ""
            echo "To install:"
            echo "  sudo ./build-pkg.sh install"
            echo "  or: sudo pacman -U ${PKG_NAME}-${PKG_VERSION}-${PKG_REL}-x86_64.pkg.tar.zst"
            ;;
        *)
            echo "Usage: $0 [check|prepare|build|package|install|all]"
            echo ""
            echo "Commands:"
            echo "  check    - Check prerequisites"
            echo "  prepare  - Prepare sources"
            echo "  build    - Build Ollama"
            echo "  package  - Create package"
            echo "  install  - Install locally"
            echo "  all      - Do everything (default)"
            ;;
    esac
}

main "$@"
