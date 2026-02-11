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
SRC_DIR="${SCRIPT_DIR}/src"
BUILD_DIR="${SCRIPT_DIR}/build"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_step "Checking prerequisites..."
    
    local missing=()
    
    # Check for ROCm
    if ! command -v hipcc &> /dev/null; then
        missing+=("rocm-hip-sdk (hipcc)")
    else
        log_info "ROCm found: $(hipcc --version 2>&1 | head -1)"
    fi
    
    # Check for Go
    if ! command -v go &> /dev/null; then
        missing+=("go")
    else
        log_info "Go found: $(go version)"
    fi
    
    # Check for cmake
    if ! command -v cmake &> /dev/null; then
        missing+=("cmake")
    else
        log_info "CMake found: $(cmake --version | head -1)"
    fi
    
    # Check for git
    if ! command -v git &> /dev/null; then
        missing+=("git")
    fi
    
    # Detect GPU architecture
    if command -v rocm_agent_enumerator &> /dev/null; then
        DETECTED_ARCH=$(rocm_agent_enumerator 2>/dev/null | grep -v "gfx000" | head -1)
        if [ -n "$DETECTED_ARCH" ]; then
            log_info "Detected ROCm architecture: $DETECTED_ARCH"
            if [ "$DETECTED_ARCH" != "$ROCM_ARCH" ]; then
                log_warn "Detected architecture ($DETECTED_ARCH) differs from default ($ROCM_ARCH)"
                ROCM_ARCH="$DETECTED_ARCH"
            fi
        fi
    fi
    
    # Check Python dependencies
    log_step "Checking Python dependencies..."
    
    if ! command -v python3 &> /dev/null; then
        missing+=("python")
    else
        python3 --version
    fi
    
    local python_pkgs=("torch" "transformers" "safetensors" "numpy")
    for pkg in "${python_pkgs[@]}"; do
        if ! python3 -c "import $pkg" 2>/dev/null; then
            log_warn "Python package '$pkg' not found"
        else
            log_info "Python package '$pkg' found"
        fi
    done
    
    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing prerequisites: ${missing[*]}"
        echo ""
        echo "Install with:"
        echo "  sudo pacman -S ${missing[*]}"
        exit 1
    fi
    
    log_info "All prerequisites checked"
}

# Prepare sources
prepare_sources() {
    log_step "Preparing sources..."
    
    mkdir -p "$SRC_DIR"
    
    # Clone Ollama if not present
    if [ ! -d "$SRC_DIR/ollama" ]; then
        log_info "Cloning Ollama repository (v${PKG_VERSION})..."
        git clone --depth 1 --branch "v${PKG_VERSION}" https://github.com/ollama/ollama.git "$SRC_DIR/ollama"
    else
        log_info "Ollama source already exists"
    fi
    
    # Clone AirLLM if not present
    if [ ! -d "$SRC_DIR/airllm" ]; then
        log_info "Cloning AirLLM repository..."
        git clone --depth 1 https://github.com/lyogavin/AirLLM.git "$SRC_DIR/airllm"
    else
        log_info "AirLLM source already exists"
    fi
    
    # Initialize submodules
    cd "$SRC_DIR/ollama"
    if [ ! -f ".submodules_initialized" ]; then
        log_info "Initializing git submodules..."
        git submodule update --init --recursive
        touch .submodules_initialized
    fi
    
    cd "$SCRIPT_DIR"
    
    # Apply AirLLM integration
    log_step "Setting up AirLLM integration..."
    
    # Create airllmrunner directory and copy files
    mkdir -p "$SRC_DIR/ollama/runner/airllmrunner"
    
    if [ -f "${SCRIPT_DIR}/runner/airllmrunner/runner.go" ]; then
        cp "${SCRIPT_DIR}/runner/airllmrunner/"*.go "$SRC_DIR/ollama/runner/airllmrunner/"
        log_info "AirLLM runner Go files copied"
    fi
    
    if [ -f "${SCRIPT_DIR}/runner/airllmrunner/airllm_runner.py" ]; then
        mkdir -p "$SRC_DIR/ollama/runner/airllmrunner"
        cp "${SCRIPT_DIR}/runner/airllmrunner/airllm_runner.py" "$SRC_DIR/ollama/runner/airllmrunner/"
        log_info "AirLLM runner Python file copied"
    fi
    
    log_info "Sources prepared"
}

# Build Ollama
build_ollama() {
    log_step "Building Ollama with ROCm support..."
    
    cd "$SRC_DIR/ollama"
    
    # Set build environment
    export HIP_PATH="/opt/rocm"
    export HIPCXX="/opt/rocm/llvm/bin/clang++"
    export AMDGPU_TARGETS="${ROCM_ARCH}"
    export HSA_OVERRIDE_GFX_VERSION=11.0.0
    export GOFLAGS="-trimpath -buildmode=pie"
    export CGO_ENABLED=1
    
    # Clean previous build
    rm -rf build
    mkdir -p build
    
    # Build ggml with ROCm
    log_info "Configuring ggml with ROCm..."
    cd build
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -DLLAMA_CURL=ON \
        -DLLAMA_HIPBLAS=ON \
        -DLLAMA_CUDA=OFF \
        -DCMAKE_HIP_COMPILER_ROCM_ROOT="/opt/rocm" \
        -DAMDGPU_TARGETS="${ROCM_ARCH}" \
        -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
        -DCMAKE_C_COMPILER=gcc \
        2>&1 | tee cmake.log
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log_error "CMake configuration failed"
        exit 1
    fi
    
    log_info "Building ggml libraries..."
    cmake --build . --config Release -j$(nproc) 2>&1 | tee build.log
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log_error "Build failed"
        exit 1
    fi
    
    cd "$SRC_DIR/ollama"
    
    # Build Ollama binary
    log_info "Building Ollama binary..."
    go build \
        -tags="rocm" \
        -o "ollama-bin" \
        -ldflags="-w -s -X=github.com/ollama/ollama/version.Version=${PKG_VERSION}" \
        . 2>&1
    
    if [ $? -ne 0 ]; then
        log_error "Ollama binary build failed"
        exit 1
    fi
    
    log_info "Ollama build complete"
    
    # Verify binary
    if [ -f "ollama-bin" ]; then
        log_info "Binary created: $(ls -lh ollama-bin | awk '{print $5}')"
        file ollama-bin
    fi
}

# Prepare AirLLM
prepare_airllm() {
    log_step "Preparing AirLLM..."
    
    mkdir -p "$BUILD_DIR"
    
    if [ -d "$SRC_DIR/airllm/air_llm" ]; then
        rm -rf "$BUILD_DIR/airllm"
        cp -r "$SRC_DIR/airllm/air_llm" "$BUILD_DIR/airllm"
        log_info "AirLLM prepared"
    else
        log_error "AirLLM source not found"
        exit 1
    fi
}

# Create package structure
create_package_structure() {
    log_step "Creating package structure..."
    
    PKG_DIR="$BUILD_DIR/pkg"
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
    install -Dm755 "$SRC_DIR/ollama/ollama-bin" "$PKG_DIR/usr/bin/ollama"
    
    # Install libraries
    if [ -d "$SRC_DIR/ollama/build/lib/ollama" ]; then
        log_info "Installing Ollama libraries..."
        cp -r "$SRC_DIR/ollama/build/lib/ollama/"* "$PKG_DIR/usr/lib/ollama/"
    fi
    
    # Copy additional libraries
    if [ -d "$SRC_DIR/ollama/build" ]; then
        cd "$SRC_DIR/ollama/build"
        for lib in libggml*.so*; do
            if [ -e "$lib" ]; then
                cp -P "$lib" "$PKG_DIR/usr/lib/ollama/"
            fi
        done
        cd "$SCRIPT_DIR"
    fi
    
    # Install AirLLM
    cp -r "$BUILD_DIR/airllm" "$PKG_DIR/usr/share/ollama/"
    log_info "AirLLM installed"
    
    # Install AirLLM runner
    if [ -f "$SRC_DIR/ollama/runner/airllmrunner/airllm_runner.py" ]; then
        install -Dm755 "$SRC_DIR/ollama/runner/airllmrunner/airllm_runner.py" "$PKG_DIR/usr/share/ollama/airllm_runner.py"
    elif [ -f "${SCRIPT_DIR}/runner/airllmrunner/airllm_runner.py" ]; then
        install -Dm755 "${SCRIPT_DIR}/runner/airllmrunner/airllm_runner.py" "$PKG_DIR/usr/share/ollama/airllm_runner.py"
    fi
    
    # Install systemd service
    cat > "$PKG_DIR/usr/lib/systemd/system/ollama.service" << 'EOF'
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
    cat > "$PKG_DIR/usr/lib/sysusers.d/ollama.conf" << 'EOF'
u ollama - "Ollama service user" -
EOF

    # Install environment config
    cat > "$PKG_DIR/etc/default/ollama" << 'EOF'
# Ollama configuration for ROCm with AirLLM
export OLLAMA_MODELS="/run/media/piotro/CACHE1/airllm"
export OLLAMA_HOST="127.0.0.1:11434"
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export AIRLLM_COMPRESSION="4bit"
export AIRLLM_DEVICE="cuda:0"
export PYTHONPATH="/usr/share/ollama/airllm:/usr/share/ollama/airllm/air_llm:${PYTHONPATH}"

# ROCm specific
export HIP_VISIBLE_DEVICES=0
EOF

    # Install license
    install -Dm644 "$SRC_DIR/ollama/LICENSE" "$PKG_DIR/usr/share/licenses/${PKG_NAME}/LICENSE"
    
    # Create wrapper script
    cat > "$PKG_DIR/usr/bin/ollama-airllm" << 'EOF'
#!/bin/bash
# Ollama AirLLM wrapper

if [ -f /etc/default/ollama ]; then
    source /etc/default/ollama
fi

export OLLAMA_MODELS
export OLLAMA_HOST
export HSA_OVERRIDE_GFX_VERSION
export AIRLLM_COMPRESSION
export AIRLLM_DEVICE
export PYTHONPATH
export HIP_VISIBLE_DEVICES

exec /usr/bin/ollama "$@"
EOF
    chmod 755 "$PKG_DIR/usr/bin/ollama-airllm"
    
    log_info "Package structure created"
    
    # Display structure
    echo ""
    echo "Package contents:"
    find "$PKG_DIR" -type f | head -20
    echo "..."
}

# Build pacman package
build_pacman_package() {
    log_step "Building pacman package..."
    
    # Create .SRCINFO
    cat > "$BUILD_DIR/.SRCINFO" << EOF
pkgbase = ${PKG_NAME}
	pkgname = ${PKG_NAME}
	pkgver = ${PKG_VERSION}
	pkgrel = ${PKG_REL}
	url = https://github.com/ollama/ollama
	arch = x86_64
	license = MIT
	depends = glibc
	depends = zlib
	depends = gcc-libs
	depends = rocm-hip-sdk
	depends = python
	depends = python-pytorch-rocm
	depends = python-numpy
	depends = python-safetensors
	source = PKGBUILD

pkgname = ${PKG_NAME}
EOF

    # Create the package
    PKG_FILE="${PKG_NAME}-${PKG_VERSION}-${PKG_REL}-x86_64.pkg.tar.zst"
    
    cd "$BUILD_DIR"
    
    # Create .MTREE
    if command -v bsdtar &> /dev/null; then
        # Create package with proper metadata
        cd pkg
        bsdtar -czf ../.MTREE --format=mtree \
            --options='!all,use-set,type,uid,gid,mode,time,size,md5,sha256,link' \
            . 2>/dev/null || true
        cd ..
    fi
    
    # Create the package tarball
    cd pkg
    if command -v zstd &> /dev/null; then
        tar -cf - . | zstd -19 -T0 > "${SCRIPT_DIR}/${PKG_FILE}"
        log_info "Package created with zstd compression: ${PKG_FILE}"
    else
        tar -czf "${SCRIPT_DIR}/${PKG_FILE%.zst}.gz" .
        log_warn "Package created with gzip (zstd not available): ${PKG_FILE%.zst}.gz"
    fi
    
    cd "$SCRIPT_DIR"
    
    # Verify package
    if [ -f "$PKG_FILE" ]; then
        log_info "Package size: $(ls -lh "$PKG_FILE" | awk '{print $5}')"
        
        # Test package contents
        if command -v tar &> /dev/null; then
            log_info "Package contents summary:"
            tar -tf "$PKG_FILE" | wc -l | xargs echo "  Total files:"
            tar -tf "$PKG_FILE" | grep "^usr/bin/" | head -5 | sed 's/^/  /'
        fi
    else
        log_error "Package creation failed"
        exit 1
    fi
}

# Install locally
install_local() {
    log_step "Installing package locally..."
    
    PKG_FILE="${PKG_NAME}-${PKG_VERSION}-${PKG_REL}-x86_64.pkg.tar.zst"
    
    if [ -f "$PKG_FILE" ]; then
        log_info "Installing $PKG_FILE..."
        sudo pacman -U "$PKG_FILE"
    else
        log_warn "No package found, installing files directly..."
        
        # Copy files manually
        sudo cp -r "$BUILD_DIR/pkg/"* /
        
        # Set up sysusers
        sudo systemd-sysusers ollama.conf 2>/dev/null || true
        
        # Reload systemd
        sudo systemctl daemon-reload
        
        # Create directories
        sudo install -dm755 -o ollama -g ollama /run/media/piotro/CACHE1/airllm 2>/dev/null || true
        sudo install -dm755 -o ollama -g ollama /var/lib/ollama 2>/dev/null || true
    fi
    
    log_info "Installation complete"
}

# Test installation
test_installation() {
    log_step "Testing installation..."
    
    # Check binary
    if [ -f "/usr/bin/ollama" ] || [ -f "$BUILD_DIR/pkg/usr/bin/ollama" ]; then
        log_info "Binary exists"
    else
        log_error "Binary not found"
        return 1
    fi
    
    # Check libraries
    if ls "$BUILD_DIR/pkg/usr/lib/ollama/"libggml*.so* &> /dev/null; then
        log_info "Libraries installed"
        ls "$BUILD_DIR/pkg/usr/lib/ollama/"libggml*.so* | wc -l | xargs echo "  Library count:"
    else
        log_warn "Libraries may be missing"
    fi
    
    # Check AirLLM
    if [ -d "$BUILD_DIR/pkg/usr/share/ollama/airllm" ]; then
        log_info "AirLLM installed"
    else
        log_warn "AirLLM not found"
    fi
    
    log_info "Basic tests passed"
}

# Clean up
cleanup() {
    log_step "Cleaning up..."
    
    read -p "Remove build directory? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$BUILD_DIR"
        log_info "Build directory removed"
    fi
}

# Show help
show_help() {
    cat << EOF
Ollama AirLLM ROCm Build Script

Usage: $0 [command]

Commands:
    check       Check prerequisites
    prepare     Prepare sources (clone repositories)
    build       Build Ollama from source
    package     Create package structure
    pkgbuild    Build pacman package
    test        Test the build
    install     Install locally
    clean       Clean build directory
    all         Run complete build process (default)
    help        Show this help message

Environment:
    ROCM_ARCH   ROCm architecture (default: gfx1100)
    
Examples:
    $0              # Full build
    $0 check        # Check prerequisites only
    $0 build        # Build only
    $0 install      # Install locally

EOF
}

# Main function
main() {
    case "${1:-all}" in
        check)
            check_prerequisites
            ;;
        prepare)
            check_prerequisites
            prepare_sources
            ;;
        build)
            prepare_sources
            build_ollama
            ;;
        package)
            prepare_airllm
            create_package_structure
            ;;
        pkgbuild)
            build_pacman_package
            ;;
        test)
            test_installation
            ;;
        install)
            install_local
            ;;
        clean)
            cleanup
            ;;
        all)
            check_prerequisites
            prepare_sources
            build_ollama
            prepare_airllm
            create_package_structure
            build_pacman_package
            test_installation
            
            echo ""
            log_info "Build complete!"
            echo ""
            echo "Package: ${PKG_NAME}-${PKG_VERSION}-${PKG_REL}-x86_64.pkg.tar.zst"
            echo ""
            echo "To install:"
            echo "  sudo $0 install"
            echo "  or: sudo pacman -U ${PKG_NAME}-${PKG_VERSION}-${PKG_REL}-x86_64.pkg.tar.zst"
            echo ""
            echo "After installation:"
            echo "  sudo systemctl start ollama"
            echo "  sudo systemctl enable ollama"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main
main "$@"
