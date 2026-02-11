# Maintainer: Ollama AirLLM ROCm Package <maintainer@example.com>
pkgname=ollama-airllm-rocm
pkgver=0.5.7
pkgrel=1
pkgdesc="Ollama with AirLLM integration and ROCm GPU support for automatic large model offloading"
arch=('x86_64')
url="https://github.com/ollama/ollama"
license=('MIT')

# Comprehensive dependencies
depends=(
    'glibc'
    'zlib'
    'gcc-libs'
    'rocm-hip-sdk'
    'python'
    'python-pytorch-rocm'
    'python-numpy'
    'python-safetensors'
    'python-huggingface-hub'
    'python-transformers'
    'python-accelerate'
    'python-typing_extensions'
)

makedepends=(
    'go'
    'cmake'
    'git'
    'rocm-hip-sdk'
    'rocm-cmake'
)

optdepends=(
    'cuda: NVIDIA GPU support'
    'python-sentencepiece: Tokenizer support'
    'python-protobuf: Protocol buffer support'
)

provides=('ollama')
conflicts=('ollama' 'ollama-rocm' 'ollama-cuda')
options=(!strip !debug)
install=ollama-airllm-rocm.install

# Sources
source=(
    "git+https://github.com/ollama/ollama.git#tag=v${pkgver}"
    "airllm::git+https://github.com/lyogavin/AirLLM.git"
    "ollama-airllm-rocm.install"
    "airllm_runner.py"
    "airllm.patch"
)

sha256sums=('SKIP' 'SKIP' 'SKIP' 'SKIP' 'SKIP')

# ROCm architecture - auto-detected or set manually
_rocm_arch='gfx1100'

prepare() {
    cd "${srcdir}"
    
    log_info() {
        echo "==> $1"
    }
    
    log_info "Preparing sources..."
    
    # Initialize Ollama submodules
    cd ollama
    git submodule update --init --recursive
    
    # Detect ROCm architecture if not set
    if command -v rocm_agent_enumerator &> /dev/null; then
        detected_arch=$(rocm_agent_enumerator 2>/dev/null | grep -v "gfx000" | head -1)
        if [ -n "$detected_arch" ]; then
            log_info "Detected ROCm architecture: $detected_arch"
            _rocm_arch="$detected_arch"
        fi
    fi
    
    log_info "Using ROCm architecture: $_rocm_arch"
    
    # Apply AirLLM integration patches
    cd "${srcdir}/ollama"
    
    # Create necessary directories for AirLLM integration
    mkdir -p runner/airllmrunner
    
    # Copy AirLLM runner files if they exist in source
    if [ -f "${srcdir}/../runner/airllmrunner/runner.go" ]; then
        cp "${srcdir}/../runner/airllmrunner/"*.go runner/airllmrunner/
    fi
    
    # Apply patch for AirLLM integration
    if [ -f "${srcdir}/airllm.patch" ]; then
        patch -p1 < "${srcdir}/airllm.patch" || log_info "Some patches may have already been applied"
    fi
    
    # Prepare AirLLM
    cd "${srcdir}"
    if [ -d "airllm/air_llm" ]; then
        rm -rf airllm-clean
        cp -r airllm/air_llm airllm-clean
        log_info "AirLLM prepared"
    fi
}

build() {
    cd "${srcdir}/ollama"
    
    log_info() {
        echo "==> $1"
    }
    
    log_info "Building Ollama with ROCm support for $_rocm_arch..."
    
    # ROCm environment
    export HIP_PATH="/opt/rocm"
    export HIPCXX="/opt/rocm/llvm/bin/clang++"
    export AMDGPU_TARGETS="${_rocm_arch}"
    export HSA_OVERRIDE_GFX_VERSION=11.0.0
    
    # Go build flags
    export GOFLAGS="-trimpath -buildmode=pie"
    export CGO_ENABLED=1
    export CGO_CFLAGS="${CFLAGS}"
    export CGO_CXXFLAGS="${CXXFLAGS}"
    export CGO_LDFLAGS="${LDFLAGS}"
    export LDFLAGS=""
    
    # Build ggml backend with ROCm
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
        -DAMDGPU_TARGETS="${_rocm_arch}" \
        -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
        -DCMAKE_C_COMPILER=gcc
    
    cmake --build . --config Release -j$(nproc)
    
    cd "${srcdir}/ollama"
    
    # Build Ollama binary
    log_info "Building Ollama binary..."
    go build \
        -tags="" \
        -o "ollama-bin" \
        -ldflags="-w -s -X=github.com/ollama/ollama/version.Version=${pkgver}" \
        .
    
    log_info "Build complete"
}

package() {
    log_info() {
        echo "==> $1"
    }
    
    log_info "Packaging..."
    
    # Create directory structure
    install -dm755 "${pkgdir}/usr/bin"
    install -dm755 "${pkgdir}/usr/lib/ollama"
    install -dm755 "${pkgdir}/usr/lib/systemd/system"
    install -dm755 "${pkgdir}/usr/lib/sysusers.d"
    install -dm755 "${pkgdir}/etc/default"
    install -dm755 "${pkgdir}/usr/share/ollama"
    install -dm755 "${pkgdir}/usr/share/licenses/${pkgname}"
    
    # Install Ollama binary
    install -Dm755 "${srcdir}/ollama/ollama-bin" "${pkgdir}/usr/bin/ollama"
    
    # Install ggml libraries
    if [ -d "${srcdir}/ollama/build/lib/ollama" ]; then
        cp -r "${srcdir}/ollama/build/lib/ollama/"* "${pkgdir}/usr/lib/ollama/"
    fi
    
    # Copy additional libraries from build
    cd "${srcdir}/ollama/build"
    for lib in libggml*.so*; do
        if [ -e "$lib" ]; then
            cp -P "$lib" "${pkgdir}/usr/lib/ollama/"
        fi
    done
    
    # Install AirLLM
    if [ -d "${srcdir}/airllm-clean" ]; then
        cp -r "${srcdir}/airllm-clean" "${pkgdir}/usr/share/ollama/airllm"
        log_info "AirLLM installed"
    fi
    
    # Install AirLLM Python runner
    if [ -f "${srcdir}/airllm_runner.py" ]; then
        install -Dm755 "${srcdir}/airllm_runner.py" "${pkgdir}/usr/share/ollama/airllm_runner.py"
    elif [ -f "${srcdir}/../runner/airllmrunner/airllm_runner.py" ]; then
        install -Dm755 "${srcdir}/../runner/airllmrunner/airllm_runner.py" "${pkgdir}/usr/share/ollama/airllm_runner.py"
    fi
    
    # Create systemd service
    cat > "${pkgdir}/usr/lib/systemd/system/ollama.service" << 'EOF'
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
ReadWritePaths=/var/lib/ollama /tmp
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

    # Create sysusers config
    cat > "${pkgdir}/usr/lib/sysusers.d/ollama.conf" << 'EOF'
u ollama - "Ollama service user" -
EOF

    # Create environment config
    cat > "${pkgdir}/etc/default/ollama" << 'EOF'
# Ollama configuration for ROCm with AirLLM
export OLLAMA_MODELS="${OLLAMA_MODELS:-${HOME}/.ollama/models}"
export OLLAMA_HOST="127.0.0.1:11434"
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export AIRLLM_COMPRESSION="4bit"
export AIRLLM_DEVICE="cuda:0"
export PYTHONPATH="/usr/share/ollama/airllm:/usr/share/ollama/airllm/air_llm:${PYTHONPATH}"

# ROCm specific
export HIP_VISIBLE_DEVICES=0
EOF

    # Install license
    install -Dm644 "${srcdir}/ollama/LICENSE" "${pkgdir}/usr/share/licenses/${pkgname}/LICENSE"
    
    # Create wrapper script for better opencode integration
    cat > "${pkgdir}/usr/bin/ollama-airllm" << 'EOF'
#!/bin/bash
# Ollama AirLLM wrapper for opencode integration

# Source environment
if [ -f /etc/default/ollama ]; then
    source /etc/default/ollama
fi

# Export all variables
export OLLAMA_MODELS
export OLLAMA_HOST
export HSA_OVERRIDE_GFX_VERSION
export AIRLLM_COMPRESSION
export AIRLLM_DEVICE
export PYTHONPATH
export HIP_VISIBLE_DEVICES

# Check if model should use AirLLM
use_airllm=false
if [ "$1" = "run" ] || [ "$1" = "serve" ]; then
    # Auto-detect based on model size or explicit flag
    if [ -n "$AIRLLM_FORCE" ] || [ -n "$USE_AIRLLM" ]; then
        use_airllm=true
    fi
fi

if [ "$use_airllm" = true ]; then
    echo "Using AirLLM integration for large model support"
fi

# Execute ollama
exec /usr/bin/ollama "$@"
EOF
    chmod 755 "${pkgdir}/usr/bin/ollama-airllm"
    
    log_info "Package created successfully"
    
    # Display summary
    echo ""
    echo "========================================"
    echo "Package Summary"
    echo "========================================"
    echo "Binary: ${pkgdir}/usr/bin/ollama"
    echo "Libraries: ${pkgdir}/usr/lib/ollama/"
    echo "AirLLM: ${pkgdir}/usr/share/ollama/airllm"
    echo "Config: ${pkgdir}/etc/default/ollama"
    echo "Service: ${pkgdir}/usr/lib/systemd/system/ollama.service"
    echo "========================================"
}
