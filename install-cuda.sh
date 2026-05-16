#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_cuda() {
    log_info "Checking CUDA prerequisites..."
    
    if command -v nvcc &> /dev/null; then
        log_info "CUDA toolkit found: $(nvcc --version | grep release | awk '{print $5}' | tr -d ',')"
        return 0
    fi
    
    if lspci | grep -i nvidia | grep -qv "GK208"; then
        log_info "NVIDIA GPU detected (not GT 710)"
        if command -v nvidia-smi &> /dev/null; then
            log_warn "NVIDIA driver installed but CUDA toolkit missing"
        else
            log_warn "NVIDIA driver not loaded"
        fi
        log_info "Installing CUDA toolkit..."
        install_cuda
    elif lspci | grep -i nvidia | grep -q "GK208"; then
        log_error "Only GT 710 detected (not supported for CUDA)"
        exit 1
    else
        log_error "No NVIDIA GPU detected"
        exit 1
    fi
}

install_cuda() {
    log_info "Installing CUDA toolkit..."
    
    if [ -f /etc/arch-release ]; then
        sudo pacman -S --noconfirm cuda
    elif [ -f /etc/debian_version ]; then
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        sudo apt-get update
        sudo apt-get install -y cuda
    else
        log_error "Unsupported distro. Install CUDA manually from https://developer.nvidia.com/cuda-downloads"
        exit 1
    fi
    
    if [ -f /etc/profile.d/cuda.sh ]; then
        bash /etc/profile.d/cuda.sh 2>/dev/null || true
    fi
    export PATH="/opt/cuda/bin:/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/opt/cuda/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
}

build_cuda() {
    log_info "Building prismalama-ollama (CUDA)..."
    
    rm -rf build
    
    local cuda_available=0
    
    if command -v nvcc &> /dev/null; then
        log_info "Building with CUDA support..."
        cmake -B build -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX=/usr \
            -DLLAMA_CUDA=ON \
            -DLLAMA_HIPBLAS=OFF \
            -DOLLAMA_RUNNER_DIR=cuda
        
        cmake --build build --parallel "$(nproc)" --target ggml ggml-cuda
        cuda_available=1
    else
        log_warn "CUDA toolkit not found - building CPU-only binary"
        cmake -B build -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX=/usr \
            -DLLAMA_CUDA=OFF \
            -DLLAMA_HIPBLAS=OFF \
            -DOLLAMA_VULKAN=OFF \
            -DOLLAMA_RUNNER_DIR=cuda
        
        cmake --build build --parallel "$(nproc)" --target ggml
    fi
    
    export CGO_ENABLED=1
    export GOFLAGS="-trimpath -buildmode=pie"
    
    PKG_VERSION=$(awk -F= '/^pkgver=/{print $2; exit}' PKGBUILD)
    
    if [ $cuda_available -eq 1 ]; then
        go build -o prismalama-ollama \
            -ldflags "-w -s -X=github.com/ollama/ollama/version.Version=${PKG_VERSION}-cuda" \
            .
    else
        go build -o prismalama-ollama \
            -ldflags "-w -s -X=github.com/ollama/ollama/version.Version=${PKG_VERSION}-cpu" \
            .
        log_warn "CPU-only build - install cuda for GPU acceleration"
    fi
    
    log_info "Build complete: prismalama-ollama"
}

install_local() {
    log_info "Installing prismalama-ollama..."
    
    sudo install -Dm755 prismalama-ollama /usr/bin/ollama
    
    if [ -d build/lib/ollama/cuda ]; then
        sudo mkdir -p /usr/lib/ollama/cuda
        sudo cp -r build/lib/ollama/cuda/* /usr/lib/ollama/cuda/
    fi
    
    sudo install -Dm755 runner/airllmrunner/airllm_runner.py /usr/share/ollama/airllm_runner.py
    
    if [ -d src/airllm/air_llm ]; then
        sudo mkdir -p /usr/share/ollama/airllm
        sudo cp -r src/airllm/air_llm /usr/share/ollama/airllm/
    fi
    
    sudo mkdir -p /usr/lib/sysusers.d
    printf 'u ollama - "Ollama service user" -\n' | sudo tee /usr/lib/sysusers.d/ollama.conf > /dev/null
    
    sudo mkdir -p /usr/lib/systemd/system
    sudo tee /usr/lib/systemd/system/ollama.service > /dev/null << 'EOF'
[Unit]
Description=Prismalama (Ollama-compatible) — CUDA
After=network.target

[Service]
Type=simple
User=ollama
ExecStart=/usr/bin/ollama serve
Restart=on-failure
NoNewPrivileges=true
ProtectSystem=strict

[Install]
WantedBy=multi-user.target
EOF

    sudo mkdir -p /etc/default
    sudo tee /etc/default/ollama > /dev/null << 'EOF'
OLLAMA_HOST=127.0.0.1:11434
OLLAMA_NUM_PARALLEL=1
OLLAMA_LIBRARY_PATH=/usr/lib/ollama/cuda
OLLAMA_KEEP_ALIVE=5m
OLLAMA_USE_AIRLLM=0
CUDA_VISIBLE_DEVICES=0
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable ollama 2>/dev/null || true
    
    log_info "Installation complete!"
    echo ""
    echo "To start: sudo systemctl start ollama"
    echo "To use:   ollama serve &"
}

main() {
    check_cuda
    build_cuda
    install_local
}

main "$@"
