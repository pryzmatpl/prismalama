#!/bin/bash
# Build script for CUDA (NVIDIA RTX 3080 etc)
# Falls back to CPU+Vulkan if CUDA toolkit not found

set -e

BUILD_DIR="build_cuda"
PKG_VERSION="0.4.1"
PKG_REL="1"
PKG_NAME="prismalama-ollama-cuda"

if [ -f PKGBUILD ]; then
    PKG_VERSION=$(awk -F= '/^pkgver=/{print $2; exit}' PKGBUILD)
    PKG_REL=$(awk -F= '/^pkgrel=/{print $2; exit}' PKGBUILD)
fi

echo "Building ${PKG_NAME}..."

rm -rf build

# Check for CUDA compiler
if command -v nvcc &> /dev/null; then
    echo "CUDA toolkit found, building with CUDA..."
    cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -DLLAMA_CUDA=ON \
        -DLLAMA_HIPBLAS=OFF \
        -DOLLAMA_RUNNER_DIR=cuda

    cmake --build build --parallel "$(nproc)" --target ggml ggml-cuda
else
    echo "CUDA toolkit not found (nvcc missing)."
    echo "Install cuda toolkit or use make build-rocm for AMD GPUs."
    echo "Falling back to CPU+Vulkan build..."
    
    cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -DLLAMA_CUDA=OFF \
        -DLLAMA_HIPBLAS=OFF \
        -DOLLAMA_RUNNER_DIR=cuda

    cmake --build build --parallel "$(nproc)" --target ggml ggml-vulkan
fi

# Build ollama binary
export CGO_ENABLED=1
export GOFLAGS="-trimpath -buildmode=pie"
export LDFLAGS="-w -s -X=github.com/ollama/ollama/version.Version=${PKG_VERSION}-cuda"
go build -o prismalama-ollama \
    -ldflags "-w -s -X=github.com/ollama/ollama/version.Version=${PKG_VERSION}-cuda" \
    .

echo ""
echo "Binary built: prismalama-ollama"
echo ""
echo "To test:"
echo "  export OLLAMA_MODELS=/tmp/ollama_models"
echo "  export OLLAMA_LIBRARY_PATH=\$(pwd)/build/lib/ollama/cuda"
echo "  ./prismalama-ollama serve"
