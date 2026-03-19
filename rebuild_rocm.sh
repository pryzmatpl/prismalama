#!/bin/bash

cd /sda2/prismalama/build_ollama_airllm/build

# Set ROCM paths
export PKG_CONFIG_PATH="/opt/rocm/lib/pkgconfig:$PKG_CONFIG_PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"
export ROCM_PATH="/opt/rocm"

# Reconfigure CMake with ROCM
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DLLAMA_CURL=ON \
    -DLLAMA_HIPBLAS=ON \
    -DLLAMA_CUDA=OFF \
    -DCMAKE_HIP_COMPILER_ROCM_ROOT="/opt/rocm" \
    -DAMDGPU_TARGETS="gfx1100" \
    -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DCMAKE_C_COMPILER=gcc

# Rebuild
make -j$(nproc)

# Copy updated binary
cp ollama /usr/bin/ollama
sudo chown root:root /usr/bin/ollama
sudo chmod 755 /usr/bin/ollama

echo "Rebuilt with ROCM support. Restart Ollama." 