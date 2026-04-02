#!/bin/bash
# Install llama-cpp-python with ROCm support for direct Kimi inference

echo "=== Installing llama-cpp-python with ROCm ==="
echo ""

# Install with ROCm support
export CMAKE_ARGS="-DGGML_HIPBLAS=on -DAMDGPU_TARGETS=gfx1100"
export HIP_PATH="/opt/rocm"

echo "Building llama-cpp-python with ROCm (this may take 5-10 minutes)..."
pip3 install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

echo ""
echo "Installation complete!"
echo ""
echo "Test with: python3 /sda2/kimi-direct.py 'Hello, how are you?'"
