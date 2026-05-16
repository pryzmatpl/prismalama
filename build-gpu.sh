#!/bin/bash
# GPU-Optimized Prismalama Build Script
# Build with NVIDIA GPU support for maximum inference performance

set -e

echo "🚀 Building GPU-optimized prismalama..."

# Check for NVIDIA GPU
echo "🔍 Checking for NVIDIA GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: nvidia-smi not found. GPU drivers not installed."
    echo "   Please install NVIDIA drivers: apt-get install nvidia-driver-535"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
echo ""

# Check for Docker with NVIDIA support
echo "🔍 Checking Docker NVIDIA support..."
if ! docker info 2>/dev/null | grep -q "nvidia"; then
    echo "⚠️  WARNING: NVIDIA Container Toolkit not detected."
    echo "   Installing NVIDIA Container Toolkit..."
    
    # Install NVIDIA Container Toolkit
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
fi

# Build the Docker image
echo "🏗️  Building GPU-optimized prismalama image..."
cd /home/prizm/prismalama

docker build -f Dockerfile.gpu \
    --build-arg BASE_IMAGE=nvidia/cuda:12.2.0-devel-ubuntu22.04 \
    -t prismalama:latest-gpu \
    .

echo "✅ GPU build complete!"
echo ""
echo "📊 To run with GPU support:"
echo "   docker run -d --gpus=all -p 11434:11434 -v ollama:/root/.ollama --name prismalama-gpu prismalama:latest-gpu"
echo ""
echo "🚀 Or use docker-compose:"
echo "   docker compose -f docker-compose.gpu.yml up -d"
echo ""
echo "⚡ After deployment, pull and run models:"
echo "   docker exec prismalama-gpu ollama pull qwen3.6:27b"
echo "   docker exec prismalama-gpu ollama run qwen3.6:27b"
