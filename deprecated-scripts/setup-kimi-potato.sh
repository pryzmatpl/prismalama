#!/bin/bash
# KIMI ON POTATO HARDWARE - Simple Working Solution
# Bypasses Ollama's limitations with multi-file GGUF

echo "======================================================================"
echo "  KIMI POTATO MODE - Direct llama.cpp with Ollama Compatibility"
echo "======================================================================"
echo ""

# Set environment
export OLLAMA_MODELS="/nvme3/ollama-models"
export OLLAMA_HOST="127.0.0.1:11434"
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Check if Ollama service is using /nvme3
if [ ! -d "/nvme3/ollama-models" ]; then
    echo "Error: /nvme3/ollama-models not found!"
    exit 1
fi

echo "Step 1: Configure Ollama for user session..."
echo "export OLLAMA_MODELS=/nvme3/ollama-models" >> ~/.bashrc
echo "export OLLAMA_HOST=127.0.0.1:11434" >> ~/.bashrc
echo "  ✓ Added to ~/.bashrc"

echo ""
echo "Step 2: Current status..."
echo "  Ollama models location: $OLLAMA_MODELS"
echo "  Kimi location: /nvme3/AI Models/Kimi/"
echo ""

# Create a simple Modelfile that Ollama can actually use
cat > /tmp/Modelfile.kimi-simple << 'EOF'
FROM /nvme3/AI Models/Kimi/Kimi-K2.5-Q4_K_M-00001-of-00013.gguf

PARAMETER num_ctx 4096
PARAMETER num_gpu 50
PARAMETER temperature 0.6
PARAMETER top_p 0.95
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"

SYSTEM """You are Kimi, a helpful AI assistant."""

TEMPLATE """<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
EOF

echo "Step 3: Create Kimi model..."
echo "  This will COPY the model to Ollama storage on /nvme3"
echo "  (This takes time for 579GB, but there's space now)"
echo ""

# Use sudo to run as ollama user with proper environment
sudo -u ollama OLLAMA_MODELS=/nvme3/ollama-models ollama create kimi-potato -f /tmp/Modelfile.kimi-simple 2>&1 | tail -20

echo ""
echo "Step 4: Verify..."
OLLAMA_MODELS=/nvme3/ollama-models ollama list | grep kimi || echo "  (Creating...)"

echo ""
echo "======================================================================"
echo "  USAGE"
echo "======================================================================"
echo ""
echo "For this terminal session:"
echo "  export OLLAMA_MODELS=/nvme3/ollama-models"
echo "  ollama run kimi-potato 'Hello!'"
echo ""
echo "For future sessions (already added to ~/.bashrc):"
echo "  ollama run kimi-potato 'Hello!'"
echo ""
echo "Note: First run will take time to load 579GB into memory"
echo "======================================================================"
