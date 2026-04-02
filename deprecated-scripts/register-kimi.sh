#!/bin/bash
# Register Kimi K2.5 with Ollama AirLLM ROCm

echo "=== Registering Kimi K2.5 with Ollama ==="
echo ""

# Check if Ollama is running
if ! systemctl is-active --quiet ollama; then
    echo "Starting Ollama service..."
    sudo systemctl start ollama
    sleep 2
fi

# Register standard Kimi model
echo "[1/2] Creating kimi-k2.5 model..."
ollama create kimi-k2.5 -f /sda2/Modelfile.kimi-k2.5

# Register AirLLM-optimized version
echo "[2/2] Creating kimi-k2.5-airllm model..."
ollama create kimi-k2.5-airllm -f /sda2/Modelfile.kimi-airllm

echo ""
echo "=== Kimi Models Registered ==="
echo ""
echo "Available models:"
ollama list | grep kimi

echo ""
echo "Usage examples:"
echo "  ollama run kimi-k2.5 'Hello, who are you?'"
echo "  ollama run kimi-k2.5-airllm 'Explain quantum computing'"
echo ""
echo "For AirLLM inference (recommended for 579GB model):"
echo "  ollama run kimi-k2.5-airllm"
echo ""
