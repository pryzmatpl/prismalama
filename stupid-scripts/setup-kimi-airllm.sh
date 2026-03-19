#!/bin/bash
# Setup Kimi K2.5 for AirLLM inference without copying the model

echo "=== Kimi K2.5 AirLLM Setup ==="
echo ""

# Create symlink to model
if [ ! -L "/sda2/airllm/kimi-k2.5" ]; then
    echo "Creating symlink to Kimi model..."
    ln -s "/nvme3/AI Models/Kimi" /sda2/airllm/kimi-k2.5
fi

# Create a marker file to indicate AirLLM should be used
echo "Creating AirLLM marker..."
touch /sda2/airllm/kimi-k2.5/.use_airllm

# Update Ollama environment to use AirLLM for this model
echo "Setting up Ollama environment..."

# Check if we need to update the systemd service
if ! grep -q "OLLAMA_USE_AIRLLM" /etc/default/ollama; then
    echo "" >> /etc/default/ollama
    echo "# AirLLM configuration for large models" >> /etc/default/ollama
    echo "OLLAMA_USE_AIRLLM=1" >> /etc/default/ollama
fi

echo ""
echo "Setup complete!"
echo ""
echo "To use Kimi with AirLLM:"
echo "  1. Set environment variable: export OLLAMA_USE_AIRLLM=1"
echo "  2. Run: ollama run /sda2/airllm/kimi-k2.5 'Your prompt'"
echo ""
echo "Or use the direct AirLLM runner:"
echo "  python3 /sda2/airllm/kimi_runner.py --prompt 'Your prompt'"
