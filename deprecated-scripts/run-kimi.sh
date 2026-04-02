#!/bin/bash
# Run Kimi K2.5 with AirLLM layer offloading

# Set environment to use AirLLM runner
export OLLAMA_USE_AIRLLM=1
export AIRLLM_DEVICE=cuda:0
export AIRLLM_COMPRESSION=4bit

# Model path
MODEL_PATH="/sda2/airllm/kimi-k2.5"

# Check if prompt provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 'Your prompt here'"
    echo ""
    echo "Example:"
    echo "  $0 'Explain quantum computing'"
    exit 1
fi

PROMPT="$1"

echo "Running Kimi K2.5 with AirLLM..."
echo "Model: $MODEL_PATH"
echo "Prompt: $PROMPT"
echo ""

# Run with ollama using the AirLLM runner
ollama run "$MODEL_PATH" "$PROMPT"
