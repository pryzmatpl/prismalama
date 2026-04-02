#!/bin/bash
# Test Kimi on Potato Hardware
# Simple and direct

export OLLAMA_MODELS=/nvme3/ollama-models
export OLLAMA_HOST=127.0.0.1:11434

echo "Testing Kimi on Potato Hardware..."
echo ""
echo "Available models:"
ollama list

echo ""
echo "Running Kimi..."
ollama run kimi-potato "Hello! You are running on potato hardware. How do you feel?"
