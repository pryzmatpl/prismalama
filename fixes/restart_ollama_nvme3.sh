#!/bin/bash
# Fix Ollama tonvme3
 use models from /# Run this script to fix the model path issue

echo "=== Fixing Ollama Model Path ==="
echo ""

# Stop the running Ollama server
echo "Stopping Ollama server..."
pkill -f "ollama serve" 2>/dev/null || true
sleep 2

# Set environment
export OLLAMA_MODELS=/nvme3/ollama-models

echo "OLLAMA_MODELS set to: $OLLAMA_MODELS"
echo ""
echo "Available models:"
ls -la "$OLLAMA_MODELS/manifests/registry.ollama.ai/library/" 2>/dev/null
echo ""

# Start Ollama in background with correct env
echo "Starting Ollama server..."
nohup env OLLAMA_MODELS=/nvme3/ollama-models ollama serve > /tmp/ollama.log 2>&1 &
sleep 3

# Verify it's running with correct env
echo ""
echo "Verifying Ollama is running..."
ps aux | grep "ollama serve" | grep -v grep

echo ""
echo "To test, run:"
echo "  export OLLAMA_MODELS=/nvme3/ollama-models"
echo "  ollama list"
