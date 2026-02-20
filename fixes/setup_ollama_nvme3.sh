#!/bin/bash
# Fix Ollama to use models from /nvme3
# This script sets up the environment and fixes manifest paths

set -e

# Set the environment variable permanently
echo "Setting OLLAMA_MODELS=/nvme3/ollama-models"

# Add to shell profile
SHELL_RC=""
if [ -f "$HOME/.bashrc" ]; then
    SHELL_RC="$HOME/.bashrc"
elif [ -f "$HOME/.zshrc" ]; then
    SHELL_RC="$HOME/.zshrc"
fi

if [ -n "$SHELL_RC" ]; then
    if ! grep -q "OLLAMA_MODELS=/nvme3/ollama-models" "$SHELL_RC" 2>/dev/null; then
        echo 'export OLLAMA_MODELS=/nvme3/ollama-models' >> "$SHELL_RC"
        echo "Added to $SHELL_RC"
    fi
fi

# Export for current session
export OLLAMA_MODELS=/nvme3/ollama-models

echo "OLLAMA_MODELS is now: $OLLAMA_MODELS"
echo ""
echo "Available models:"
ls -la "$OLLAMA_MODELS/manifests/registry.ollama.ai/library/" 2>/dev/null || echo "No models found"
echo ""

# Fix the kimi manifest to point to correct paths
KIMI_MANIFEST="$OLLAMA_MODELS/manifests/registry.ollama.ai/library/kimi/latest"
if [ -f "$KIMI_MANIFEST" ]; then
    if grep -q "/var/lib/ollama/blobs" "$KIMI_MANIFEST"; then
        echo "Fixing kimi manifest paths..."
        # The kimi model should use the same paths as kimi-k2.5
        # Copy the kimi-k2.5 manifest as the base and modify
        cp "$OLLAMA_MODELS/manifests/registry.ollama.ai/library/kimi-k2.5/latest" "$KIMI_MANIFEST"
        echo "Fixed kimi manifest"
    fi
fi

echo ""
echo "To use with opencode, run:"
echo "  export OLLAMA_MODELS=/nvme3/ollama-models"
echo "  opencode --model kimi-k2.5"
