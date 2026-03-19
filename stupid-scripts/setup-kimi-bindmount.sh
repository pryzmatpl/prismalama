#!/bin/bash
# Alternative approach: Use bind mount to make Kimi accessible to Ollama
# This bypasses symlink restrictions by using Linux mount namespaces

echo "=== Kimi Bind Mount Setup for Ollama ==="
echo ""

if [ "$EUID" -ne 0 ]; then 
    echo "Please run with sudo"
    exit 1
fi

# Create mount point in Ollama's allowed path
mkdir -p /var/lib/ollama/models/kimi-k2.5

# Bind mount the Kimi directory
if mountpoint -q /var/lib/ollama/models/kimi-k2.5; then
    echo "✓ Bind mount already exists"
else
    echo "Creating bind mount..."
    mount --bind "/nvme3/AI Models/Kimi" /var/lib/ollama/models/kimi-k2.5
    echo "✓ Bind mount created"
fi

# Create Modelfile pointing to bind mount
cat > /tmp/Modelfile.kimi-bind << 'EOF'
FROM /var/lib/ollama/models/kimi-k2.5/Kimi-K2.5-Q4_K_M-00001-of-00013.gguf

PARAMETER num_ctx 8192
PARAMETER num_batch 1
PARAMETER temperature 0.6
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"

SYSTEM """You are Kimi, a large language model developed by Moonshot AI."""

TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
EOF

echo ""
echo "Creating Ollama model from bind mount..."
ollama create kimi-k2.5 -f /tmp/Modelfile.kimi-bind

echo ""
echo "✓ Setup complete!"
echo ""
echo "To make this permanent, add to /etc/fstab:"
echo '  "/nvme3/AI Models/Kimi" /var/lib/ollama/models/kimi-k2.5 none bind 0 0'
echo ""
echo "Test with:"
echo "  ollama run kimi-k2.5 'Hello!'"
