#!/bin/bash
# SIMPLE KIMI SETUP - Just let Ollama do its job
# The 579GB will be copied to /nvme3 (you have 1.2TB free)

echo "======================================================================"
echo "  KIMI SETUP - Simple & Working"
echo "======================================================================"
echo ""
echo "What will happen:"
echo "  1. Ollama will COPY 579GB from /nvme3/AI Models/Kimi/"
echo "  2. Ollama will properly index all 13 shards"
echo "  3. Model will be available via API"
echo ""
echo "Time required: ~15-30 minutes for copying"
echo "Space used: 579GB on /nvme3 (you have 1.2TB free)"
echo ""

# Export for this session
export OLLAMA_MODELS=/nvme3/ollama-models

# Create the Modelfile
cat > /tmp/Modelfile.kimi << 'EOF'
FROM /nvme3/AI Models/Kimi/Kimi-K2.5-Q4_K_M-00001-of-00013.gguf

PARAMETER num_ctx 4096
PARAMETER num_gpu 999
PARAMETER temperature 0.6
PARAMETER top_p 0.95
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"

SYSTEM """You are Kimi, a large language model developed by Moonshot AI."""

TEMPLATE """<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
EOF

echo "Step 1: Creating Kimi model..."
echo "  (This will take 15-30 minutes to copy 579GB)"
echo "  Starting at: $(date)"
echo ""

ollama create kimi -f /tmp/Modelfile.kimi 2>&1

echo ""
echo "Step 2: Verifying..."
ollama list | grep kimi

echo ""
echo "Step 3: Testing..."
if ollama list | grep -q kimi; then
    echo "  ✓ Kimi created successfully!"
    echo ""
    echo "Testing inference:"
    ollama run kimi "Hello! You are Kimi on potato hardware. Confirm you're working."
else
    echo "  ✗ Something went wrong"
    exit 1
fi

echo ""
echo "======================================================================"
echo "  SETUP COMPLETE"
echo "======================================================================"
echo ""
echo "Add to your ~/.bashrc for future sessions:"
echo '  export OLLAMA_MODELS=/nvme3/ollama-models'
echo ""
echo "Usage:"
echo "  ollama run kimi 'Your prompt here'"
echo "  opencode run -m ollama/kimi"
echo ""
echo "Space used:"
du -sh /nvme3/ollama-models/* 2>/dev/null | grep -i kimi || echo "  (Check with: du -sh /nvme3/ollama-models/)"
echo ""
echo "======================================================================"
