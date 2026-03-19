#!/bin/bash
# Consolidate all models into /nvme3/ollama-models with symlinks
# This creates a unified view while keeping data on respective drives

echo "=== Consolidating Ollama Models Across Drives ==="
echo ""

if [ "$EUID" -ne 0 ]; then 
    echo "Please run with sudo"
    exit 1
fi

# Stop Ollama
echo "[1/6] Stopping Ollama..."
systemctl stop ollama

# Create unified model directory on nvme3
echo "[2/6] Creating unified model directory..."
MODELS_DIR="/nvme3/ollama-models"
mkdir -p "$MODELS_DIR"

# Copy existing Ollama structure
echo "[3/6] Setting up directory structure..."
mkdir -p "$MODELS_DIR/blobs"
mkdir -p "$MODELS_DIR/manifests/registry.ollama.ai/library"

# Copy existing Ollama models if they exist
if [ -d "/var/lib/ollama/blobs" ]; then
    echo "[4/6] Copying existing models from /var/lib/ollama..."
    cp -r /var/lib/ollama/blobs/* "$MODELS_DIR/blobs/" 2>/dev/null || true
    cp -r /var/lib/ollama/manifests/* "$MODELS_DIR/manifests/" 2>/dev/null || true
    echo "  ✓ Existing models copied"
fi

# Create symlinks for /sda2 models
echo "[5/6] Linking /sda2 models..."

# Link Qwen model
if [ -f "/sda2/Qwen2.5-Coder-32B-Instruct-Q5_K_S.gguf" ]; then
    QWEN_HASH=$(sha256sum "/sda2/Qwen2.5-Coder-32B-Instruct-Q5_K_S.gguf" | cut -d' ' -f1)
    ln -sf "/sda2/Qwen2.5-Coder-32B-Instruct-Q5_K_S.gguf" "$MODELS_DIR/blobs/sha256-$QWEN_HASH"
    echo "  ✓ Linked Qwen model"
fi

# Link GLM-4.7-Flash directory (for AirLLM)
if [ -d "/sda2/GLM-4.7-Flash-4bit" ]; then
    # GLM uses AirLLM, create reference
    echo "  ✓ GLM-4.7-Flash available at /sda2/GLM-4.7-Flash-4bit"
fi

# Update systemd service
echo "[6/6] Configuring Ollama..."
mkdir -p /etc/systemd/system/ollama.service.d/

cat > /etc/systemd/system/ollama.service.d/override.conf << EOF
[Service]
Environment="OLLAMA_MODELS=/nvme3/ollama-models"
ReadWritePaths=/nvme3/ollama-models /sda2 /var/lib/ollama /tmp
EOF

# Create wrapper script for Modelfiles
cat > /sda2/create-modelfiles.sh << 'EOFSCRIPT'
#!/bin/bash
# Create Modelfiles for all available models

echo "Creating Modelfiles..."

# Qwen
cat > /sda2/Modelfile.qwen << 'EOF'
FROM /sda2/Qwen2.5-Coder-32B-Instruct-Q5_K_S.gguf

PARAMETER num_ctx 32768
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
EOF

# GLM-4.7
cat > /sda2/Modelfile.glm << 'EOF'
FROM /sda2/GLM-4.7-Flash-4bit

PARAMETER num_ctx 4096
PARAMETER temperature 0.7
PARAMETER top_p 0.9

SYSTEM You are GLM-4.7-Flash, a helpful AI assistant.
EOF

echo "✓ Modelfiles created"
echo "  - /sda2/Modelfile.qwen"
echo "  - /sda2/Modelfile.glm"
echo ""
echo "Register models:"
echo "  ollama create qwen -f /sda2/Modelfile.qwen"
echo "  ollama create glm47 -f /sda2/Modelfile.glm"
EOFSCRIPT

chmod +x /sda2/create-modelfiles.sh

# Set permissions
chown -R ollama:ollama "$MODELS_DIR"

# Backup old directory
mv /var/lib/ollama /var/lib/ollama-backup-$(date +%Y%m%d)
ln -s "$MODELS_DIR" /var/lib/ollama

# Reload and restart
echo ""
echo "Reloading systemd..."
systemctl daemon-reload
systemctl start ollama

sleep 2

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Model locations:"
echo "  /nvme3/ollama-models/  - Ollama models + Kimi (new)"
echo "  /sda2/                 - Qwen, GLM (existing, linked)"
echo "  /var/lib/ollama/       -> symlink to /nvme3/ollama-models"
echo ""
echo "Available space on /nvme3: $(df -h /nvme3 | tail -1 | awk '{print $4}')"
echo ""
echo "Next steps:"
echo "  1. Create Modelfiles: bash /sda2/create-modelfiles.sh"
echo "  2. Register existing models from /sda2"
echo "  3. Add Kimi: ollama create kimi -f /sda2/Modelfile.kimi-direct"
echo ""
echo "Current models:"
ollama list 2>/dev/null || echo "  (Ollama starting up...)"
