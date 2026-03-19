#!/bin/bash
# Move all models to /nvme3 - simple and clean

echo "=== Moving All Models to /nvme3 ==="
echo ""

if [ "$EUID" -ne 0 ]; then 
    echo "Please run with sudo"
    exit 1
fi

# Stop Ollama
echo "[1/5] Stopping Ollama..."
systemctl stop ollama

# Create models directory on nvme3
echo "[2/5] Creating /nvme3/ollama-models..."
mkdir -p /nvme3/ollama-models

# Move existing Ollama models
echo "[3/5] Moving models from /var/lib/ollama..."
if [ -d "/var/lib/ollama" ]; then
    mv /var/lib/ollama/* /nvme3/ollama-models/ 2>/dev/null || true
    rm -rf /var/lib/ollama
fi

# Move models from /sda2 to /nvme3
echo "[4/5] Moving models from /sda2..."
mkdir -p /nvme3/models

# Move Qwen GGUF
if [ -f "/sda2/Qwen2.5-Coder-32B-Instruct-Q5_K_S.gguf" ]; then
    mv "/sda2/Qwen2.5-Coder-32B-Instruct-Q5_K_S.gguf" /nvme3/models/
    echo "  ✓ Qwen GGUF moved"
fi

# Move Qwen full directory (62GB safetensors)
if [ -d "/sda2/Qwen2.5-Coder-32B-Instruct" ]; then
    echo "  Moving Qwen directory (62GB, this may take a few minutes)..."
    mv "/sda2/Qwen2.5-Coder-32B-Instruct" /nvme3/models/
    echo "  ✓ Qwen directory moved"
fi

# Move GLM
if [ -d "/sda2/GLM-4.7-Flash-4bit" ]; then
    mv "/sda2/GLM-4.7-Flash-4bit" /nvme3/models/
    echo "  ✓ GLM-4.7 moved"
fi

if [ -d "/sda2/GLM-4.7-Flash" ]; then
    mv "/sda2/GLM-4.7-Flash" /nvme3/models/ 2>/dev/null || true
fi

# Move other models
for model in /sda2/*.gguf; do
    if [ -f "$model" ] && [ -s "$model" ]; then
        mv "$model" /nvme3/models/
        echo "  ✓ $(basename $model) moved"
    fi
done

# Move Step3-VL
if [ -d "/sda2/Step3-VL-10B-Base" ]; then
    mv "/sda2/Step3-VL-10B-Base" /nvme3/models/ 2>/dev/null || true
fi

# Update Ollama to use new location
echo "[5/5] Updating Ollama configuration..."
mkdir -p /etc/systemd/system/ollama.service.d/

cat > /etc/systemd/system/ollama.service.d/override.conf << 'EOF'
[Service]
Environment="OLLAMA_MODELS=/nvme3/ollama-models"
ReadWritePaths=/nvme3/ollama-models /nvme3/models /tmp
EOF

# Create symlink for compatibility
ln -s /nvme3/ollama-models /var/lib/ollama

# Set permissions
chown -R ollama:ollama /nvme3/ollama-models
chown -R ollama:ollama /nvme3/models

# Create Modelfiles for moved models
cat > /sda2/Modelfile.qwen << 'EOF'
FROM /nvme3/models/Qwen2.5-Coder-32B-Instruct

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

cat > /sda2/Modelfile.glm << 'EOF'
FROM /nvme3/models/GLM-4.7-Flash-4bit

PARAMETER num_ctx 4096
PARAMETER temperature 0.7
PARAMETER top_p 0.9

SYSTEM You are GLM-4.7-Flash, a helpful AI assistant.
EOF

cat > /sda2/Modelfile.kimi << 'EOF'
FROM /nvme3/AI Models/Kimi/Kimi-K2.5-Q4_K_M-00001-of-00013.gguf

PARAMETER num_ctx 8192
PARAMETER temperature 0.6
PARAMETER top_p 0.95
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"

SYSTEM You are Kimi, a large language model developed by Moonshot AI.

TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
EOF

# Reload and restart
echo ""
echo "Reloading systemd..."
systemctl daemon-reload
systemctl start ollama

sleep 2

echo ""
echo "=== Done! ==="
echo ""
echo "All models moved to /nvme3:"
du -sh /nvme3/models/* /nvme3/ollama-models/* 2>/dev/null | head -20
echo ""
echo "Available space: $(df -h /nvme3 | tail -1 | awk '{print $4}')"
echo ""
echo "Re-register models:"
echo "  ollama create qwen -f /sda2/Modelfile.qwen     # 62GB safetensors"
echo "  ollama create glm47 -f /sda2/Modelfile.glm     # GLM-4.7-Flash"
echo "  ollama create kimi -f /sda2/Modelfile.kimi     # 579GB Kimi"
echo ""
echo "Test:"
echo "  ollama run qwen 'Hello!'"
