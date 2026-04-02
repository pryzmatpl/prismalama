#!/bin/bash
# Alternative: Just change OLLAMA_MODELS without moving existing models

echo "=== Configuring Ollama to use /nvme3 for models ==="
echo ""

if [ "$EUID" -ne 0 ]; then 
    echo "Please run with sudo"
    exit 1
fi

# Stop Ollama
systemctl stop ollama

# Create override directory
mkdir -p /etc/systemd/system/ollama.service.d/

# Create override with new model path
cat > /etc/systemd/system/ollama.service.d/override.conf << 'EOF'
[Service]
Environment="OLLAMA_MODELS=/nvme3/ollama-models"
ReadWritePaths=/nvme3/ollama-models /var/lib/ollama /tmp
EOF

# Create the directory
mkdir -p /nvme3/ollama-models
chown ollama:ollama /nvme3/ollama-models

# Update systemd
systemctl daemon-reload
systemctl start ollama

sleep 2

echo "✓ Ollama now uses /nvme3/ollama-models for new models"
echo ""
echo "Existing models still in /var/lib/ollama/"
echo "New models (including Kimi) will go to /nvme3/ollama-models/"
echo ""
echo "Add Kimi:"
echo "  ollama create kimi-k2.5 -f /sda2/Modelfile.kimi-direct"
echo ""
echo "Note: You may want to migrate existing models manually:"
echo "  sudo cp -r /var/lib/ollama/* /nvme3/ollama-models/"
