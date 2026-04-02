#!/bin/bash
# Move Ollama models to /nvme3 where there's space for Kimi

echo "=== Moving Ollama Models to /nvme3 ==="
echo ""

if [ "$EUID" -ne 0 ]; then 
    echo "Please run with sudo"
    exit 1
fi

# Stop Ollama
echo "[1/5] Stopping Ollama..."
systemctl stop ollama

# Create new models directory on nvme3
echo "[2/5] Creating models directory on /nvme3..."
mkdir -p "/nvme3/ollama-models"

# Copy existing models (or move them)
echo "[3/5] Migrating existing models..."
if [ -d "/var/lib/ollama" ]; then
    # Copy blobs and manifests
    cp -r /var/lib/ollama/blobs "/nvme3/ollama-models/" 2>/dev/null || true
    cp -r /var/lib/ollama/manifests "/nvme3/ollama-models/" 2>/dev/null || true
    echo "  ✓ Existing models copied"
fi

# Create symlink from old location to new (for backwards compatibility)
echo "[4/5] Creating compatibility symlinks..."
if [ -d "/var/lib/ollama" ]; then
    mv /var/lib/ollama /var/lib/ollama-backup
fi
ln -s "/nvme3/ollama-models" /var/lib/ollama

# Update systemd service to use new location
echo "[5/5] Updating systemd service..."
cat > /etc/systemd/system/ollama.service.d/override.conf << 'EOF'
[Service]
Environment="OLLAMA_MODELS=/nvme3/ollama-models"
ReadWritePaths=/nvme3/ollama-models /tmp
EOF

# Also update the main environment file
echo 'OLLAMA_MODELS="/nvme3/ollama-models"' > /etc/default/ollama

# Set permissions
chown -R ollama:ollama "/nvme3/ollama-models"
chown ollama:ollama /var/lib/ollama

# Reload and start
echo ""
echo "Reloading systemd..."
systemctl daemon-reload
systemctl start ollama

sleep 2

echo ""
echo "=== Migration Complete! ==="
echo ""
echo "Ollama models are now stored on /nvme3"
echo "Available space: $(df -h /nvme3 | tail -1 | awk '{print $4}')"
echo ""
echo "Now you can add Kimi:"
echo "  ollama create kimi-k2.5 -f /sda2/Modelfile.kimi-direct"
echo ""
echo "Or restore from backup:"
echo "  sudo mv /var/lib/ollama-backup /var/lib/ollama-old"
