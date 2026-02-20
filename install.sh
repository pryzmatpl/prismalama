#!/bin/bash
# Quick install script for prismalama on /sda2
# Run with sudo

set -e

echo "=== Prismalama Installation for /sda2 ==="
echo ""

if [ "$EUID" -ne 0 ]; then 
    echo "Please run with sudo"
    exit 1
fi

cd /sda2/prismalama

echo "[1/5] Installing binaries..."
cp build/pkg/usr/bin/ollama /usr/bin/ollama.prismalama
cp build/pkg/usr/bin/ollama-airllm /usr/bin/ollama-airllm
chmod +x /usr/bin/ollama.prismalama /usr/bin/ollama-airllm

echo "[2/5] Installing libraries..."
mkdir -p /usr/lib/ollama
cp -r build/pkg/usr/lib/ollama/* /usr/lib/ollama/ 2>/dev/null || true

echo "[3/5] Installing AirLLM..."
mkdir -p /usr/share/ollama
cp -r build/pkg/usr/share/ollama/airllm /usr/share/ollama/ 2>/dev/null || true
cp build/pkg/usr/share/ollama/airllm_runner.py /usr/share/ollama/ 2>/dev/null || true

echo "[4/5] Installing systemd service..."
cp build/pkg/usr/lib/systemd/system/ollama.service /usr/lib/systemd/system/ollama.service
cp build/pkg/usr/lib/sysusers.d/ollama.conf /usr/lib/sysusers.d/ollama.conf

echo "[5/5] Installing configuration..."
cp build/pkg/etc/default/ollama /etc/default/ollama

# Create directories
mkdir -p /sda2/airllm
mkdir -p /var/lib/ollama

# Set permissions
if id "ollama" &>/dev/null; then
    chown -R ollama:ollama /sda2/airllm 2>/dev/null || true
    chown -R ollama:ollama /var/lib/ollama 2>/dev/null || true
fi

# Reload and restart
systemctl daemon-reload
systemctl restart ollama

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Check status: systemctl status ollama"
echo "View logs: journalctl -u ollama -f"
