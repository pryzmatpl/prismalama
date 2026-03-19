#!/bin/bash
# Prismalama Sync Script for /sda2 migration
# Run this script with sudo to complete the setup

set -e

echo "=== Prismalama Sync for /sda2 ==="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    log_error "Please run this script with sudo"
    exit 1
fi

# Update systemd service
log_info "Updating systemd service..."
cat > /usr/lib/systemd/system/ollama.service << 'EOF'
[Unit]
Description=Ollama Server with AirLLM Integration (ROCm)
Documentation=https://github.com/ollama/ollama
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=ollama
EnvironmentFile=/etc/default/ollama
ExecStart=/usr/bin/ollama serve
Restart=always
RestartSec=3

# Security settings
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/sda2/airllm /var/lib/ollama /tmp
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

# Update environment config
log_info "Updating environment configuration..."
cat > /etc/default/ollama << 'EOF'
OLLAMA_MODELS="/sda2/airllm"
OLLAMA_HOST="127.0.0.1:11434"
HSA_OVERRIDE_GFX_VERSION=11.0.0
AIRLLM_COMPRESSION="4bit"
AIRLLM_DEVICE="cuda:0"
PYTHONPATH="/usr/share/ollama/airllm:/usr/share/ollama/airllm/air_llm:${PYTHONPATH}"

# ROCm specific
HIP_VISIBLE_DEVICES=0
EOF

# Install new binaries
log_info "Installing prismalama binaries..."
if [ -f "/sda2/prismalama/build/pkg/usr/bin/ollama" ]; then
    cp /sda2/prismalama/build/pkg/usr/bin/ollama /usr/bin/ollama.prismalama
    cp /sda2/prismalama/build/pkg/usr/bin/ollama-airllm /usr/bin/ollama-airllm
    chmod +x /usr/bin/ollama.prismalama /usr/bin/ollama-airllm
    log_info "Binaries installed successfully"
else
    log_warn "Build binaries not found, using existing installation"
fi

# Create necessary directories
log_info "Creating directories..."
mkdir -p /sda2/airllm
mkdir -p /var/lib/ollama

# Set permissions
log_info "Setting permissions..."
if id "ollama" &>/dev/null; then
    chown -R ollama:ollama /sda2/airllm 2>/dev/null || true
    chown -R ollama:ollama /var/lib/ollama 2>/dev/null || true
else
    log_warn "ollama user not found, creating..."
    useradd -r -s /bin/false ollama 2>/dev/null || true
    chown -R ollama:ollama /sda2/airllm 2>/dev/null || true
    chown -R ollama:ollama /var/lib/ollama 2>/dev/null || true
fi

# Reload systemd
log_info "Reloading systemd..."
systemctl daemon-reload

# Restart service
log_info "Restarting ollama service..."
systemctl restart ollama
sleep 2

# Check status
if systemctl is-active --quiet ollama; then
    log_info "Ollama service is running!"
    systemctl status ollama --no-pager -l
else
    log_error "Ollama service failed to start. Checking logs..."
    journalctl -u ollama --no-pager -n 20
fi

echo ""
echo "=== Sync Complete ==="
echo ""
echo "Next steps:"
echo "1. Register models with: ollama create qwen25-coder -f /sda2/Modelfile"
echo "2. Verify models with: ollama list"
echo "3. Test with: ollama run qwen25-coder"
echo ""
