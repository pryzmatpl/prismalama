#!/bin/bash
# Update prismalama configuration for /nvme3 setup
# This ensures Ollama + ROCm + AirLLM work together for Kimi on potato hardware

echo "=== Updating prismalama for /nvme3 Configuration ==="
echo ""
echo "Goal: Ollama + ROCm + AirLLM for Kimi on potato hardware 🥔🚀"
echo ""

if [ "$EUID" -ne 0 ]; then 
    echo "Please run with sudo"
    exit 1
fi

cd /sda2/prismalama

# Step 1: Update environment configuration
echo "[1/5] Updating environment configuration..."
cat > build/pkg/etc/default/ollama << 'EOF'
OLLAMA_MODELS="/nvme3/ollama-models"
OLLAMA_HOST="127.0.0.1:11434"
HSA_OVERRIDE_GFX_VERSION=11.0.0
AIRLLM_COMPRESSION="4bit"
AIRLLM_DEVICE="cuda:0"
PYTHONPATH="/usr/share/ollama/airllm:/usr/share/ollama/airllm/air_llm:${PYTHONPATH}"

# ROCm specific
HIP_VISIBLE_DEVICES=0

# Potato Machine Optimizations
OLLAMA_NUM_PARALLEL=1
OLLAMA_MAX_LOADED_MODELS=1
OLLAMA_CONTEXT_LENGTH=8192
EOF

# Step 2: Update systemd service
echo "[2/5] Updating systemd service for /nvme3..."
cat > build/pkg/usr/lib/systemd/system/ollama.service << 'EOF'
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
ReadWritePaths=/nvme3/ollama-models /nvme3/models /nvme3/AI\x20Models /tmp
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

# Step 3: Rebuild the package
echo "[3/5] Rebuilding prismalama package..."
if [ -f "PKGBUILD" ]; then
    # Update paths in PKGBUILD
    sed -i 's|OLLAMA_MODELS=\"/run/media/piotro/CACHE/airllm\"|OLLAMA_MODELS=\"/nvme3/ollama-models\"|g' PKGBUILD
    sed -i 's|ReadWritePaths=/run/media/piotro/CACHE/airllm|ReadWritePaths=/nvme3/ollama-models /nvme3/models|g' PKGBUILD
    
    # Build
    makepkg -si --noconfirm 2>/dev/null || {
        echo "  Package build attempted (may require manual intervention)"
    }
fi

# Step 4: Install updated configuration
echo "[4/5] Installing updated configuration..."
cp build/pkg/etc/default/ollama /etc/default/ollama
cp build/pkg/usr/lib/systemd/system/ollama.service /usr/lib/systemd/system/ollama.service

# Update systemd override
mkdir -p /etc/systemd/system/ollama.service.d/
cat > /etc/systemd/system/ollama.service.d/override.conf << 'EOF'
[Service]
Environment="OLLAMA_MODELS=/nvme3/ollama-models"
ReadWritePaths=/nvme3/ollama-models /nvme3/models /nvme3/AI\x20Models /tmp
EOF

# Step 5: Restart and verify
echo "[5/5] Restarting Ollama..."
systemctl daemon-reload
systemctl restart ollama

sleep 3

echo ""
echo "=== Configuration Updated! ==="
echo ""
echo "Ollama + ROCm + AirLLM Configuration:"
echo "  Models: /nvme3/ollama-models"
echo "  ROCm: gfx1100 (RX 7900 XTX)"
echo "  AirLLM: 4-bit compression, layer offloading"
echo "  Optimized for: Potato hardware (low memory)"
echo ""
echo "Status:"
systemctl is-active ollama && echo "  ✓ Ollama is running" || echo "  ✗ Ollama is not running"
echo ""
echo "Models available:"
ollama list 2>/dev/null || echo "  (Loading...)"
echo ""
echo "Test Kimi on potato hardware:"
echo "  ollama run kimi 'Hello! Can you help me save humanity?'"
