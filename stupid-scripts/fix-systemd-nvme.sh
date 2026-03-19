#!/bin/bash
# Fix systemd service to allow access to /nvme3

echo "Updating Ollama systemd service to access /nvme3..."

# Create updated service file
sudo tee /usr/lib/systemd/system/ollama.service > /dev/null << 'EOF'
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
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/sda2/airllm /var/lib/ollama /nvme3
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

echo "✓ Service file updated"
echo ""
echo "Reloading systemd and restarting Ollama..."
sudo systemctl daemon-reload
sudo systemctl restart ollama

sleep 2

echo ""
echo "Checking status:"
systemctl status ollama --no-pager | head -10

echo ""
echo "✓ Done! Try running:"
echo "  ollama run kimi-k2.5 'Hello!'"
