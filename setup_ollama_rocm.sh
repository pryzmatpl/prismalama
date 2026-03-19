#!/bin/bash

# Install Ollama AirLLM ROCM package
pacman -U /sda2/prismalama/build_ollama_airllm/ollama-airllm-rocm-0.16.2-3-x86_64.pkg.tar.zst --noconfirm

# Create systemd service file with ROCM environment
cat > /usr/lib/systemd/system/ollama.service <<EOF
[Unit]
Description=Ollama Server with AirLLM Integration (ROCm)
After=network.target
StartLimitIntervalSec=0

[Service]
Type=simple
Restart=always
RestartSec=3
ExecStart=/usr/bin/ollama serve
Environment=OLLAMA_MODELS=/nvme3/models
Environment=OLLAMA_KV_CACHE_TYPE=q8_0
Environment=OLLAMA_NUM_PARALLEL=1
Environment=HSA_ENABLE_SDMA=0
Environment=HIP_VISIBLE_DEVICES=2
ReadWritePaths=
ReadWritePaths=/nvme3/models
ReadWritePaths=/var/lib/ollama
ReadWritePaths=/tmp

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd configuration
systemctl daemon-reload

# Start and enable Ollama service
systemctl start ollama
systemctl enable ollama

echo "Ollama with ROCM support installed."
echo "Service is starting. Run 'ollama ps' after a few seconds to check GPU usage."