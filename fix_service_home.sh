#!/bin/bash

# Edit the service file to add HOME
sed -i '/^Environment=OLLAMA_MODELS=/i Environment=HOME=/var/lib/ollama' /usr/lib/systemd/system/ollama.service

# Reload and start
systemctl daemon-reload
systemctl start ollama

echo "Added HOME to service, reloaded and started."