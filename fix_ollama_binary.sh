#!/bin/bash

# Copy Ollama binary to system location
cp /sda2/prismalama/build_ollama_airllm/ollama-bin /usr/bin/ollama

# Set ownership and permissions
chown root:root /usr/bin/ollama
chmod 755 /usr/bin/ollama

# Start Ollama service
systemctl start ollama

echo "Ollama binary installed and service started. Check with 'ollama ps' after a few seconds."