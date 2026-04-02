#!/bin/bash
# Simple wrapper to run Kimi via Ollama using the existing llama.cpp integration
# This demonstrates how Ollama + prismalama + llama.cpp work together

echo "=== Kimi K2.5 via Ollama (llama.cpp backend) ==="
echo ""
echo "Architecture:"
echo "  Ollama (API server) → llamarunner (llama.cpp) → Kimi GGUF files"
echo ""
echo "To add Kimi to Ollama without copying 579GB:"
echo ""
echo "1. Create symlink in blobs directory:"
echo "   sudo ln -s '/nvme3/AI Models/Kimi/Kimi-K2.5-Q4_K_M-00001-of-00013.gguf' \\"
echo "     /var/lib/ollama/blobs/sha256-\$(sha256sum '/nvme3/AI Models/Kimi/Kimi-K2.5-Q4_K_M-00001-of-00013.gguf' | cut -d' ' -f1)"
echo ""
echo "2. Run the setup script with sudo:"
echo "   sudo bash /sda2/add-kimi-to-ollama.sh"
echo ""
echo "3. Once added, use Kimi through Ollama:"
echo "   ollama run kimi-k2.5 'Hello!'"
echo ""
echo "Current working models:"
echo "   ollama list"
echo ""
ollama list
