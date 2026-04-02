# Add Kimi K2.5 to Ollama via Symlinks
# Run these commands with sudo

# 1. Calculate hash of main model file
KIMI_HASH=$(sha256sum "/nvme3/AI Models/Kimi/Kimi-K2.5-Q4_K_M-00001-of-00013.gguf" | cut -d' ' -f1)
echo "Model hash: $KIMI_HASH"

# 2. Create symlink in Ollama blobs directory
sudo ln -s "/nvme3/AI Models/Kimi/Kimi-K2.5-Q4_K_M-00001-of-00013.gguf" "/var/lib/ollama/blobs/sha256-$KIMI_HASH"

# 3. Create config blob
CONFIG_JSON='{"model_format":"gguf","model_family":"kimi","model_type":"K2.5","file_type":"Q4_K_M","parameters":"32B","context_length":32768}'
echo "$CONFIG_JSON" | sudo tee "/var/lib/ollama/blobs/sha256-$(echo -n "$CONFIG_JSON" | sha256sum | cut -d' ' -f1)" > /dev/null

# 4. Create system prompt blob
echo -n "You are Kimi, a large language model developed by Moonshot AI." | sudo tee "/var/lib/ollama/blobs/sha256-$(echo -n "You are Kimi" | sha256sum | cut -d' ' -f1)" > /dev/null

# 5. Create manifest directory
sudo mkdir -p "/var/lib/ollama/manifests/registry.ollama.ai/library/kimi-k2.5"

# 6. Create manifest (simplified version - you'll need to manually edit)
echo "Create manifest manually at: /var/lib/ollama/manifests/registry.ollama.ai/library/kimi-k2.5/latest"
