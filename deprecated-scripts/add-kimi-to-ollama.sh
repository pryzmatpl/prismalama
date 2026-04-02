#!/bin/bash
# Add Kimi K2.5 to Ollama via symlinks (no copying!)
# This leverages Ollama's existing llama.cpp integration

set -e

KIMI_DIR="/nvme3/AI Models/Kimi"
OLLAMA_BLOBS="/var/lib/ollama/blobs"
OLLAMA_MANIFESTS="/var/lib/ollama/manifests/registry.ollama.ai/library"

echo "=== Adding Kimi K2.5 to Ollama (Symlink Method) ==="
echo ""

# Check if running as root/sudo
if [ "$EUID" -ne 0 ]; then 
    echo "Please run with sudo"
    exit 1
fi

# Create model directory
mkdir -p "$OLLAMA_MANIFESTS/kimi-k2.5"

echo "[1/4] Calculating hashes and creating symlinks..."

# Function to calculate sha256 and create symlink
link_model_file() {
    local file="$1"
    local filename=$(basename "$file")
    echo "  Processing: $filename"
    
    # Calculate sha256 hash
    local hash=$(sha256sum "$file" | cut -d' ' -f1)
    local blob_name="sha256-$hash"
    
    # Create symlink if it doesn't exist
    if [ ! -L "$OLLAMA_BLOBS/$blob_name" ]; then
        ln -s "$file" "$OLLAMA_BLOBS/$blob_name"
        echo "    ✓ Linked: $blob_name"
    else
        echo "    ✓ Already linked: $blob_name"
    fi
    
    # Return the hash for manifest
    echo "$hash"
}

# Process first shard (main file) - Ollama will auto-detect multi-shard
MAIN_FILE="$KIMI_DIR/Kimi-K2.5-Q4_K_M-00001-of-00013.gguf"
MAIN_HASH=$(link_model_file "$MAIN_FILE")
MAIN_SIZE=$(stat -c%s "$MAIN_FILE")

echo ""
echo "[2/4] Creating config blob..."

# Create config JSON
CONFIG_JSON='{"model_format":"gguf","model_family":"kimi","model_families":["kimi"],"model_type":"K2.5","file_type":"Q4_K_M","architecture":"MoE","parameters":"32B","context_length":32768,"embedding_length":8192," quantization":"Q4_K_M"}'
CONFIG_HASH=$(echo -n "$CONFIG_JSON" | sha256sum | cut -d' ' -f1)
CONFIG_BLOB="sha256-$CONFIG_HASH"

if [ ! -f "$OLLAMA_BLOBS/$CONFIG_BLOB" ]; then
    echo "$CONFIG_JSON" > "$OLLAMA_BLOBS/$CONFIG_BLOB"
fi
echo "  ✓ Config: $CONFIG_BLOB"

echo ""
echo "[3/4] Creating system prompt blob..."

# System prompt
SYSTEM_PROMPT="You are Kimi, a large language model developed by Moonshot AI. You are helpful, harmless, and honest. You excel at long-context understanding and can process up to 256,000 tokens."
SYSTEM_HASH=$(echo -n "$SYSTEM_PROMPT" | sha256sum | cut -d' ' -f1)
SYSTEM_BLOB="sha256-$SYSTEM_HASH"

if [ ! -f "$OLLAMA_BLOBS/$SYSTEM_BLOB" ]; then
    echo -n "$SYSTEM_PROMPT" > "$OLLAMA_BLOBS/$SYSTEM_BLOB"
fi
echo "  ✓ System prompt: $SYSTEM_BLOB"

echo ""
echo "[4/4] Creating manifest..."

# Create manifest JSON
MANIFEST=$(cat <<EOF
{
  "schemaVersion": 2,
  "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
  "config": {
    "mediaType": "application/vnd.docker.container.image.v1+json",
    "digest": "$CONFIG_BLOB",
    "size": $(stat -c%s "$OLLAMA_BLOBS/$CONFIG_BLOB")
  },
  "layers": [
    {
      "mediaType": "application/vnd.ollama.image.model",
      "digest": "sha256-$MAIN_HASH",
      "size": $MAIN_SIZE,
      "from": "$MAIN_FILE"
    },
    {
      "mediaType": "application/vnd.ollama.image.system",
      "digest": "$SYSTEM_BLOB",
      "size": $(stat -c%s "$OLLAMA_BLOBS/$SYSTEM_BLOB")
    }
  ]
}
EOF
)

echo "$MANIFEST" > "$OLLAMA_MANIFESTS/kimi-k2.5/latest"
echo "  ✓ Manifest created"

# Set ownership
chown -R ollama:ollama "$OLLAMA_BLOBS"
chown -R ollama:ollama "$OLLAMA_MANIFESTS"

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Kimi K2.5 is now available in Ollama:"
echo "  ollama list | grep kimi"
echo ""
echo "Run inference:"
echo "  ollama run kimi-k2.5 'Hello!'"
echo ""
echo "The model is NOT copied - it uses symlinks to:"
echo "  /nvme3/AI Models/Kimi/"
