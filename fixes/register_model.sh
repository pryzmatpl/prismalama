#!/bin/bash
# Register GGUF models with Ollama manifest system
# Usage: ./register_model.sh /path/to/model.gguf model-name

GGUF_FILE=$1
MODEL_NAME=$2

if [ -z "$GGUF_FILE" ] || [ -z "$MODEL_NAME" ]; then
    echo "Usage: $0 <path-to-gguf-file> <model-name>"
    exit 1
fi

# Set OLLAMA_MODELS if not already set
export OLLAMA_MODELS=${OLLAMA_MODELS:-/nvme3}

# Get the model file size
SIZE=$(stat -c%s "$GGUF_FILE")

# Calculate SHA256 digest
DIGEST=$(sha256sum "$GGUF_FILE" | cut -d' ' -f1)

# Create blobs directory
BLOBS_DIR="$OLLAMA_MODELS/blobs"
mkdir -p "$BLOBS_DIR"

# Copy file to blobs with sha256 name
cp "$GGUF_FILE" "$BLOBS_DIR/sha256-$DIGEST"

# Create manifest directory
MANIFEST_DIR="$OLLAMA_MODELS/manifests/registry.ollama.ai/library/$MODEL_NAME"
mkdir -p "$MANIFEST_DIR"

# Create config.json for the model
CONFIG_JSON=$(cat << 'EOF'
{
  "model_format": "gguf",
  "model_family": "llama",
  "model_type": "7B",
  "file_type": "Q4_K_M",
  "architecture": "llama",
  "os": "linux",
  "rootfs": {
    "type": "layers"
  }
}
EOF
)

CONFIG_DIGEST=$(echo -n "$CONFIG_JSON" | sha256sum | cut -d' ' -f1)
CONFIG_SIZE=$(echo -n "$CONFIG_JSON" | wc -c)

# Write config to blobs
echo "$CONFIG_JSON" > "$BLOBS_DIR/sha256-$CONFIG_DIGEST"

# Create the manifest file
cat > "$MANIFEST_DIR/latest" << EOF
{
  "schemaVersion": 2,
  "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
  "config": {
    "mediaType": "application/vnd.ollama.image.config",
    "digest": "sha256:$CONFIG_DIGEST",
    "size": $CONFIG_SIZE
  },
  "layers": [
    {
      "mediaType": "application/vnd.ollama.image.model",
      "digest": "sha256:$DIGEST",
      "size": $SIZE
    }
  ]
}
EOF

echo "Registered model: $MODEL_NAME"
echo "Model file: $GGUF_FILE"
echo "Digest: sha256:$DIGEST"
echo "Size: $SIZE bytes"
echo ""
echo "To use with opencode, run:"
echo "  export OLLAMA_MODELS=/nvme3"
echo "  opencode --model $MODEL_NAME"
