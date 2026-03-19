#!/bin/bash
# Fix the corrupted Kimi manifest

echo "Fixing Kimi manifest..."

cat > /tmp/kimi-fix.json << 'EOF'
{
  "schemaVersion": 2,
  "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
  "config": {
    "mediaType": "application/vnd.docker.container.image.v1+json",
    "digest": "sha256-51323c79e7167dd64bebcaf4e75960134caf3c45979702c23ee22240cdd62eeb",
    "size": 225
  },
  "layers": [
    {
      "mediaType": "application/vnd.ollama.image.model",
      "digest": "sha256-b477ea345b3358daddd6d817b84edb79e3ebf616db4b7b1ebe1086b12271da65",
      "size": 46304137792,
      "from": "/nvme3/AI Models/Kimi/Kimi-K2.5-Q4_K_M-00001-of-00013.gguf"
    },
    {
      "mediaType": "application/vnd.ollama.image.system",
      "digest": "sha256-e829a22705eccf388cf37209c9364014ab8b8e0c83b478620861133f131042df",
      "size": 179
    }
  ]
}
EOF

sudo cp /tmp/kimi-fix.json /var/lib/ollama/manifests/registry.ollama.ai/library/kimi-k2.5/latest
sudo chown ollama:ollama /var/lib/ollama/manifests/registry.ollama.ai/library/kimi-k2.5/latest

echo "✓ Manifest fixed!"
echo ""
echo "Verifying:"
ollama list | grep kimi

echo ""
echo "Test with:"
echo "  ollama run kimi-k2.5 'Hello!'"
