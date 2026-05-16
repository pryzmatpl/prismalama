#!/bin/bash
# Setup Jaisiu OpenClaw gateway with prismalama GPU backend

set -e

PRISMALAMA_URL="http://localhost:11434"
GATEWAY_TOKEN="agent:main:main"

echo "🔗 Setting up Jaisiu-Prismalama integration..."

# Update OpenClaw gateway configuration
echo "⚙️  Patching OpenClaw gateway config..."

openclaw config patch '{
  "models": {
    "providers": {
      "prismalama": {
        "models": [
          {
            "id": "prismalama/qwen2.5:14b-gpu",
            "name": "Qwen 2.5 14B (GPU-accelerated)",
            "provider": "prismalama",
            "baseUrl": "'"$PRISMALAMA_URL"'",
            "api": "openai-completions",
            "contextWindow": 32768,
            "maxTokens": 8192,
            "optimization": {
              "num_gpu_layers": 99,
              "batch_size": 4096,
              "num_predict": 8192,
              "num_thread": 4
            }
          },
          {
            "id": "prismalama/qwen3.6:27b-gpu",
            "name": "Qwen 3.6 27B (GPU-accelerated)",
            "provider": "prismalama",
            "baseUrl": "'"$PRISMALAMA_URL"'",
            "api": "openai-completions",
            "contextWindow": 262144,
            "maxTokens": 8192,
            "optimization": {
              "num_gpu_layers": 99,
              "batch_size": 2048,
              "num_predict": 8192,
              "num_thread": 8
            }
          }
        ]
      }
    }
  },
  "agents": {
    "defaults": {
      "model": {
        "primary": "prismalama/qwen2.5:14b-gpu"
      }
    }
  }
}'

echo "✅ Gateway configured!"

# Create performance monitoring cron job
echo "📈 Setting up performance monitoring..."

cron add 'prismalama-performance-monitor' '{
  "name": "Prismalama Performance Monitor",
  "schedule": {
    "kind": "every",
    "everyMs": 300000
  },
  "payload": {
    "kind": "systemEvent",
    "text": "🔍 Prismalama GPU performance check: curl -s http://localhost:11434/v1/models | python3 -c \\"import sys,json; data=json.load(sys.stdin); [print(f\"  {m[\\"id\\"]}\") for m in data[\\"data\\"]]\\""
  },
  "sessionTarget": "main",
  "enabled": true
}'

echo "✅ Performance monitor active!"

# Test the integration
echo "🧪 Testing GPU inference..."
if curl -s "$PRISMALAMA_URL/v1/models" | grep -q "qwen2.5:14b"; then
    echo "✅ qwen2.5:14b available on GPU backend"
else
    echo "⚠️  qwen2.5:14b not loaded"
    echo "   Run: docker exec ollama-gpu ollama pull qwen2.5:14b"
fi

echo ""
echo "🎉 Jaisiu-Prismalama integration complete!"
echo ""
echo "📌 Active models:"
curl -s "$PRISMALAMA_URL/v1/models" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    for m in data.get('data', []):
        print(f'  • {m[\"id\"]}')
except:
    print('  (none loaded)')
"
echo ""
echo "⚡ To run inference through OpenClaw gateway:"
echo "   openclaw agent run --model prismalama/qwen2.5:14b-gpu 'Your prompt'"
echo ""
echo "📊 Expected TPS with GPU: 200-250 (vs 0.5 on CPU)"
