# LAST-FIX.md

This documents the last known steps to validate that models run from `/nvme3`.

## 1) Point Ollama to /nvme3

```sh
export OLLAMA_MODELS=/nvme3/ollama-models
```

Optional (persist in shell rc):

```sh
echo 'export OLLAMA_MODELS=/nvme3/ollama-models' >> ~/.bashrc
```

## 2) Start a server bound to /nvme3

Use a dedicated port so you do not interrupt an existing instance:

```sh
nohup env OLLAMA_MODELS=/nvme3/ollama-models OLLAMA_HOST=127.0.0.1:11435 ollama serve > /tmp/ollama-nvme3.log 2>&1 &
```

## 3) Validate model discovery from /nvme3

```sh
curl -s http://127.0.0.1:11435/api/tags
```

Expected: a JSON list of models (kimi, qwen, minimax, etc.) sourced from `/nvme3/ollama-models`.

## 4) Fix broken manifests (if any)

Known issue: some manifests still contain `/var/lib/ollama/blobs` in the `layers[].from` field.

- Example broken manifest:
  - `/nvme3/ollama-models/manifests/registry.ollama.ai/library/kimi/latest`

Correct approach:
1. Re-register from the actual GGUF file on `/nvme3` using a proper filename (not just a hash), so llama.cpp can parse split names.
2. Ensure all referenced blobs exist under `/nvme3/ollama-models/blobs`.

If you have a single GGUF on `/nvme3`:

```sh
cat > /tmp/Modelfile << 'EOF'
FROM /nvme3/models/kimi-k2.5.Q4_K_M.gguf
EOF

OLLAMA_MODELS=/nvme3/ollama-models ollama create kimi -f /tmp/Modelfile
```

If you have split parts, keep the original split filenames (e.g. `model-00001-of-00002.gguf`) and reference the first part in the Modelfile.

## 5) Run a short inference test

```sh
curl -s http://127.0.0.1:11435/api/generate -d '{"model":"kimi:latest","prompt":"Say ok.","stream":false,"num_predict":8}'
```

Expected: a response JSON with `eval_count` and `eval_duration` populated.

## 6) Quick throughput check

```sh
python - <<'PY'
import json, subprocess
payload = {
  "model": "kimi:latest",
  "prompt": "Return the single word ok.",
  "stream": False,
  "num_predict": 64
}
resp = subprocess.check_output([
  "curl","-s","http://127.0.0.1:11435/api/generate",
  "-d", json.dumps(payload)
])
print(resp.decode())
data = json.loads(resp)
if data.get("eval_count") and data.get("eval_duration"):
  tps = data["eval_count"] / (data["eval_duration"] / 1e9)
  print(f"TOKENS_PER_SEC={tps:.2f}")
PY
```

## 7) Logs for troubleshooting

```sh
tail -n 200 /tmp/ollama-nvme3.log
```

Look for:
- `bad manifest filepath` (missing or wrong blob path)
- `invalid split file name` (GGUF split naming issue)
- `unable to load model` (path, permissions, or format issues)

## Known issues discovered during last validation

- `kimi` manifest referenced `/var/lib/ollama/blobs/...` and produced `invalid split file name`.
- `kimi-k2.5` manifest referenced a missing blob `sha256-fe178...`.
- `minimax-m2.1` loaded but `/api/generate` timed out (needs a shorter prompt or a verified model registration).
