#!/usr/bin/env bash
# Record live prismalama /api/generate tok/s. Prints one JSON object per run.
# Usage:
#   scripts/xtx-generate-bench.sh qwen3:0.6b
#   scripts/xtx-generate-bench.sh qwen35-uncensored:latest 16 256
set -euo pipefail

MODEL="${1:-qwen3:0.6b}"
NPRED="${2:-16}"
NCTX="${3:-256}"
PROMPT="${4:-Say hi in one word.}"
HOST="${PRISMALAMA_HOST:-http://127.0.0.1:11434}"

python3 - "$HOST" "$MODEL" "$PROMPT" "$NPRED" "$NCTX" <<'PY'
import json, sys, urllib.request
host, model, prompt, npred, nctx = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
body = json.dumps({
    "model": model,
    "prompt": prompt,
    "stream": False,
    "options": {"num_predict": npred, "num_ctx": nctx, "temperature": 0},
}).encode()
req = urllib.request.Request(host + "/api/generate", data=body, method="POST")
with urllib.request.urlopen(req, timeout=1800) as resp:
    d = json.load(resp)
ec = int(d.get("eval_count") or 0)
ed = int(d.get("eval_duration") or 0)
pc = int(d.get("prompt_eval_count") or 0)
pd = int(d.get("prompt_eval_duration") or 0)
out = {
    "model": model,
    "error": d.get("error"),
    "response": (d.get("response") or "")[:160],
    "prompt_eval_count": pc,
    "prompt_tok_s": (pc / (pd / 1e9)) if pd else None,
    "eval_count": ec,
    "eval_tok_s": (ec / (ed / 1e9)) if ed else None,
    "load_s": (d.get("load_duration") or 0) / 1e9,
    "total_s": (d.get("total_duration") or 0) / 1e9,
    "num_predict": npred,
    "num_ctx": nctx,
}
print(json.dumps(out, ensure_ascii=False))
if d.get("error"):
    sys.exit(1)
PY
