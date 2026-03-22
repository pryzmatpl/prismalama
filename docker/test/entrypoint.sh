#!/usr/bin/env bash
# Mount repo at /workspace; keep system ollama + GGML in /usr for CGO tests.
set -euo pipefail
export CGO_ENABLED=1
export OLLAMA_BIN="${OLLAMA_BIN:-/usr/bin/ollama}"
export OLLAMA_LIBRARY_PATH="${OLLAMA_LIBRARY_PATH:-/usr/lib/ollama}"
export LD_LIBRARY_PATH="/usr/lib/ollama:${LD_LIBRARY_PATH:-}"
if [[ -d /workspace ]]; then
	cd /workspace
fi
exec "$@"
