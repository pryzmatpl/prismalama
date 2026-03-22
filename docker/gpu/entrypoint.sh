#!/usr/bin/env bash
set -euo pipefail
export CGO_ENABLED=1
export OLLAMA_LIBRARY_PATH="${OLLAMA_LIBRARY_PATH:-/usr/lib/ollama/rocm}"
export LD_LIBRARY_PATH="/usr/lib/ollama/rocm:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
if [[ -d /workspace ]]; then
	cd /workspace
fi
exec /usr/bin/ollama "$@"
