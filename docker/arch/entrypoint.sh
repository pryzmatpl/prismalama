#!/usr/bin/env bash
set -euo pipefail
export OLLAMA_LIBRARY_PATH="${OLLAMA_LIBRARY_PATH:-/usr/lib/ollama/rocm}"
export LD_LIBRARY_PATH="/usr/lib/ollama/rocm:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
# Match service expectations: writable store + stable HOME for cache
export HOME="${HOME:-/var/lib/ollama}"
if [[ -d /var/lib/ollama ]]; then
	cd /var/lib/ollama
fi
if ! id ollama &>/dev/null; then
	exec /usr/bin/ollama "$@"
fi
exec runuser -u ollama -- env HOME=/var/lib/ollama OLLAMA_HOST="${OLLAMA_HOST:-0.0.0.0:11434}" \
	OLLAMA_MODELS="${OLLAMA_MODELS:-/var/lib/ollama}" \
	OLLAMA_LIBRARY_PATH="${OLLAMA_LIBRARY_PATH:-/usr/lib/ollama/rocm}" \
	LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/usr/lib/ollama/rocm:/opt/rocm/lib}" \
	/usr/bin/ollama "$@"
