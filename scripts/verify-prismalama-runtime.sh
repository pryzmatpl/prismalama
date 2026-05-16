#!/usr/bin/env bash
# Print GET /api/prismalama/capabilities (operator_hints + env passthrough).
# Requires a running Prismalama/Ollama-compatible server.
#
# Usage:
#   OLLAMA_HOST=127.0.0.1:11434 ./verify-prismalama-runtime.sh

set -euo pipefail

: "${OLLAMA_HOST:=127.0.0.1:11434}"
case "${OLLAMA_HOST}" in
http://* | https://*) BASE="${OLLAMA_HOST%/}" ;;
*) BASE="http://${OLLAMA_HOST%/}" ;;
esac

URL="${BASE}/api/prismalama/capabilities"

if ! command -v curl >/dev/null; then
	echo "error: curl required" >&2
	exit 1
fi
if ! command -v python3 >/dev/null; then
	echo "error: python3 required" >&2
	exit 1
fi

echo "GET ${URL}"
curl -sS -f "${URL}" | python3 -c 'import json,sys
j=json.load(sys.stdin)
print("version:", j.get("version",""))
env=j.get("environment") or {}
for k in sorted(env.keys()):
    print(f"  {k}: {env[k]!r}")
hints=j.get("operator_hints") or []
print("operator_hints:")
for h in hints:
    print(" -", h)
'
