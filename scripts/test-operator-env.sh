#!/usr/bin/env bash
# Smoke test for scripts/operator-env-large-models.sh.
# Sources the operator-env script in a subshell and asserts each documented
# default value. Bash-only; no Go test runner required.
#
# Usage:
#   bash scripts/test-operator-env.sh
# Exit: 0 on success, non-zero on first assertion failure.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="${REPO_ROOT}/scripts/operator-env-large-models.sh"

if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: $SCRIPT not found" >&2
    exit 1
fi

# Run the script in a clean subshell, then check the resulting env.
# We use 'env -i' to ensure no inherited env vars mask the defaults.
# Allow through PATH and HOME (bash requires them).
fail=0
check() {
    local name="$1"
    local want="$2"
    local got="$3"
    if [[ "$got" == "$want" ]]; then
        echo "  ok    $name=$got"
    else
        echo "  FAIL  $name: want=$want got=$got" >&2
        fail=1
    fi
}

echo "Source: $SCRIPT"
echo "Asserting defaults after sourcing with no inherited env..."

# shellcheck disable=SC1090
result="$(env -i PATH="$PATH" HOME="$HOME" bash -c "set -a; source '$SCRIPT'; set +a; \
    echo \"OLLAMA_MEMORY_POLICY=\${OLLAMA_MEMORY_POLICY:-} \
OLLAMA_LAYER_STREAMING=\${OLLAMA_LAYER_STREAMING:-} \
OLLAMA_STREAMING_BUDGET=\${OLLAMA_STREAMING_BUDGET:-} \
OLLAMA_NEW_ENGINE=\${OLLAMA_NEW_ENGINE:-} \
OLLAMA_MMAP_ALLOW_LOW_RAM=\${OLLAMA_MMAP_ALLOW_LOW_RAM:-} \
OLLAMA_VULKAN=\${OLLAMA_VULKAN:-}\"")"

# Parse the key=value pairs
declare -A kv
for pair in $result; do
    k="${pair%%=*}"
    v="${pair#*=}"
    kv["$k"]="$v"
done

check "OLLAMA_MEMORY_POLICY"        "balanced" "${kv[OLLAMA_MEMORY_POLICY]:-}"
check "OLLAMA_LAYER_STREAMING"      "1"        "${kv[OLLAMA_LAYER_STREAMING]:-}"
check "OLLAMA_STREAMING_BUDGET"     "6442450944" "${kv[OLLAMA_STREAMING_BUDGET]:-}"
check "OLLAMA_NEW_ENGINE"           "1"        "${kv[OLLAMA_NEW_ENGINE]:-}"
check "OLLAMA_MMAP_ALLOW_LOW_RAM"   "1"        "${kv[OLLAMA_MMAP_ALLOW_LOW_RAM]:-}"
check "OLLAMA_VULKAN"               "1"        "${kv[OLLAMA_VULKAN]:-}"

if [[ "$fail" -ne 0 ]]; then
    echo "" >&2
    echo "FAIL: one or more operator-env defaults drifted from docs." >&2
    echo "Update scripts/operator-env-large-models.sh AND docs/PACKAGING_DEFAULTS.md together." >&2
    exit 1
fi

echo ""
echo "OK: operator-env-large-models.sh defaults match docs/PACKAGING_DEFAULTS.md"