#!/usr/bin/env bash
# Ship gate: integration tests, then prismalama-ollama package (see docs/DEVELOPER.md § Ship gate).
# See docs/SHIP_CHECK.md for prerequisites, what runs in order, and failure-mode table.
#
# Usage:
#   ./scripts/ship-check.sh                 # integration + package (default)
#   SHIP_SKIP_PKG=1 ./scripts/ship-check.sh # integration only (fast; CPU host OK)
#   ./scripts/ship-check.sh --list          # print the steps without running them
#   ./scripts/ship-check.sh --help          # show help
#
# Env vars (all optional):
#   SHIP_INTEGRATION_TIMEOUT (default 15m)  # bound for go test
#   SHIP_GO_TEST_EXTRA                      # extra args to go test (e.g. '-run=TestBlueSky')
#   SHIP_SKIP_PKG=1                         # skip build-rocm.sh (package) step
set -euo pipefail
root="$(cd "$(dirname "${0}")/.." && pwd)"

# Argument parsing — minimal; only --help and --list are recognized.
case "${1:-}" in
    --help|-h)
        cat <<EOF
ship-check.sh — Prismalama ship gate

Usage:
  ./scripts/ship-check.sh                 # integration + package
  SHIP_SKIP_PKG=1 ./scripts/ship-check.sh # integration only (fast; CPU OK)
  ./scripts/ship-check.sh --list          # print the steps without running
  ./scripts/ship-check.sh --help          # this message

Env:
  SHIP_INTEGRATION_TIMEOUT (default 15m)
  SHIP_GO_TEST_EXTRA       extra args to go test
  SHIP_SKIP_PKG=1          skip build-rocm.sh
EOF
        exit 0
        ;;
    --list)
        cat <<EOF
ship-check.sh — step list (no execution):

  1. cd \${REPO_ROOT}
  2. echo "== integration (CGO) =="
     CGO_ENABLED=1 go test -tags=integration ./integration \\
        -count=1 -timeout \${SHIP_INTEGRATION_TIMEOUT:-15m} \\
        \${SHIP_GO_TEST_EXTRA}
  3. [unless SHIP_SKIP_PKG=1]
     echo "== prismalama-ollama package =="
     exec \${REPO_ROOT}/build-rocm.sh

See docs/SHIP_CHECK.md for prerequisites and failure modes.
EOF
        exit 0
        ;;
    "")
        ;; # default: run normally
    *)
        echo "ship-check.sh: unknown argument: $1" >&2
        echo "  try: --list or --help" >&2
        exit 64 # EX_USAGE
        ;;
esac

cd "${root}"

: "${SHIP_INTEGRATION_TIMEOUT:=15m}"
: "${SHIP_GO_TEST_EXTRA:=}" # e.g. '-run=TestBlueSky|TestShipMemoryPolicyEnv' (quote; | is special in shell)

echo "== integration (CGO) =="
# shellcheck disable=SC2086
CGO_ENABLED=1 go test -tags=integration ./integration -count=1 -timeout "${SHIP_INTEGRATION_TIMEOUT}" ${SHIP_GO_TEST_EXTRA}

if [[ "${SHIP_SKIP_PKG:-}" == "1" ]]; then
	echo "ship-check: SHIP_SKIP_PKG=1 — skipping ./build-rocm.sh"
	exit 0
fi

echo "== prismalama-ollama package =="
exec "${root}/build-rocm.sh"