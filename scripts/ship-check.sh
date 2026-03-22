#!/usr/bin/env bash
# Ship gate: integration tests, then prismalama-ollama package (see docs/DEVELOPER.md § Ship gate).
set -euo pipefail
root="$(cd "$(dirname "${0}")/.." && pwd)"
cd "${root}"

: "${SHIP_INTEGRATION_TIMEOUT:=15m}"
: "${SHIP_GO_TEST_EXTRA:=}" # e.g. -run TestBlueSky for a fast path

echo "== integration (CGO) =="
# shellcheck disable=SC2086
CGO_ENABLED=1 go test -tags=integration ./integration -count=1 -timeout "${SHIP_INTEGRATION_TIMEOUT}" ${SHIP_GO_TEST_EXTRA}

if [[ "${SHIP_SKIP_PKG:-}" == "1" ]]; then
	echo "ship-check: SHIP_SKIP_PKG=1 — skipping ./build-rocm.sh"
	exit 0
fi

echo "== prismalama-ollama package =="
exec "${root}/build-rocm.sh"
