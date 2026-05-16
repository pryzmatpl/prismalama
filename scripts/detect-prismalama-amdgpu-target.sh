#!/usr/bin/env bash
# Print AMDGPU gfxNNNN for HIP (rocminfo), exit 1 if unavailable.
# Used by PKGBUILD when PRISMALAMA_AMDGPU_TARGETS is unset.

set -euo pipefail

if ! command -v rocminfo >/dev/null 2>&1; then
	exit 1
fi

# rocminfo separates agents with lines of asterisks; GPU agents include "Device Type: GPU".
# Requires GNU awk (match third argument) — Arch default awk is gawk.
rocminfo 2>/dev/null | awk '
BEGIN { RS="\n\\*{7}\n"; ok=0 }
/Device Type:[^\n]*GPU/ {
	if (match($0, /Name:[[:space:]]+(gfx[0-9a-z]+)/, a)) {
		print tolower(a[1])
		ok=1
		exit 0
	}
}
END { if (!ok) exit 1 }
'
