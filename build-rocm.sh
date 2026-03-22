#!/usr/bin/env bash
# Build Arch package from this repository (Prismalama — not upstream release binaries).
set -euo pipefail
cd "$(dirname "$0")"
export PRISMALAMA_AMDGPU_TARGETS="${PRISMALAMA_AMDGPU_TARGETS:-gfx1100}"
echo "PRISMALAMA_AMDGPU_TARGETS=${PRISMALAMA_AMDGPU_TARGETS}"
echo "Optional: make -f Makefile.sync sync"
exec makepkg -sf "$@"
