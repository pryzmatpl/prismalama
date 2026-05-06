#!/usr/bin/env bash
# Build Arch package from this repository (Prismalama — not upstream release binaries).
set -euo pipefail
cd "$(dirname "$0")"
export PRISMALAMA_BACKENDS="${PRISMALAMA_BACKENDS:-amd}"
echo "PRISMALAMA_BACKENDS=${PRISMALAMA_BACKENDS} (default amd for this script)"
# Leave PRISMALAMA_AMDGPU_TARGETS unset so PKGBUILD can auto-pick gfx from rocminfo when HIP is enabled.
if [[ -n "${PRISMALAMA_AMDGPU_TARGETS:-}" ]]; then
	echo "PRISMALAMA_AMDGPU_TARGETS=${PRISMALAMA_AMDGPU_TARGETS} (explicit)"
else
	echo "PRISMALAMA_AMDGPU_TARGETS unset — PKGBUILD uses scripts/detect-prismalama-amdgpu-target.sh when rocminfo works"
fi
echo "Optional: make -f Makefile.sync sync"
exec makepkg -sf "$@"
