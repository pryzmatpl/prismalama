#!/usr/bin/env bash
# Build docker/test image and run a command with the repo mounted at /workspace.
# CPU GGML image — no makepkg/ROCm; use host ./build-rocm.sh for Arch packages.
# Usage:
#   ./scripts/docker-test.sh                         # make ship-check-fast
#   PRISMALAMA_DOCKER_GPU=1 ./scripts/docker-test.sh # optional ROCm devices (host must match)
#   ./scripts/docker-test.sh make ship-check         # needs SHIP_SKIP_PKG=1 (see Makefile docker-test-integration)
#   ./scripts/docker-test.sh bash -l                 # shell (-it added automatically for tty)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${PRISMALAMA_TEST_IMAGE:-prismalama-test}"
DOCKER_OPTS=(
	--rm
	-v "${ROOT}:/workspace:rw"
	-w /workspace
	-e CGO_ENABLED=1
	-e OLLAMA_BIN=/usr/bin/ollama
	-e OLLAMA_LIBRARY_PATH=/usr/lib/ollama
	-e "LD_LIBRARY_PATH=/usr/lib/ollama"
)

if [[ "${PRISMALAMA_DOCKER_GPU:-}" == "1" ]]; then
	# ROCm (AMD): requires matching host drivers; optional
	DOCKER_OPTS+=(--device /dev/kfd --device /dev/dri --group-add video)
fi

docker build -f "${ROOT}/docker/test/Dockerfile" -t "${IMAGE}" "${ROOT}"

CMD=(make ship-check-fast)
if [[ $# -gt 0 ]]; then
	CMD=("$@")
fi

IT=()
if [[ "${CMD[0]}" == "bash" || "${CMD[0]}" == "sh" ]]; then
	IT=( -it )
fi

exec docker run "${IT[@]}" "${DOCKER_OPTS[@]}" "${IMAGE}" "${CMD[@]}"
