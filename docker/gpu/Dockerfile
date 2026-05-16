# Prismalama GPU image: GGML CPU + HIP (ROCm) + Vulkan, Go ollama — suitable for bare metal or Kubernetes with AMD GPU.
# NVIDIA/CUDA is a different stack; extend with nvidia/cuda base if needed.
#
# Build:
#   docker build -f docker/gpu/Dockerfile -t prismalama-gpu .
#   docker build -f docker/gpu/Dockerfile --build-arg AMDGPU_TARGETS=gfx1100 -t prismalama-gpu .
#
# Run (Docker):
#   docker run --rm --device /dev/kfd --device /dev/dri -e HIP_VISIBLE_DEVICES=0 -p 11434:11434 prismalama-gpu
#
# See docker/gpu/README.md for Kubernetes (device plugin, resources, securityContext).

# Noble (24.04) ships the `glslc` package (Google shaderc CLI); Jammy has neither — required for ggml-vulkan CMake.
ARG ROCM_BASE=rocm/dev-ubuntu-24.04:7.2-complete
FROM ${ROCM_BASE}

ENV DEBIAN_FRONTEND=noninteractive
ENV CGO_ENABLED=1
ENV ROCM_PATH=/opt/rocm
ENV HIP_PATH=/opt/rocm
ENV PATH=/opt/rocm/bin:/usr/local/go/bin:${PATH}

# glslc for ggml-vulkan shader compile; Vulkan ICD/runtime; curl for Go bootstrap
RUN apt-get update && apt-get install -y --no-install-recommends \
	ca-certificates \
	curl \
	cmake \
	ninja-build \
	pkg-config \
	build-essential \
	git \
	make \
	glslc \
	libvulkan1 \
	libvulkan-dev \
	mesa-vulkan-drivers \
	vulkan-tools \
	&& rm -rf /var/lib/apt/lists/*

# Go (match go.mod)
ARG GO_VERSION=1.24.1
RUN curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" | tar -C /usr/local -xz

# Default ISA: RX 7900 class; override at build time for data-center SKUs
ARG AMDGPU_TARGETS=gfx1100
ENV AMDGPU_TARGETS=${AMDGPU_TARGETS}

WORKDIR /src

COPY go.mod go.sum ./
RUN go mod download

COPY . .

# GGML: CPU + HIP + Vulkan into lib/ollama/rocm (matches Ollama packaging / PKGBUILD)
RUN cmake -B build -G Ninja \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_INSTALL_PREFIX=/usr \
		-DGGML_BUILD=ON \
		-DGGML_SHARED=ON \
		-DGGML_BACKEND_DL=ON \
		-DGGML_BACKEND_SHARED=ON \
		-DGGML_CPU_ALL_VARIANTS=ON \
		-DLLAMA_HIPBLAS=ON \
		-DLLAMA_CUDA=OFF \
		-DGGML_CUDA=OFF \
		-DAMDGPU_TARGETS="${AMDGPU_TARGETS}" \
		-DOLLAMA_RUNNER_DIR=rocm \
	&& cmake --build build --parallel "$(nproc)" \
	&& cmake --install build --prefix /usr 2>&1 | tail -30

RUN go build -trimpath -buildmode=pie -o /usr/bin/ollama \
		-ldflags "-w -s -X=github.com/ollama/ollama/version.Version=prismalama-gpu" \
		.

WORKDIR /workspace

RUN chmod +x /src/docker/gpu/entrypoint.sh

ENV OLLAMA_LIBRARY_PATH=/usr/lib/ollama/rocm
ENV LD_LIBRARY_PATH=/usr/lib/ollama/rocm:/opt/rocm/lib
ENV OLLAMA_HOST=0.0.0.0:11434

EXPOSE 11434

ENTRYPOINT ["/src/docker/gpu/entrypoint.sh"]
CMD ["serve"]
