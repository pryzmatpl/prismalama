# Prismalama isolated test image: GGML CPU (+ Vulkan when it builds), Go CGO ollama binary.
# ROCm/HIP is not included (use host GPU workflows or extend this image if needed).
# Build: docker build -f docker/test/Dockerfile -t prismalama-test .
# Run:   ./scripts/docker-test.sh   OR   make docker-test

FROM golang:1.24-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
	ca-certificates \
	cmake \
	ninja-build \
	g++ \
	pkg-config \
	make \
	git \
	&& rm -rf /var/lib/apt/lists/*
# Intentionally no Vulkan dev packages: CMake would require glslc/shader toolchain; test image is CPU GGML only.

ENV CGO_ENABLED=1
ENV OLLAMA_BIN=/usr/bin/ollama
ENV OLLAMA_LIBRARY_PATH=/usr/lib/ollama
ENV LD_LIBRARY_PATH=/usr/lib/ollama

WORKDIR /src

# Cache deps
COPY go.mod go.sum ./
RUN go mod download

COPY . .

# GGML: CPU only (no HIP / no Vulkan in this image)
RUN cmake -B build -G Ninja \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_INSTALL_PREFIX=/usr \
		-DLLAMA_HIPBLAS=OFF \
		-DLLAMA_CUDA=OFF \
	&& cmake --build build --parallel "$(nproc)" --target ggml-cpu \
	&& cmake --install build --prefix /usr --component CPU

# Prismalama binary (served to integration tests via OLLAMA_BIN)
RUN go build -trimpath -buildmode=pie -o /usr/bin/ollama \
		-ldflags "-w -s -X=github.com/ollama/ollama/version.Version=docker-test" \
		.

WORKDIR /workspace

RUN chmod +x /src/docker/test/entrypoint.sh

ENTRYPOINT ["/src/docker/test/entrypoint.sh"]
CMD ["make", "ship-check-fast"]
