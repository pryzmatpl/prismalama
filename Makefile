.PHONY: help
help:
	@echo "Available targets:"
	@echo "  build                Build ollama binaries"
	@echo "  build-all            Build all variants"
	@echo "  build-rocm           Build ROCm variant"
	@echo "  build-pkg            Build Arch package"
	@echo "  build-pkg-clean      Clean and build Arch package"
	@echo "  clean                Remove build artifacts"
	@echo "  clean-pkg            Remove package artifacts"
	@echo "  update-subrepos      Sync upstream subrepos"
	@echo "  update-pkg           Update package metadata"
	@echo "  ship-check           Integration tests then build prismalama-ollama (see scripts/ship-check.sh)"
	@echo "  ship-check-fast      BlueSky integration only, no package"
	@echo "  docker-test-build    Build docker/test image (prismalama-test)"
	@echo "  docker-test          Run ship-check-fast inside container (see scripts/docker-test.sh)"
	@echo "  docker-test-integration  Full integration in Docker (SHIP_SKIP_PKG=1, no makepkg)"
	@echo "  docker-test-shell    Interactive shell in test image"
	@echo "  docker-gpu-build     Build docker/gpu image (AMD ROCm HIP + Vulkan GGML, prismalama-gpu)"
	@echo "  docker-gpu-run       Run prismalama-gpu with GPU devices (see docker/gpu/README.md)"
	@echo "  docker-arch-build    Build docker/arch from PKGBUILD (Arch base, matches pacman install; long build)"
	@echo "  docker-arch-prebuilt-build  Build docker/arch from docker/arch/prismalama.pkg.tar.zst (fast)"
	@echo "  docker-arch-run      Run prismalama-arch with AMD GPU devices (see docker/arch/README.md)"
	@echo
	@echo "Examples:"
	@echo "  make build"
	@echo "  make build-pkg"
	@echo "  make update-subrepos"
	@echo "  make docker-gpu-build   # AMD GPU image (long build)"
	@echo "  make docker-arch-build  # Arch PKGBUILD image (matches pacman install; long build)"

.PHONY: build
build:
	./build.sh

.PHONY: build-all
build-all:
	./build-all.sh

.PHONY: build-rocm
build-rocm:
	./build-rocm.sh

.PHONY: build-pkg
build-pkg:
	./build-pkg.sh

.PHONY: build-pkg-clean
build-pkg-clean:
	$(MAKE) clean-pkg
	./build-pkg.sh

.PHONY: clean
clean:
	rm -rf build build_ollama_airllm build_ollama_airllm_rocm

.PHONY: clean-pkg
clean-pkg:
	rm -rf build build_ollama_airllm build_ollama_airllm_rocm *.pkg.tar.zst

.PHONY: update-subrepos
update-subrepos:
	$(MAKE) -f Makefile.sync sync

.PHONY: update-pkg
update-pkg:
	./pkg-update

.PHONY: ship-check
ship-check:
	./scripts/ship-check.sh

.PHONY: ship-check-fast
ship-check-fast:
	SHIP_GO_TEST_EXTRA='-run=TestBlueSky|TestShipMemoryPolicyEnv|TestShipAdaptiveMemoryEnv|TestShipGpuOverheadDefault|TestShipVulkanMmapDefault|TestShipEngineDispatchOptOut|TestShipEngineDispatchMultipartAirLLM|TestShipEngineKindString|TestShipLayerStreamingEnvDefault|TestShipLayerStreamingEnvEnable|TestShipStreamingBudgetDefault|TestShipStreamingBudgetOverride|TestShipStreamingLayerMapGGUF|TestShipStreamingBackendInterface|TestShipStreamingInferenceStreamerLifecycle|TestShipStreamingComputeBackendInterface' SHIP_INTEGRATION_TIMEOUT=5m SHIP_SKIP_PKG=1 ./scripts/ship-check.sh

.PHONY: docker-test-build
docker-test-build:
	docker build -f docker/test/Dockerfile -t prismalama-test .

.PHONY: docker-test
docker-test:
	./scripts/docker-test.sh

.PHONY: docker-test-integration
docker-test-integration: docker-test-build
	docker run --rm -v "$$(pwd):/workspace:rw" -w /workspace \
		-e CGO_ENABLED=1 -e OLLAMA_BIN=/usr/bin/ollama \
		-e OLLAMA_LIBRARY_PATH=/usr/lib/ollama -e LD_LIBRARY_PATH=/usr/lib/ollama \
		-e SHIP_SKIP_PKG=1 \
		prismalama-test make ship-check

.PHONY: docker-test-shell
docker-test-shell: docker-test-build
	docker run --rm -it -v "$$(pwd):/workspace:rw" -w /workspace \
		-e CGO_ENABLED=1 -e OLLAMA_BIN=/usr/bin/ollama \
		-e OLLAMA_LIBRARY_PATH=/usr/lib/ollama -e LD_LIBRARY_PATH=/usr/lib/ollama \
		prismalama-test bash -l

.PHONY: docker-gpu-build
docker-gpu-build:
	docker build -f docker/gpu/Dockerfile -t prismalama-gpu .

.PHONY: docker-gpu-run
docker-gpu-run:
	docker run --rm -p 11434:11434 \
		--device /dev/kfd --device /dev/dri \
		--group-add video --group-add render \
		-e HIP_VISIBLE_DEVICES=0 \
		prismalama-gpu

.PHONY: docker-arch-build
docker-arch-build:
	docker build -f docker/arch/Dockerfile -t prismalama-arch .

.PHONY: docker-arch-prebuilt-build
docker-arch-prebuilt-build:
	@test -f docker/arch/prismalama.pkg.tar.zst || (echo "Missing docker/arch/prismalama.pkg.tar.zst — run: cp prismalama-ollama-*.pkg.tar.zst docker/arch/prismalama.pkg.tar.zst" >&2; exit 1)
	docker build -f docker/arch/Dockerfile.prebuilt -t prismalama-arch docker/arch

.PHONY: docker-arch-run
docker-arch-run:
	docker run --rm -p 11434:11434 \
		--device /dev/kfd --device /dev/dri \
		--group-add video --group-add render \
		-e HIP_VISIBLE_DEVICES=0 \
		prismalama-arch
