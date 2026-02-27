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
	@echo
	@echo "Examples:"
	@echo "  make build"
	@echo "  make build-pkg"
	@echo "  make update-subrepos"

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
