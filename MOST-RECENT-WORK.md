---
> **→ see [NEXT.md](./NEXT.md) for the current phase (Phase 0 / JAISIU-2156) and in-flight tickets. This file remains the BC4 + GPU-detection history.**

## Goal
- Get BC4 weight-as-image inference working with GPU-enabled builds and verify GPU detection/benchmarks.

## Constraints & Preferences
- BC4 compression (8x ratio, hardware-decoded on AMD/Intel GPUs).
- Must compile with Go 1.24+.
- Avoid committing build artifacts (binaries).

## Progress
### Done
- BC4 weightimage implementation, Vulkan shader, and ggml-vulkan pipeline integration; unit tests pass (8/8).
- Docker GPU image build fixed and committed: `26e039c0 fix(docker/gpu): enable GGML build flags and build all targets`.
- Docker GPU image (`prismalama-gpu`) builds HIP + Vulkan GGML libraries successfully.
- Extracted built GGML libs from container: `build/lib/ollama/rocm/` contains `libggml-hip.so`, `libggml-vulkan.so`, `libggml-cpu-x64.so`, ROCm runtime libs.
- Rebuilt host binary with Go 1.24.2 and HIP flags; binaries used for testing.
- CPU inference benchmarks recorded (no GPU):
  - `phi3:mini` ~9 TPS on host; ~6-9 TPS in Docker.
  - `qwen3:0.6b` ~14–16 TPS CPU.
- Commits already pushed to `origin/main` for BC4 work and docker fix.

### In Progress
- GPU detection still failing at runtime (`total_vram="0 B"` in logs) for both host and Docker.

### Blocked
- GPU detection failure in Ollama runner; HIP runtime sees GPU (`rocminfo` shows `gfx1100`), but Ollama runner discovery returns zero VRAM.

## Key Decisions
- For Docker GPU build, enable full GGML build flags and build all targets rather than specific `ggml-cpu/hip/vulkan` targets (fixes missing target error).

## Next Steps
- Diagnose Ollama runner GPU discovery path (why `total_vram="0 B"` despite `rocminfo` seeing GPU).
- Validate amdgpu/ROCm device access assumptions for Ollama runner in container and host.
- Optional: add explicit logging around `ml.GetDevicesFromRunner` / HIP discovery if needed.

## Critical Context
- GPU detection log consistently shows: `total_vram="0 B"` and `discovering available GPUs...`.
- Docker GPU image built via `docker/gpu/Dockerfile` now uses:
  - `-DGGML_BUILD=ON -DGGML_SHARED=ON -DGGML_BACKEND_DL=ON -DGGML_BACKEND_SHARED=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_CUDA=OFF -DLLAMA_HIPBLAS=ON`
  - `cmake --build build --parallel "$(nproc)" && cmake --install build --prefix /usr`
- Docker container has GPU devices accessible (`/dev/kfd`, `/dev/dri/*`), `rocminfo` shows `gfx1100` in container, but Ollama still reports 0 VRAM.
- Untracked build artifacts present at times (`src/bin/ollama-gpu`), not committed.

## Relevant Files
- `ml/weightimage/weightimage.go`: weight-to-image conversion.
- `ml/weightimage/compression.go`: BC4/DCT compression.
- `ml/backend/ggml/ggml/src/ggml-vulkan/vulkan-shaders/bc4_decompress.comp`: BC4 shader.
- `ml/backend/ggml/ggml/src/ggml-vulkan/ggml-vulkan.cpp`: bc4 pipeline integration.
- `docker/gpu/Dockerfile`: GPU image build; updated GGML build flags and build-all approach.
- `build/lib/ollama/rocm/`: extracted HIP/Vulkan ggml shared libs for runtime discovery.
- `llm/server.go` and `discover/runner.go`: GPU discovery path and runner spawn logic.
---
