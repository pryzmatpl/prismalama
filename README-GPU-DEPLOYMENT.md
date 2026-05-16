# GPU deployment notes (legacy helper scripts)

This document describes the **top-level helper scripts** in this repository. For maintained container
flows, prefer:

- `docker/gpu/README.md` (Prismalama GPU image: ROCm/HIP/Vulkan)
- `docker/arch/README.md` (Arch package image)

## Scripts in this directory

- `build-gpu.sh`
- `start-prismalama-gpu.sh`
- `deploy-gpu-stack.sh`
- `setup-jaisiu-integration.sh`
- `benchmark-gpu.py`
- `docker-compose.gpu.yml`

## Important limitations

- These scripts are convenience utilities, not CI-validated release installers.
- Several script paths target NVIDIA + Docker assumptions and may use upstream `ollama/ollama` images.
- Jaisiu/OpenClaw integration snippets are environment-specific and can require manual edits.
- Throughput depends on model, quantization, request shape, batching, and driver/runtime versions.

## Minimal safety checks before use

```bash
cd /path/to/prismalama
test -x ./build-gpu.sh
test -x ./start-prismalama-gpu.sh
docker --version
```

Then validate runtime behavior (engine and env) on the running server:

```bash
cd /path/to/prismalama/scripts
./verify-prismalama-runtime.sh
```

## Performance expectations

Do not assume fixed numbers (for example "221 TPS"). Benchmark your own stack:

```bash
cd /path/to/prismalama
python3 benchmark-gpu.py
```
