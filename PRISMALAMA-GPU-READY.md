# Prismalama GPU docs status

This file replaces an older "deployment complete / target met" report that made hard
performance claims. Treat GPU throughput as **workload + model + driver + backend dependent**.

## What is actually in this repository

- `docker/gpu/` contains the maintained Prismalama GPU container path (AMD ROCm + HIP + Vulkan).
- `docker/arch/` contains the Arch package container path (`prismalama-ollama` package inside image).
- Top-level helper scripts (`build-gpu.sh`, `start-prismalama-gpu.sh`, `deploy-gpu-stack.sh`) exist,
  but are convenience scripts and are not the source of truth for production behavior.
- `docker-compose.gpu.yml` exists for local orchestration experiments.

## Source-of-truth docs

- `docker/gpu/README.md`
- `docker/arch/README.md`
- `docs/RUNTIME_DISPATCH.md`
- `README-PKGBUILD.md`

## Promise boundary

Prismalama does **not** guarantee fixed TPS targets in repository docs. Any number should be treated
as a local benchmark example, not a contractual performance result.
