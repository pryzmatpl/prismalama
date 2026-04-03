# Prismalama GPU container (AMD ROCm + Vulkan + HIP)

![Prismalama Logo](../../logo.jpg)

Single image with **GGML CPU**, **HIP (ROCm)**, and **Vulkan** backends, plus the Prismalama `ollama` binary. Intended for **GPU inference** on AMD GPUs and **Kubernetes** workloads that need GPU isolation without installing Prismalama on the host.

## Build

```bash
docker build -f docker/gpu/Dockerfile -t prismalama-gpu .
# Narrow ISA for faster links (default: gfx1100):
docker build -f docker/gpu/Dockerfile --build-arg AMDGPU_TARGETS=gfx1100 -t prismalama-gpu .
#
# Different ROCm base (must include HIP toolchain + dev libs):
docker build -f docker/gpu/Dockerfile --build-arg ROCM_BASE=rocm/dev-ubuntu-24.04:7.2 -t prismalama-gpu .
```

## Run (Docker)

```bash
docker run --rm -p 11434:11434 \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  -e HIP_VISIBLE_DEVICES=0 \
  -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  prismalama-gpu
```

Point clients at `http://<host>:11434`. Set `OLLAMA_MODELS` to a mounted volume for persistent models.

## Kubernetes

- Install an **AMD GPU device plugin** (or equivalent) so nodes expose `amd.com/gpu` (name varies by plugin).
- Mount model storage (PVC, hostPath, or object store sidecar — your policy).
- Typical **GPU** request/limit: `amd.com/gpu: "1"` (confirm with your plugin).
- **Security**: GPU access often requires `privileged: false` but **device plugins** inject devices; some clusters require `privileged: true` or specific `capabilities` — follow your cluster’s GPU guide.
- **ROCm env**: pass `HIP_VISIBLE_DEVICES`, `HSA_OVERRIDE_GFX_VERSION` if your GPU needs a gfx version override.

Example: `docker/gpu/k8s/example-deployment.yaml`.

## CPU-only test image

For lightweight CI without a GPU, use `docker/test/Dockerfile` (CPU GGML only). **Do not** expect HIP/Vulkan GPU behavior from that image.

## Related

- `Makefile` targets: `docker-gpu-build`, `docker-gpu-run`
- `README-PKGBUILD.md` — native Arch package (same GGML layout under `/usr/lib/ollama/rocm`).
