# Prismalama Arch package image

Docker images that install the same **`prismalama-ollama`** pacman package as a native Arch host: files under `/usr/bin/ollama`, `/usr/lib/ollama/rocm`, `/usr/share/ollama/` (AirLLM assets), and the same defaults style as `PKGBUILD`.

## Which Dockerfile

| File | Use when |
|------|----------|
| **`Dockerfile`** | You want a **full build inside Docker** (ROCm HIP + Vulkan GGML + Go). Slow; no local `makepkg` required. |
| **`Dockerfile.prebuilt`** | You already ran **`./build-rocm.sh`** / **`makepkg -sf`** on Arch and want a **lean, fast** image that only installs the `.pkg.tar.zst`. |

## Full build (from repository root)

```bash
docker build -f docker/arch/Dockerfile -t prismalama-arch .
# Optional: match GPU ISA (default gfx1100)
docker build -f docker/arch/Dockerfile --build-arg PRISMALAMA_AMDGPU_TARGETS=gfx1030 -t prismalama-arch .
```

## Prebuilt package (recommended for CI / fast iteration)

```bash
makepkg -sf   # or ./build-rocm.sh
cp prismalama-ollama-*.pkg.tar.zst docker/arch/prismalama.pkg.tar.zst
docker build -f docker/arch/Dockerfile.prebuilt -t prismalama-arch docker/arch
```

## Run (AMD GPU)

Same device expectations as `docker/gpu`: pass the ROCm devices and groups.

```bash
docker run --rm -p 11434:11434 \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  -e HIP_VISIBLE_DEVICES=0 \
  -v ollama-models:/var/lib/ollama \
  prismalama-arch
```

The container listens on **`0.0.0.0:11434`** and stores models under **`/var/lib/ollama`** (overrides the PKGBUILD default `/nvme3/models` for portability). Override with `-e OLLAMA_MODELS=...` if needed.

## AirLLM (optional)

The package ships AirLLM files under `/usr/share/ollama/`. PyTorch + `transformers` are **not** installed in these images (same as default Arch: GGML-first). To use AirLLM in a container, extend the image with `python-pytorch-rocm` and your chosen HF deps, or install on the host and use native `systemctl` instead.

## Kubernetes

Reuse the same patterns as **`docker/gpu/k8s/example-deployment.yaml`** (devices, env, PVC for `OLLAMA_MODELS`); swap the image name to your `prismalama-arch` tag.

## Related

- **`README-PKGBUILD.md`** — native Arch install and optdepends.
- **`docker/gpu/README.md`** — Ubuntu-based GPU image (source build, no pacman).
