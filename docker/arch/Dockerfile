# Prismalama Arch-native image: builds root PKGBUILD (same as `makepkg -sf` / `./build-rocm.sh`), installs the
# resulting package into a minimal runtime — layout matches bare-metal Arch installs.
#
# Build (long — compiles ROCm HIP + Vulkan GGML + Go; ~similar time to local makepkg):
#   docker build -f docker/arch/Dockerfile -t prismalama-arch .
#   docker build -f docker/arch/Dockerfile --build-arg PRISMALAMA_AMDGPU_TARGETS=gfx1030 -t prismalama-arch .
#
# Run (AMD GPU):
#   docker run --rm -p 11434:11434 --device /dev/kfd --device /dev/dri \
#     --group-add video --group-add render -e HIP_VISIBLE_DEVICES=0 prismalama-arch
#
# See docker/arch/README.md

# -----------------------------------------------------------------------------
FROM archlinux:latest AS builder

RUN pacman -Syu --noconfirm \
	&& pacman -S --needed --noconfirm \
		base-devel fakeroot git \
		go cmake ninja gcc zlib \
		rocm-hip-sdk rocm-hip-runtime \
		vulkan-headers vulkan-icd-loader shaderc \
	&& pacman -Scc --noconfirm || true

RUN useradd -m build
WORKDIR /home/build/prismalama

COPY --chown=build:build . .

ARG PRISMALAMA_AMDGPU_TARGETS=gfx1100
ENV PRISMALAMA_AMDGPU_TARGETS=${PRISMALAMA_AMDGPU_TARGETS}

USER build
RUN makepkg -f --noconfirm

# -----------------------------------------------------------------------------
FROM archlinux:latest

RUN pacman -Syu --noconfirm \
	&& pacman -S --needed --noconfirm systemd \
	&& pacman -Scc --noconfirm || true

COPY --from=builder /home/build/prismalama/prismalama-ollama-*.pkg.tar.zst /tmp/

RUN pacman -U --noconfirm /tmp/prismalama-ollama-*.pkg.tar.zst \
	&& rm -f /tmp/*.pkg.tar.zst

# post_install uses systemd-sysusers; ensure service user + model dir exist in minimal images
RUN getent passwd ollama >/dev/null 2>&1 || useradd -r -d /var/lib/ollama -s /usr/sbin/nologin ollama
RUN install -d -m 0755 -o ollama -g ollama /var/lib/ollama

COPY docker/arch/entrypoint.sh /usr/local/bin/prismalama-entrypoint.sh
RUN chmod +x /usr/local/bin/prismalama-entrypoint.sh

ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_MODELS=/var/lib/ollama
ENV OLLAMA_LIBRARY_PATH=/usr/lib/ollama/rocm
ENV LD_LIBRARY_PATH=/usr/lib/ollama/rocm:/opt/rocm/lib

EXPOSE 11434

WORKDIR /var/lib/ollama

ENTRYPOINT ["/usr/local/bin/prismalama-entrypoint.sh"]
CMD ["serve"]
