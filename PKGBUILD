# Maintainer: Prismalama — build THIS tree (prismallama.cpp + GGML + AirLLM hooks), not upstream release tarballs.
# Build from repository root:  make -f Makefile.sync sync   # optional: refresh vendored llama.cpp/ggml
#                               makepkg -sf
# Override GPU ISA (faster link): PRISMALAMA_AMDGPU_TARGETS=gfx1030 makepkg -sf
#
# epoch: pkgver here is the Prismalama snapshot (see README-PKGBUILD.md § Versioning). Older installs may have used
# pkgver aligned with upstream Ollama (e.g. 0.18.x). Without epoch, pacman incorrectly reports a downgrade when
# going from 0.18.* to 0.4.*. epoch=1 makes current packages sort after those legacy builds.

pkgname=prismalama-ollama
epoch=1
pkgver=0.4.1
pkgrel=4
pkgdesc="Prismalama: Ollama-compatible server built from source (ROCm HIP + Vulkan GGML, AirLLM runner, large-model paths)"
arch=('x86_64')
url="https://github.com/piotroxp/prismallama.cpp"
license=('MIT')
install=prismalama-ollama.install

depends=(
	'glibc'
	'gcc-libs'
	'zlib'
	'rocm-hip-runtime'
	'vulkan-icd-loader'
)

makedepends=(
	'go'
	'cmake'
	'ninja'
	'gcc'
	'rocm-hip-sdk'
	'vulkan-headers'
)

optdepends=(
	'python-pytorch-rocm: AirLLM runner (weight streaming, multi-part GGUF / HF layouts)'
	'python-transformers: AirLLM tokenizer / model metadata'
	'python-safetensors: safetensors checkpoints for AirLLM'
)

provides=('ollama')
conflicts=('ollama' 'ollama-rocm' 'ollama-cuda' 'ollama-airllm-rocm')

options=(!strip !debug)

# Default model store (edit before build if needed)
_model_dir="/nvme3/models"
# gfx1100 = RX 7900 class; override when building for another AMD GPU
_AMDGPU_TARGETS="${PRISMALAMA_AMDGPU_TARGETS:-gfx1100}"

build() {
	cd "${startdir}"

	# GGML shared backends: CPU + HIP (ROCm) + Vulkan (heterogeneous / broad GPU path)
	cmake -B build -G Ninja \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_INSTALL_PREFIX=/usr \
		-DLLAMA_HIPBLAS=ON \
		-DLLAMA_CUDA=OFF \
		-DAMDGPU_TARGETS="${_AMDGPU_TARGETS}" \
		-DOLLAMA_RUNNER_DIR=rocm

	cmake --build build --parallel "$(nproc)" --target ggml-cpu ggml-hip

	_vulkan=0
	cmake --build build --parallel "$(nproc)" --target ggml-vulkan && _vulkan=1 || true
	echo "${_vulkan}" > "${startdir}/.prismalama_vulkan"

	export CGO_ENABLED=1
	export CGO_CFLAGS="${CGO_CFLAGS:-}"
	export CGO_CXXFLAGS="${CGO_CXXFLAGS:-}"
	export GOFLAGS="-buildmode=pie -trimpath"
	_xver="${pkgver}-r${pkgrel}-prismalama"
	# Do not use -o ollama: a directory named ollama/ may exist in the tree (install would fail).
	go build -o prismalama-ollama \
		-ldflags "-w -s -X=github.com/ollama/ollama/version.Version=${_xver}" \
		.
}

package() {
	cd "${startdir}"

	# CMake install() uses absolute DESTINATION /usr/...; --prefix "${pkgdir}/usr" does not remap those.
	# Use DESTDIR (Arch/cmake convention) so files land under "${pkgdir}/usr/...".
	DESTDIR="${pkgdir}" cmake --install build --prefix /usr --component CPU
	DESTDIR="${pkgdir}" cmake --install build --prefix /usr --component HIP
	if [[ -f "${startdir}/.prismalama_vulkan" ]] && [[ "$(cat "${startdir}/.prismalama_vulkan")" == "1" ]]; then
		DESTDIR="${pkgdir}" cmake --install build --prefix /usr --component Vulkan
	fi

	install -Dm755 prismalama-ollama "${pkgdir}/usr/bin/ollama"

	install -Dm644 LICENSE "${pkgdir}/usr/share/licenses/${pkgname}/LICENSE"

	# AirLLM subprocess (weight streaming, layer-wise execution on constrained VRAM)
	install -Dm755 runner/airllmrunner/airllm_runner.py "${pkgdir}/usr/share/ollama/airllm_runner.py"
	if [[ -d src/airllm/air_llm ]]; then
		cp -a src/airllm/air_llm "${pkgdir}/usr/share/ollama/airllm"
	fi

	install -dm755 "${pkgdir}/usr/lib/sysusers.d"
	printf '%s\n' 'u ollama - "Ollama service user" -' > "${pkgdir}/usr/lib/sysusers.d/ollama.conf"

	install -dm755 "${pkgdir}/usr/lib/systemd/system"
	cat > "${pkgdir}/usr/lib/systemd/system/ollama.service" << 'EOF'
[Unit]
Description=Prismalama (Ollama-compatible) — ROCm/Vulkan GGML + AirLLM
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=ollama
EnvironmentFile=-/etc/default/ollama
ExecStart=/usr/bin/ollama serve
Restart=on-failure
RestartSec=3
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/nvme3/models /var/lib/ollama /tmp
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

	install -dm755 "${pkgdir}/etc/default"
	cat > "${pkgdir}/etc/default/ollama" << EOF
# Prismalama — see docs/DEVELOPER.md
OLLAMA_MODELS=${_model_dir}
OLLAMA_HOST=127.0.0.1:11434
OLLAMA_NUM_PARALLEL=1
# GGML backends are installed under /usr/lib/ollama/rocm (see CMake OLLAMA_RUNNER_DIR)
OLLAMA_LIBRARY_PATH=/usr/lib/ollama/rocm

# AMD
HIP_VISIBLE_DEVICES=0
HSA_OVERRIDE_GFX_VERSION=11.0.0

# Prefer AirLLM when the runner detects streaming layouts (safetensors / multi-part GGUF / OLLAMA_USE_AIRLLM)
OLLAMA_USE_AIRLLM=1
AIRLLM_COMPRESSION=4bit
AIRLLM_DEVICE=cuda:0
EOF
}
