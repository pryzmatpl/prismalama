# Maintainer: Prismalama — build THIS tree (prismallama.cpp + GGML + AirLLM hooks), not upstream release tarballs.
# Build from repository root:  make -f Makefile.sync sync   # optional: refresh vendored llama.cpp/ggml
#                               makepkg -sf
#
# GPU stack selection (controls pacman deps + CMake backends):
#   PRISMALAMA_BACKENDS=amd     — ROCm HIP + Vulkan (installs rocm-hip-sdk)
#   PRISMALAMA_BACKENDS=nvidia  — CUDA + Vulkan (installs cuda; no ROCm)
#   PRISMALAMA_BACKENDS=all     — HIP + CUDA + Vulkan (both stacks)
#   PRISMALAMA_BACKENDS=minimal — CPU + Vulkan only (no ROCm / no CUDA toolkit required)
#   PRISMALAMA_BACKENDS=auto    — default: infer from pacman -Q (rocm-hip-sdk / cuda), else lspci PCI vendor, else minimal
#
# AMDGPU ISA: unset PRISMALAMA_AMDGPU_TARGETS on the target PC so PKGBUILD runs scripts/detect-prismalama-amdgpu-target.sh (rocminfo).
# Override: PRISMALAMA_AMDGPU_TARGETS=gfx1030 makepkg -sf   Disable auto: PRISMALAMA_AMDGPU_AUTO=0
#
# epoch: pkgver here is the Prismalama snapshot (see README-PKGBUILD.md § Versioning). Older installs may have used
# pkgver aligned with upstream Ollama (e.g. 0.18.x). Without epoch, pacman incorrectly reports a downgrade when
# going from 0.18.* to 0.4.*. epoch=1 makes current packages sort after those legacy builds.

pkgname=prismalama-ollama
epoch=1
pkgver=0.4.1
pkgrel=14
pkgdesc="Prismalama: Ollama-compatible server (GGML: optional ROCm HIP / CUDA + Vulkan; optional AirLLM)"
arch=('x86_64')
url="https://github.com/piotroxp/prismallama.cpp"
license=('MIT')
install=prismalama-ollama.install

# --- Backend profile (used by depends/makedepends and build/package) ---
_prism_back="${PRISMALAMA_BACKENDS:-auto}"
if [[ "${_prism_back}" == auto ]]; then
	if pacman -Q rocm-hip-sdk &>/dev/null && pacman -Q cuda &>/dev/null; then
		_prism_back=all
	elif pacman -Q rocm-hip-sdk &>/dev/null; then
		_prism_back=amd
	elif pacman -Q cuda &>/dev/null; then
		_prism_back=nvidia
	else
		_prism_back="${PRISMALAMA_BACKENDS_DEFAULT:-minimal}"
		if command -v lspci >/dev/null 2>&1; then
			_lp="$(lspci -nn 2>/dev/null || true)"
			if echo "${_lp}" | grep -qiE '(vga|3d|display).*\[1002:'; then
				_prism_back=amd
			elif echo "${_lp}" | grep -qiE '(vga|3d|display).*\[10de:'; then
				_prism_back=nvidia
			elif echo "${_lp}" | grep -qiE '(vga|3d|display).*\[8086:'; then
				_prism_back=minimal
			fi
		fi
	fi
fi
case "${_prism_back}" in
amd | rocm) _prism_back=amd ;;
nvidia | cuda) _prism_back=nvidia ;;
minimal | vulkan | cpu) _prism_back=minimal ;;
all) ;;
*)
	warning "Prismalama: unknown PRISMALAMA_BACKENDS='${PRISMALAMA_BACKENDS}', using minimal"
	_prism_back=minimal
	;;
esac

depends=('glibc' 'gcc-libs' 'zlib' 'vulkan-icd-loader')
makedepends=('go' 'cmake' 'ninja' 'gcc' 'vulkan-headers' 'glslang')

case "${_prism_back}" in
amd)
	makedepends+=('rocm-hip-sdk')
	depends+=('rocm-hip-runtime')
	;;
nvidia)
	makedepends+=('cuda')
	depends+=('cuda')
	;;
all)
	makedepends+=('rocm-hip-sdk' 'cuda')
	depends+=('rocm-hip-runtime' 'cuda')
	;;
minimal) ;;
esac

optdepends=(
	'python-pytorch-rocm: AirLLM runner (weight streaming, multi-part GGUF / HF layouts)'
	'python-transformers: AirLLM (often AUR; else: pip install transformers safetensors — see README-PKGBUILD.md)'
	'python-safetensors: AirLLM weights (often AUR; pip alternative — see README-PKGBUILD.md)'
)

provides=('ollama')
conflicts=('ollama' 'ollama-rocm' 'ollama-cuda' 'ollama-airllm-rocm')

options=(!strip !debug)

# Default model store (edit before build if needed)
_model_dir="/nvme3/models"

build() {
	cd "${startdir}"

	warning "Prismalama: backend profile=${_prism_back} (set PRISMALAMA_BACKENDS to amd|nvidia|all|minimal|auto)"

	local _LLAMA_HIPBLAS=OFF
	local _AMDGPU_TARGETS=""
	if [[ "${_prism_back}" == amd || "${_prism_back}" == all ]]; then
		_LLAMA_HIPBLAS=ON
		local _detected=""
		if [[ -n "${PRISMALAMA_AMDGPU_TARGETS:-}" ]]; then
			_AMDGPU_TARGETS="${PRISMALAMA_AMDGPU_TARGETS}"
			warning "Prismalama: AMDGPU_TARGETS=${_AMDGPU_TARGETS} (PRISMALAMA_AMDGPU_TARGETS)"
		elif [[ "${PRISMALAMA_AMDGPU_AUTO:-1}" != "0" ]] && _detected="$(bash "${startdir}/scripts/detect-prismalama-amdgpu-target.sh" 2>/dev/null)" && [[ -n "${_detected}" ]]; then
			_AMDGPU_TARGETS="${_detected}"
			warning "Prismalama: AMDGPU_TARGETS=${_AMDGPU_TARGETS} (auto-detected). Override: PRISMALAMA_AMDGPU_TARGETS=gfx…  Disable: PRISMALAMA_AMDGPU_AUTO=0"
		else
			_AMDGPU_TARGETS="gfx1100"
			warning "Prismalama: AMDGPU_TARGETS=${_AMDGPU_TARGETS} (fallback; no rocminfo GPU). Install/use ROCm on the build host or set PRISMALAMA_AMDGPU_TARGETS — see README-PKGBUILD.md"
		fi
	fi

	local _LLAMA_CUDA=OFF
	local -a _cmake_cuda=()
	if [[ "${_prism_back}" == nvidia || "${_prism_back}" == all ]]; then
		if [[ "${PRISMALAMA_CUDA_AUTO:-1}" != "0" ]] && command -v nvcc >/dev/null 2>&1; then
			_LLAMA_CUDA=ON
			_cmake_cuda+=("-DCMAKE_CUDA_ARCHITECTURES=${PRISMALAMA_CUDA_ARCHITECTURES:-native}")
			warning "Prismalama: LLAMA_CUDA=ON (nvcc=$(command -v nvcc)). Arch: PRISMALAMA_CUDA_ARCHITECTURES (default native). Disable: PRISMALAMA_CUDA_AUTO=0"
		elif [[ "${PRISMALAMA_CUDA_AUTO:-1}" != "0" ]]; then
			warning "Prismalama: LLAMA_CUDA skipped — nvcc not found (install cuda). Silence: PRISMALAMA_CUDA_AUTO=0"
		fi
	fi

	local -a _cmake_hip=("-DLLAMA_HIPBLAS=${_LLAMA_HIPBLAS}")
	if [[ "${_LLAMA_HIPBLAS}" == ON ]]; then
		_cmake_hip+=("-DAMDGPU_TARGETS=${_AMDGPU_TARGETS}")
	fi

	cmake -B build -G Ninja \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_INSTALL_PREFIX=/usr \
		"${_cmake_hip[@]}" \
		-DLLAMA_CUDA="${_LLAMA_CUDA}" \
		"${_cmake_cuda[@]}" \
		-DOLLAMA_RUNNER_DIR=rocm

	cmake --build build --parallel "$(nproc)" --target ggml

	if [[ "${_LLAMA_HIPBLAS}" == ON ]]; then
		cmake --build build --parallel "$(nproc)" --target ggml-hip
	fi

	if [[ "${_LLAMA_CUDA}" == ON ]]; then
		if [[ -f build/CMakeCache.txt ]] && grep -q '^CMAKE_CUDA_COMPILER:FILEPATH=.\+/nvcc' build/CMakeCache.txt; then
			cmake --build build --parallel "$(nproc)" --target ggml-cuda
			echo 1 > "${startdir}/.prismalama_cuda"
		else
			warning "Prismalama: LLAMA_CUDA requested but CMake did not pick a CUDA compiler — skipping ggml-cuda (check CUDAToolkit / nvcc)"
			echo 0 > "${startdir}/.prismalama_cuda"
		fi
	else
		echo 0 > "${startdir}/.prismalama_cuda"
	fi

	cmake --build build --parallel "$(nproc)" --target ggml-vulkan

	export CGO_ENABLED=1
	export CGO_CFLAGS="${CGO_CFLAGS:-}"
	export CGO_CXXFLAGS="${CGO_CXXFLAGS:-}"
	export GOFLAGS="-buildmode=pie -trimpath"
	_xver="${pkgver}-r${pkgrel}-prismalama"
	go build -o prismalama-ollama \
		-ldflags "-w -s -X=github.com/ollama/ollama/version.Version=${_xver}" \
		.
}

package() {
	cd "${startdir}"

	DESTDIR="${pkgdir}" cmake --install build --prefix /usr --component CPU
	if [[ "${_prism_back}" == amd || "${_prism_back}" == all ]]; then
		DESTDIR="${pkgdir}" cmake --install build --prefix /usr --component HIP
	fi
	if [[ -f "${startdir}/.prismalama_cuda" ]] && [[ "$(cat "${startdir}/.prismalama_cuda")" == "1" ]]; then
		DESTDIR="${pkgdir}" cmake --install build --prefix /usr --component CUDA
	fi
	DESTDIR="${pkgdir}" cmake --install build --prefix /usr --component Vulkan

	install -Dm755 prismalama-ollama "${pkgdir}/usr/bin/ollama"

	install -Dm644 LICENSE "${pkgdir}/usr/share/licenses/${pkgname}/LICENSE"

	install -Dm755 runner/airllmrunner/airllm_runner.py "${pkgdir}/usr/share/ollama/airllm_runner.py"
	if [[ -d src/airllm/air_llm ]]; then
		cp -a src/airllm/air_llm "${pkgdir}/usr/share/ollama/airllm"
	fi

	install -dm755 "${pkgdir}/usr/lib/sysusers.d"
	printf '%s\n' 'u ollama - "Ollama service user" -' > "${pkgdir}/usr/lib/sysusers.d/ollama.conf"

	install -dm755 "${pkgdir}/usr/lib/systemd/system"
	cat > "${pkgdir}/usr/lib/systemd/system/ollama.service" << 'EOF'
[Unit]
Description=Prismalama (Ollama-compatible) — ROCm / CUDA / Vulkan GGML + AirLLM
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
	local _extra_env=""
	case "${_prism_back}" in
	amd | all)
		_extra_env="
# AMD (ROCm)
HIP_VISIBLE_DEVICES=0
HSA_OVERRIDE_GFX_VERSION=11.0.0
"
		;;
	nvidia)
		_extra_env="
# NVIDIA (CUDA GGML when built with nvcc)
# CUDA_VISIBLE_DEVICES=0
"
		;;
	esac

	cat > "${pkgdir}/etc/default/ollama" << EOF
# Prismalama — profile ${_prism_back} (from PRISMALAMA_BACKENDS at package build time)
OLLAMA_MODELS=${_model_dir}
OLLAMA_HOST=127.0.0.1:11434
OLLAMA_NUM_PARALLEL=1
OLLAMA_KEEP_ALIVE=5m
OLLAMA_LIBRARY_PATH=/usr/lib/ollama/rocm
OLLAMA_LAYER_STREAMING=1
${_extra_env}
OLLAMA_USE_AIRLLM=0
AIRLLM_COMPRESSION=4bit
AIRLLM_DEVICE=cuda:0
EOF
}
