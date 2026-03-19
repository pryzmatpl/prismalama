# Maintainer: Prismalama Ollama Package
pkgname=prismalama-ollama
pkgver=0.18.2
pkgrel=1
pkgdesc="Ollama for large models on AMD GPUs with ROCm"
arch=('x86_64')
url="https://github.com/ollama/ollama"
license=('MIT')

depends=(
    'glibc'
    'zlib'
    'gcc-libs'
    'rocm-hip-sdk'
)

provides=('ollama')
conflicts=('ollama' 'ollama-rocm' 'ollama-cuda' 'ollama-airllm-rocm')
options=(!strip !debug)

_model_dir="/nvme3/models"

prepare() {
    cd "${srcdir}"
    if [ ! -f ollama-linux-amd64.tar.zst ]; then
        curl -fSLO https://github.com/ollama/ollama/releases/download/v${pkgver}/ollama-linux-amd64.tar.zst
    fi
    if [ ! -f ollama-linux-amd64-rocm.tar.zst ]; then
        curl -fSLO https://github.com/ollama/ollama/releases/download/v${pkgver}/ollama-linux-amd64-rocm.tar.zst
    fi
    rm -rf bin lib
    tar xf ollama-linux-amd64.tar.zst
    tar xf ollama-linux-amd64-rocm.tar.zst
}

package() {
    install -dm755 "${pkgdir}/usr/bin"
    install -dm755 "${pkgdir}/usr/lib/ollama"
    install -dm755 "${pkgdir}/usr/lib/systemd/system"
    install -dm755 "${pkgdir}/etc/default"

    install -Dm755 "${srcdir}/bin/ollama" "${pkgdir}/usr/bin/ollama"
    cp -r "${srcdir}/lib/ollama/"* "${pkgdir}/usr/lib/ollama/"

    cat > "${pkgdir}/usr/lib/systemd/system/ollama.service" << 'EOF'
[Unit]
Description=Ollama with ROCm for Large Models
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=ollama
EnvironmentFile=/etc/default/ollama
ExecStart=/usr/bin/ollama serve
Restart=always
RestartSec=3
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/nvme3/models /var/lib/ollama /tmp
PrivateTmp=true
Environment=HIP_VISIBLE_DEVICES=0
Environment=HSA_OVERRIDE_GFX_VERSION=11.0.0

[Install]
WantedBy=multi-user.target
EOF

    cat > "${pkgdir}/etc/default/ollama" << EOF
OLLAMA_MODELS=${_model_dir}
OLLAMA_HOST=127.0.0.1:11434
OLLAMA_NUM_PARALLEL=1
HIP_VISIBLE_DEVICES=0
HSA_OVERRIDE_GFX_VERSION=11.0.0
EOF
}
