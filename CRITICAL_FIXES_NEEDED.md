# 🚨 CRITICAL FIXES NEEDED - Quick Reference

## IMMEDIATE ACTION ITEMS

### 1. PKGBUILD Must Build From Source
**File:** `PKGBUILD`

**Current Problem:**
```bash
package() {
  cd "$srcdir"
  cp -r usr "$pkgdir/"  # Just copies, doesn't build!
}
```

**Required Structure:**
```bash
pkgname=ollama-airllm-rocm
pkgver=v0.4.1.r5053.4b15df6b
pkgrel=1
arch=('x86_64')
depends=('glibc' 'zlib' 'gcc-libs' 'rocm-hip-sdk' 'cmake' 'go' 'git' 
         'python' 'python-torch' 'python-transformers' 'python-accelerate' 
         'python-bitsandbytes' 'python-tqdm')
makedepends=('rocm-hip-sdk' 'cmake' 'go' 'git')
source=("ollama::git+https://github.com/ollama/ollama.git#commit=4b15df6b")
sha256sums=('SKIP')

prepare() {
  cd "$srcdir/ollama"
  # Apply any patches if needed
}

build() {
  cd "$srcdir/ollama"
  
  # Detect ROCM architecture
  ROCM_ARCH=$(rocm_agent_enumerator 2>/dev/null | head -1 || echo "gfx1100")
  export AMDGPU_TARGETS="$ROCM_ARCH"
  export HIP_PATH="$(hipconfig -R)"
  
  # Build with CMake + ROCM
  mkdir -p build
  cd build
  cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLX_ENGINE=OFF \
    -DLLAMA_CURL=ON \
    -DLLAMA_HIPBLAS=ON \
    -DLLAMA_CUDA=OFF \
    -DCMAKE_HIP_COMPILER_ROCM_ROOT="$HIP_PATH" \
    -DCMAKE_INSTALL_PREFIX=/usr
  
  cmake --build . --config Release -j$(nproc)
  
  # Build Go binary
  cd "$srcdir/ollama"
  export GOFLAGS="-trimpath -buildmode=pie"
  export CGO_ENABLED=1
  go build -tags="" -o ollama \
    -ldflags="-w -s -X=github.com/ollama/ollama/version.Version=${pkgver}" .
}

package() {
  cd "$srcdir/ollama"
  
  # Install binary
  install -Dm755 ollama "$pkgdir/usr/bin/ollama"
  
  # Install libraries from CMake build
  install -Dm755 build/lib/ollama/*.so "$pkgdir/usr/lib/ollama/"
  
  # Install AirLLM
  cp -r airllm/air_llm "$pkgdir/usr/share/ollama/airllm"
  cp runner/airllmrunner/airllm_runner.py "$pkgdir/usr/share/ollama/"
  
  # Install systemd service
  install -Dm644 scripts/ollama.service "$pkgdir/usr/lib/systemd/system/ollama.service"
  
  # Install config
  install -Dm644 /dev/stdin "$pkgdir/etc/default/ollama" <<EOF
export OLLAMA_MODELS="\${HOME}/.ollama/models"
export PYTHONPATH="/usr/share/ollama/airllm:\${PYTHONPATH}"
export AIRLLM_COMPRESSION="4bit"
EOF
}
```

---

### 2. Fix Hardcoded Paths

**Files to fix:**
- `build-rocm.sh:113` - Change `/run/media/piotro/CACHE1/airllm` → Use `$OLLAMA_MODELS` or `$HOME/.ollama/models`
- `runner/airllmrunner/runner.go:113` - Remove hardcoded path, use `/usr/share/ollama/airllm` only
- `runner/airllmrunner/airllm_runner.py:127` - Same as above

**Fix for runner.go:**
```go
cmd.Env = append(os.Environ(),
    "AIRLLM_COMPRESSION="+os.Getenv("AIRLLM_COMPRESSION"),
    "PYTHONPATH=/usr/share/ollama/airllm",
)
```

**Fix for airllm_runner.py:**
```python
sys.path.insert(0, "/usr/share/ollama/airllm")
# Remove the hardcoded /run/media/piotro/CACHE1/prismalama/airllm/air_llm line
```

---

### 3. Set AirLLM Device for ROCM

**File:** `runner/airllmrunner/airllm_runner.py:137`

**Current:**
```python
self.model = AutoModel(compressed=self.compression)
```

**Required:**
```python
import os
# PyTorch uses CUDA API for ROCM, so device="cuda:0" works
# But we should detect if ROCM is available
device = "cuda:0"
if os.environ.get("ROCM_PATH") or os.path.exists("/opt/rocm"):
    device = "cuda:0"  # PyTorch+ROCM still uses cuda:0

self.model = AutoModel(
    compressed=self.compression,
    device=device,
    dtype=torch.float16
)
```

---

### 4. Verify ROCM Compilation

**Test after build:**
```bash
# Check if binaries are linked against ROCM
ldd build/lib/ollama/libggml-hip.so | grep -i rocm

# Check if HIP code was compiled
strings build/lib/ollama/libggml-hip.so | grep -i "hip\|rocm" | head -5

# Verify architecture
readelf -d build/lib/ollama/libggml-hip.so | grep -i "gfx"
```

---

### 5. Add Python Dependencies

**In PKGBUILD depends array:**
```bash
depends=(
  'glibc' 'zlib' 'gcc-libs' 'rocm-hip-sdk'
  'python' 
  'python-torch'  # Check if Arch has ROCM variant
  'python-transformers'
  'python-accelerate'
  'python-bitsandbytes'
  'python-tqdm'
)
```

**Note:** Arch's `python-torch` may not have ROCM support. May need to build PyTorch with ROCM or use AUR package.

---

## TESTING CHECKLIST

After fixes, test:

```bash
# 1. Build package
makepkg -s

# 2. Install
sudo pacman -U ollama-airllm-rocm-*.pkg.tar.zst

# 3. Check ROCM detection
rocm-smi
rocm_agent_enumerator

# 4. Test ollama
ollama --version

# 5. Test with model (if available)
OLLAMA_MODELS=/path/to/models ollama run <model>

# 6. Check GPU usage during inference
rocm-smi -d 0
```

---

## QUICK WINS

1. **Fix PKGBUILD structure** - This is the #1 blocker
2. **Remove hardcoded paths** - 5 minute fix
3. **Add device parameter** - 2 minute fix
4. **Test build** - Verify it actually compiles

---

**Priority Order:**
1. PKGBUILD build() function
2. Hardcoded paths
3. Device configuration
4. Dependencies
5. Testing
