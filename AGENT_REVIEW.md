# Agent Review: Ollama-AirLLM-ROCM Build Status

**Date:** 2026-02-08  
**Reviewer:** Second Agent  
**Hardware:** AMD RX 7900 XTX (gfx1100)  
**ROCM Version:** 4.0.0+unknown (ROCK module loaded, HSA Runtime 1.18)

---

## 🚨 CRITICAL ISSUES

### 1. PKGBUILD Does NOT Build From Source
**Location:** `/run/media/developer/CACHE/prismalama/PKGBUILD`

**Problem:** The PKGBUILD only has a `package()` function that copies pre-built binaries. It's missing:
- `prepare()` function
- `build()` function  
- Source code download/extraction
- Actual compilation steps

**Current PKGBUILD:**
```bash
package() {
  cd "$srcdir"
  cp -r usr "$pkgdir/"
  cp -r etc "$pkgdir/" 2>/dev/null || true
  # ... just copies files, doesn't build
}
```

**Required:** A proper PKGBUILD that:
1. Downloads/clones ollama source
2. Builds with ROCM support using CMake
3. Integrates AirLLM during build
4. Packages everything correctly

**Action:** The other agent needs to create a source-based PKGBUILD that calls the build process.

---

### 2. Build Script Not Integrated with PKGBUILD
**Location:** `/run/media/developer/CACHE/prismalama/build-rocm.sh`

**Problem:** There's a `build-rocm.sh` script that does the actual building, but:
- PKGBUILD doesn't call it
- Script has hardcoded paths (`/run/media/piotro/CACHE1/...`)
- Script doesn't properly detect ROCM architecture in all cases
- CMake configuration may not be optimal

**Issues in build-rocm.sh:**
- Line 19: Falls back to `gfx1100` if detection fails (good, but should verify)
- Line 35-37: Uses `hipconfig` but path might be wrong
- Line 39-46: CMake flags look correct but need verification
- Line 113: Hardcoded path `/run/media/piotro/CACHE1/airllm` won't work in package

**Action:** Integrate build-rocm.sh logic into PKGBUILD's build() function, or refactor to be PKGBUILD-compatible.

---

### 3. AirLLM ROCM Device Configuration Missing
**Location:** `/run/media/developer/CACHE/prismalama/runner/airllmrunner/airllm_runner.py`

**Problem:** AirLLM is initialized with `device="cuda:0"` by default, but for ROCM it should use:
- `device="cuda:0"` (PyTorch uses CUDA API for ROCM) OR
- `device="hip:0"` if AirLLM supports it

**Current code (line 137):**
```python
self.model = AutoModel(compressed=self.compression)
# Missing device parameter!
```

**Required:** Need to detect ROCM and set device appropriately:
```python
import os
device = "cuda:0"  # PyTorch uses CUDA API for ROCM
if os.environ.get("ROCM_PATH"):
    device = "cuda:0"  # Still cuda:0 for PyTorch+ROCM
self.model = AutoModel(compressed=self.compression, device=device)
```

**Action:** Verify AirLLM works with ROCM/PyTorch and set device correctly.

---

### 4. Hardcoded Paths in Multiple Files
**Locations:**
- `build-rocm.sh:113` - `/run/media/piotro/CACHE1/airllm`
- `runner/airllmrunner/runner.go:113` - `/run/media/piotro/CACHE1/prismalama/airllm/air_llm`
- `runner/airllmrunner/airllm_runner.py:127` - Same hardcoded path

**Problem:** These paths are user-specific and won't work in a packaged installation.

**Required:** Use environment variables or package-relative paths:
- `/usr/share/ollama/airllm` (already in some places)
- `$OLLAMA_MODELS` environment variable
- Relative paths from installation directory

**Action:** Replace all hardcoded paths with package-appropriate locations.

---

### 5. ROCM Architecture Detection Needs Verification
**Hardware:** RX 7900 XTX = gfx1100 ✓ (correctly detected)

**Current detection:**
```bash
ROCM_ARCH=$(rocm_agent_enumerator 2>/dev/null | head -1)
if [ -z "$ROCM_ARCH" ]; then
    ROCM_ARCH="gfx1100"  # Fallback
fi
```

**Issue:** The CMakeLists.txt filters AMDGPU_TARGETS (line 127):
```cmake
list(FILTER AMDGPU_TARGETS INCLUDE REGEX "^gfx(94[012]|101[02]|1030|110[012]|120[01])$")
```

**gfx1100 is in the list** ✓, but need to ensure:
1. Detection works reliably
2. CMake receives the correct architecture
3. Build actually uses ROCM (not CPU fallback)

**Action:** Verify ROCM compilation is actually happening and producing HIP binaries.

---

## ⚠️ MODERATE ISSUES

### 6. AirLLM VRAM Offloading Logic
**Requirement:** "if the model is bigger than available vram, we start offloading to ram and streaming back on inference"

**Current Status:** AirLLM does layer-by-layer loading, but:
- Need to verify it automatically offloads to RAM when VRAM is full
- Need to ensure streaming works correctly
- Need to test with models larger than VRAM

**Action:** Test with a model larger than available VRAM (24GB for RX 7900 XTX) to verify offloading works.

---

### 7. Python Dependencies Not in PKGBUILD
**Problem:** AirLLM requires Python packages:
- torch (with ROCM support)
- transformers
- accelerate
- bitsandbytes (for compression)
- tqdm

**Current:** These are not listed as dependencies in PKGBUILD.

**Required:** Add to `depends` or `makedepends`:
```bash
depends=(... 'python-torch' 'python-transformers' 'python-accelerate' 'python-bitsandbytes' 'python-tqdm')
```

**Action:** Verify which packages are available in Arch repos and add them.

---

### 8. Systemd Service Configuration
**Location:** `build-rocm.sh:81-104`

**Issues:**
- Hardcoded paths in `ReadWritePaths`
- User `ollama` needs proper permissions
- Environment variables need to be set correctly

**Action:** Review and fix systemd service file.

---

## ✅ WHAT'S WORKING

1. **ROCM Detection:** Hardware correctly identified as gfx1100
2. **CMake Configuration:** CMakeLists.txt has proper ROCM/HIP support
3. **AirLLM Integration:** Basic integration code exists in runner/
4. **Build Script Structure:** build-rocm.sh has the right approach
5. **Package Structure:** Directory layout looks reasonable

---

## 📋 REQUIRED ACTIONS FOR OTHER AGENT

### Priority 1 (Blocking):
1. **Create source-based PKGBUILD** that actually builds ollama with ROCM
2. **Fix hardcoded paths** in all files
3. **Verify ROCM compilation** is happening (check build logs)
4. **Set AirLLM device** correctly for ROCM

### Priority 2 (Important):
5. **Add Python dependencies** to PKGBUILD
6. **Test VRAM offloading** with large models
7. **Verify streaming** works correctly
8. **Fix systemd service** paths

### Priority 3 (Polish):
9. **Add proper error handling** in build script
10. **Document ROCM requirements** in PKGBUILD
11. **Add build verification** steps

---

## 🔍 VERIFICATION CHECKLIST

Before considering the build complete, verify:

- [ ] PKGBUILD builds from source (not just packages binaries)
- [ ] `makepkg` successfully compiles ollama with ROCM
- [ ] Generated binaries are linked against ROCM libraries
- [ ] AirLLM can load models using ROCM GPU
- [ ] Models larger than VRAM offload to RAM correctly
- [ ] Inference streams correctly
- [ ] Package installs and runs on clean Arch system
- [ ] Systemd service starts correctly
- [ ] opencode can use ollama models automatically

---

## 📝 NOTES FOR OTHER AGENT

1. **ROCM Version:** You have ROCM 4.0.0+unknown. Check if this is compatible with the ollama/ggml ROCM support. Some versions may have issues.

2. **PyTorch ROCM:** AirLLM uses PyTorch. Ensure PyTorch is compiled with ROCM support, not just CUDA. Arch's `python-torch` may need ROCM variant.

3. **Build Environment:** The PKGBUILD build() function runs in a clean chroot. All dependencies must be in `depends`/`makedepends`.

4. **Testing:** After building, test with:
   ```bash
   # Check ROCM is being used
   ROCM_INFO=1 ollama run <model>
   
   # Check GPU usage
   rocm-smi
   
   # Test large model offloading
   ollama run <model-larger-than-24gb>
   ```

5. **Integration with opencode:** The requirement says "when I use opencode for using ollama models, this kicks in automatically." This suggests opencode has some integration. Check if there's an opencode config file that needs to point to this ollama installation.

---

## 🎯 GOAL REMINDER

The end goal is:
- ✅ Valid, working PKGBUILD for Arch Linux
- ✅ Pacman package that installs correctly
- ✅ Ollama compiled with ROCM for gfx1100 (RX 7900 XTX)
- ✅ AirLLM integrated and working
- ✅ Automatic VRAM→RAM offloading for large models
- ✅ Streaming inference
- ✅ Works automatically with opencode

**Current Status:** ~40% complete. Build infrastructure exists but PKGBUILD doesn't actually build from source.

---

**End of Review**
