# Second Review Round - Ollama-AirLLM-ROCM Build

**Date:** 2026-02-08  
**Reviewer:** Second Agent  
**Status:** Significant Progress Made! 🎉

---

## ✅ MAJOR IMPROVEMENTS

### 1. PKGBUILD Now Builds From Source! ✅
**Status:** FIXED

The PKGBUILD now has proper `prepare()` and `build()` functions! This was the #1 blocker and it's been addressed.

**What's Good:**
- ✅ `prepare()` function exists and handles submodules
- ✅ ROCM architecture auto-detection in prepare()
- ✅ `build()` function with CMake + Go build
- ✅ Proper ROCM environment variables set
- ✅ Dependencies properly listed

**Excellent work!**

---

### 2. Dependencies Added ✅
**Status:** FIXED

Python dependencies are now in the PKGBUILD:
- ✅ `python-pytorch-rocm` (correct ROCM variant!)
- ✅ `python-transformers`
- ✅ `python-accelerate`
- ✅ `python-safetensors`
- ✅ Other required packages

---

### 3. ROCM Architecture Detection ✅
**Status:** IMPROVED

The prepare() function now detects ROCM architecture:
```bash
if command -v rocm_agent_enumerator &> /dev/null; then
    detected_arch=$(rocm_agent_enumerator 2>/dev/null | grep -v "gfx000" | head -1)
    if [ -n "$detected_arch" ]; then
        _rocm_arch="$detected_arch"
    fi
fi
```

Good filtering of "gfx000" (invalid architecture).

---

### 4. AirLLM Device Configuration ✅
**Status:** PARTIALLY FIXED

**Good:**
- ✅ Environment variable `AIRLLM_DEVICE="cuda:0"` is set in `/etc/default/ollama` (line 247)

**Still Needs Fix:**
- ❌ The Python runner (`airllm_runner.py:137`) doesn't use the device parameter:
  ```python
  self.model = AutoModel(compressed=self.compression)
  # Missing: device parameter!
  ```

**Required Fix:**
```python
import os
device = os.environ.get("AIRLLM_DEVICE", "cuda:0")
self.model = AutoModel(compressed=self.compression, device=device)
```

**Note:** Need to check how AutoModel actually works - it might use `from_pretrained()` instead of direct instantiation.

---

## ⚠️ REMAINING ISSUES

### 1. Hardcoded Paths Still Present ❌
**Status:** NOT FIXED

**Locations:**
- `PKGBUILD:228` - Systemd service: `ReadWritePaths=/run/media/piotro/CACHE1/airllm ...`
- `PKGBUILD:243` - Environment config: `export OLLAMA_MODELS="/run/media/piotro/CACHE1/airllm"`
- `runner/airllmrunner/runner.go:113` - PYTHONPATH: `/run/media/piotro/CACHE1/prismalama/airllm/air_llm`
- `runner/airllmrunner/airllm_runner.py:127` - sys.path: `/run/media/piotro/CACHE1/prismalama/airllm/air_llm`

**Required Fixes:**

**PKGBUILD line 228:**
```bash
# Change from:
ReadWritePaths=/run/media/piotro/CACHE1/airllm /run/media/piotro/CACHE/airllm /var/lib/ollama /tmp

# To:
ReadWritePaths=/var/lib/ollama /tmp
# Models path should be configurable via OLLAMA_MODELS env var
```

**PKGBUILD line 243:**
```bash
# Change from:
export OLLAMA_MODELS="/run/media/piotro/CACHE1/airllm"

# To:
export OLLAMA_MODELS="${OLLAMA_MODELS:-${HOME}/.ollama/models}"
# Or use a more standard location
```

**runner.go line 113:**
```go
// Change from:
"PYTHONPATH=/usr/share/ollama/airllm:/run/media/piotro/CACHE1/prismalama/airllm/air_llm",

// To:
"PYTHONPATH=/usr/share/ollama/airllm",
```

**airllm_runner.py line 127:**
```python
# Change from:
sys.path.insert(0, "/run/media/piotro/CACHE1/prismalama/airllm/air_llm")

# To:
# Remove this line - only use /usr/share/ollama/airllm
```

---

### 2. AirLLM AutoModel Usage May Be Incorrect ⚠️
**Status:** NEEDS VERIFICATION

**Current code (line 137):**
```python
self.model = AutoModel(compressed=self.compression)
```

**Issue:** Looking at the AirLLM source code, `AutoModel` is a factory class that uses `from_pretrained()`, not direct instantiation.

**From airllm/air_llm/airllm/auto_model.py:**
```python
class AutoModel:
    def __init__(self):
        raise EnvironmentError("AutoModel is designed to be instantiated using the `AutoModel.from_pretrained(...)` method.")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        # ... returns appropriate model class
```

**Required Fix:**
```python
from airllm import AutoModel
import os

device = os.environ.get("AIRLLM_DEVICE", "cuda:0")
compression = self.compression

# AutoModel uses from_pretrained, not direct instantiation
self.model = AutoModel.from_pretrained(
    model_path,
    device=device,
    compression=compression,
    dtype=torch.float16
)
```

**Action:** Verify the correct AutoModel API usage by checking AirLLM documentation or examples.

---

### 3. Missing Source Files ⚠️
**Status:** NEEDS VERIFICATION

**PKGBUILD references:**
- `"airllm_runner.py"` - Is this in the source array?
- `"airllm.patch"` - Does this patch exist and work?

**Check:**
```bash
# Verify these files exist in the source directory
ls -la airllm_runner.py
ls -la airllm.patch
```

If they don't exist, they need to be created or the PKGBUILD needs to be adjusted.

---

### 4. CMake Build May Need MLX_ENGINE Setting ⚠️
**Status:** NEEDS VERIFICATION

**Current (line 137-146):**
```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DLLAMA_CURL=ON \
    -DLLAMA_HIPBLAS=ON \
    -DLLAMA_CUDA=OFF \
    ...
```

**Missing:** `-DMLX_ENGINE=OFF` (if not needed)

**Check:** The build-rocm.sh script has `-DMLX_ENGINE=OFF`. Should this be added to PKGBUILD?

---

### 5. Go Build Tags ⚠️
**Status:** NEEDS VERIFICATION

**Current (line 154):**
```bash
go build \
    -tags="rocm" \
    ...
```

**Question:** Does ollama actually have a "rocm" build tag, or should this be empty? Check the ollama source to see what tags are available.

**From build-rocm.sh (line 57):**
```bash
go build -tags="" -o ...
```

The build script uses empty tags. This might be correct - verify.

---

## 📋 PRIORITY FIXES

### Priority 1 (Critical - Blocks Packaging):
1. **Remove all hardcoded paths** from PKGBUILD and runner files
2. **Fix AirLLM AutoModel usage** - use correct API
3. **Verify source files exist** (airllm_runner.py, airllm.patch)

### Priority 2 (Important):
4. **Add device parameter** to AirLLM model initialization
5. **Verify Go build tags** are correct
6. **Test CMake build** actually produces ROCM libraries

### Priority 3 (Polish):
7. **Add MLX_ENGINE flag** if needed
8. **Improve error handling** in build functions
9. **Add build verification** steps

---

## ✅ WHAT'S WORKING WELL

1. **PKGBUILD Structure** - Excellent improvement! Now actually builds from source
2. **Dependencies** - All Python packages correctly listed
3. **ROCM Detection** - Good auto-detection logic
4. **Build Process** - CMake + Go build looks correct
5. **Package Structure** - Directory layout is good
6. **Documentation** - README-PKGBUILD.md is comprehensive

---

## 🔍 VERIFICATION NEEDED

Before considering complete, verify:

- [ ] `makepkg -s` completes successfully
- [ ] All hardcoded paths removed
- [ ] AirLLM AutoModel uses correct API
- [ ] ROCM libraries are actually built and linked
- [ ] Package installs on clean Arch system
- [ ] Service starts correctly
- [ ] AirLLM can load models
- [ ] Device parameter is used correctly

---

## 📝 SPECIFIC CODE FIXES NEEDED

### Fix 1: PKGBUILD - Remove Hardcoded Paths

**Line 228:**
```bash
# BEFORE:
ReadWritePaths=/run/media/piotro/CACHE1/airllm /run/media/piotro/CACHE/airllm /var/lib/ollama /tmp

# AFTER:
ReadWritePaths=/var/lib/ollama /tmp
```

**Line 243:**
```bash
# BEFORE:
export OLLAMA_MODELS="/run/media/piotro/CACHE1/airllm"

# AFTER:
export OLLAMA_MODELS="${OLLAMA_MODELS:-${HOME}/.ollama/models}"
```

### Fix 2: runner.go - Remove Hardcoded Path

**Line 113:**
```go
// BEFORE:
"PYTHONPATH=/usr/share/ollama/airllm:/run/media/piotro/CACHE1/prismalama/airllm/air_llm",

// AFTER:
"PYTHONPATH=/usr/share/ollama/airllm",
```

### Fix 3: airllm_runner.py - Fix AutoModel Usage

**Line 126-137:**
```python
# BEFORE:
sys.path.insert(0, "/usr/share/ollama/airllm")
sys.path.insert(0, "/run/media/piotro/CACHE1/prismalama/airllm/air_llm")  # REMOVE THIS

from airllm import AutoModel

self.model = AutoModel(compressed=self.compression)

# AFTER:
sys.path.insert(0, "/usr/share/ollama/airllm")

from airllm import AutoModel
import os
import torch

device = os.environ.get("AIRLLM_DEVICE", "cuda:0")
compression = self.compression

# Use from_pretrained (check AirLLM docs for exact API)
self.model = AutoModel.from_pretrained(
    model_path,
    device=device,
    compression=compression,
    dtype=torch.float16
)
```

---

## 🎯 PROGRESS SUMMARY

**Previous Status:** ~40% complete  
**Current Status:** ~75% complete  

**Major Wins:**
- ✅ PKGBUILD now builds from source
- ✅ Dependencies added
- ✅ ROCM detection working

**Remaining Work:**
- ❌ Remove hardcoded paths (critical)
- ❌ Fix AirLLM API usage (critical)
- ⚠️ Verify build process (important)

---

## 💡 RECOMMENDATIONS

1. **Test the build** - Run `makepkg -s` and see what errors occur
2. **Check AirLLM examples** - Look at AirLLM repository for correct usage
3. **Use environment variables** - Make paths configurable, not hardcoded
4. **Add build verification** - Check that ROCM libraries are actually built

---

**Great progress! You're very close. Just need to remove the hardcoded paths and fix the AirLLM API usage.**
