# Quick Fixes Needed - Round 2 Review

## 🚨 Critical Fixes (Do These First)

### 1. Fix AirLLM AutoModel API Usage

**File:** `runner/airllmrunner/airllm_runner.py`  
**Line:** 137

**Current (WRONG - will crash):**
```python
from airllm import AutoModel
self.model = AutoModel(compressed=self.compression)  # ❌ This will raise EnvironmentError!
```

**Fix (CORRECT):**
```python
from airllm import AutoModel
import os

device = os.environ.get("AIRLLM_DEVICE", "cuda:0")
compression = self.compression

# AutoModel uses from_pretrained, not direct instantiation
# Based on AirLLM examples, the API is:
self.model = AutoModel.from_pretrained(
    model_path,  # Path to model directory or HuggingFace repo ID
    device=device,
    compression=compression,
    profiling_mode=False
)
# Note: dtype defaults to torch.float16, can be specified if needed
```

**Why:** AutoModel.__init__() raises an error. You must use `from_pretrained()` classmethod.

---

### 2. Remove Hardcoded Path from airllm_runner.py

**File:** `runner/airllmrunner/airllm_runner.py`  
**Line:** 127

**Current:**
```python
sys.path.insert(0, "/usr/share/ollama/airllm")
sys.path.insert(0, "/run/media/piotro/CACHE1/prismalama/airllm/air_llm")  # ❌ REMOVE
```

**Fix:**
```python
sys.path.insert(0, "/usr/share/ollama/airllm")
# Remove the second line - only use package path
```

---

### 3. Remove Hardcoded Path from runner.go

**File:** `runner/airllmrunner/runner.go`  
**Line:** 113

**Current:**
```go
cmd.Env = append(os.Environ(),
    "AIRLLM_COMPRESSION="+os.Getenv("AIRLLM_COMPRESSION"),
    "PYTHONPATH=/usr/share/ollama/airllm:/run/media/piotro/CACHE1/prismalama/airllm/air_llm",  // ❌
)
```

**Fix:**
```go
cmd.Env = append(os.Environ(),
    "AIRLLM_COMPRESSION="+os.Getenv("AIRLLM_COMPRESSION"),
    "PYTHONPATH=/usr/share/ollama/airllm",  // ✅ Only package path
)
```

---

### 4. Fix PKGBUILD Systemd Service Paths

**File:** `PKGBUILD`  
**Line:** 228

**Current:**
```bash
ReadWritePaths=/run/media/piotro/CACHE1/airllm /run/media/piotro/CACHE/airllm /var/lib/ollama /tmp
```

**Fix:**
```bash
ReadWritePaths=/var/lib/ollama /tmp
# Models path should be configurable via OLLAMA_MODELS env var
```

---

### 5. Fix PKGBUILD Environment Config

**File:** `PKGBUILD`  
**Line:** 243

**Current:**
```bash
export OLLAMA_MODELS="/run/media/piotro/CACHE1/airllm"
```

**Fix:**
```bash
export OLLAMA_MODELS="${OLLAMA_MODELS:-${HOME}/.ollama/models}"
# Use standard location with fallback to user's home
```

---

## ⚠️ Important Fixes

### 6. Verify Source Files Exist

**Check if these exist:**
```bash
ls -la airllm_runner.py
ls -la airllm.patch
```

If they don't exist, either:
- Create them, OR
- Remove from PKGBUILD source array, OR
- Adjust PKGBUILD to handle missing files gracefully

---

### 7. Check Go Build Tags

**File:** `PKGBUILD`  
**Line:** 154

**Current:**
```bash
go build -tags="rocm" ...
```

**Question:** Does ollama have a "rocm" build tag?

**Check:** Look at ollama source or try:
```bash
go build -tags="" ...  # Empty tags (like build-rocm.sh does)
```

---

## ✅ Testing Checklist

After fixes, test:

```bash
# 1. Build package
makepkg -s

# 2. Check for errors
# Look for:
# - "EnvironmentError" from AutoModel
# - Missing file errors
# - Path errors

# 3. If build succeeds, install
sudo pacman -U ollama-airllm-rocm-*.pkg.tar.zst

# 4. Test
ollama --version
systemctl start ollama
systemctl status ollama
```

---

## 📝 Summary

**Critical (Must Fix):**
1. AutoModel API usage (will crash otherwise)
2. Remove hardcoded paths (4 locations)

**Important:**
3. Verify source files
4. Check Go build tags

**You're 75% there! Just need these final fixes.**
