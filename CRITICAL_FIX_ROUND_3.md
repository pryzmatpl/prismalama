# Critical Fix Needed - Round 3

## 🚨 ISSUE: Model Loading Methods Don't Exist

**File:** `runner/airllmrunner/airllm_runner.py:152-155`

**Problem:** The code calls `self.model.load_tokenizer()` and `self.model.load_model()`, but these methods don't exist on the AirLLM model classes.

**Current Code (WRONG - will crash):**
```python
self.model = AutoModel.from_pretrained(
    model_path,
    device=device,
    compression=compression,
    profiling_mode=False
)
self.progress = 0.5
logger.info("AutoModel initialized, loading model...")

# Load tokenizer and model
self.tokenizer = self.model.load_tokenizer(model_path)  # ❌ Method doesn't exist!
self.progress = 0.7

self.model.load_model(model_path)  # ❌ Method doesn't exist!
self.progress = 1.0
```

**Evidence:**
1. Looking at `airllm_base.py`, the tokenizer is loaded in `__init__()`:
   ```python
   self.tokenizer = self.get_tokenizer(hf_token=hf_token)  # Line 134
   ```

2. The model is initialized in `__init__()`:
   ```python
   self.init_model()  # Line 137
   ```

3. The inference example shows:
   ```python
   model = AirLLMLlama2("model-path")
   # Then directly use:
   model.tokenizer(...)  # No separate loading step
   ```

4. `load_model()` exists only in persister classes (for loading individual layers), not on the model itself.

**Fix (CORRECT):**
```python
self.model = AutoModel.from_pretrained(
    model_path,
    device=device,
    compression=compression,
    profiling_mode=False
)
self.progress = 0.5
logger.info("AutoModel initialized")

# Tokenizer is already loaded by from_pretrained
self.tokenizer = self.model.tokenizer  # Just get reference
self.progress = 0.7

# Model is already initialized by from_pretrained
# No need to call load_model() - it's already done
self.progress = 1.0
self._model_loaded = True

self.status = ServerStatus.READY
logger.info("AirLLM model loaded successfully")
```

**Why This Works:**
- `AutoModel.from_pretrained()` returns a fully initialized model
- The tokenizer is already available as `self.model.tokenizer`
- The model structure is already initialized
- No separate loading steps are needed

**Impact:** Without this fix, the runner will crash with `AttributeError: 'AirLLM...' object has no attribute 'load_tokenizer'`

---

## Quick Fix Summary

**Change lines 151-157 from:**
```python
# Load tokenizer and model
self.tokenizer = self.model.load_tokenizer(model_path)
self.progress = 0.7

self.model.load_model(model_path)
self.progress = 1.0
self._model_loaded = True
```

**To:**
```python
# Tokenizer and model are already loaded by from_pretrained
self.tokenizer = self.model.tokenizer
self.progress = 0.7

# Model is already initialized
self.progress = 1.0
self._model_loaded = True
```

---

**This is the last critical fix needed!**
