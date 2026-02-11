# Review Summary: Ollama-AirLLM-ROCM Build

**Reviewer:** Second Agent  
**Date:** 2026-02-08  
**Hardware:** AMD RX 7900 XTX (gfx1100), ROCM 4.0.0

---

## Executive Summary

The other agent has made good progress on the AirLLM integration and build infrastructure, but **the PKGBUILD does not actually build from source**. It only packages pre-built binaries, which means the package cannot be built on a clean Arch system.

**Status:** ~40% complete - Foundation exists, but critical build steps missing.

---

## Critical Blockers

### 1. PKGBUILD Missing Build Functions ⚠️ CRITICAL
- **Current:** Only has `package()` function that copies files
- **Required:** Needs `prepare()` and `build()` functions
- **Impact:** Package cannot be built from source
- **Fix:** See `CRITICAL_FIXES_NEEDED.md` for template

### 2. Hardcoded User Paths ⚠️ CRITICAL  
- **Files:** `build-rocm.sh:113`, `runner/airllmrunner/runner.go:113`, `runner/airllmrunner/airllm_runner.py:127`
- **Problem:** Contains `/run/media/piotro/CACHE1/...` paths
- **Impact:** Won't work in packaged installation
- **Fix:** Use `/usr/share/ollama/airllm` and environment variables

### 3. AirLLM Device Not Configured ⚠️ MODERATE
- **File:** `runner/airllmrunner/airllm_runner.py:137`
- **Problem:** Missing `device` parameter for ROCM
- **Impact:** May not use GPU correctly
- **Fix:** Add `device="cuda:0"` (PyTorch uses CUDA API for ROCM)

---

## What's Working Well

✅ **ROCM Detection:** Correctly identifies gfx1100 architecture  
✅ **CMake Configuration:** Proper ROCM/HIP support in CMakeLists.txt  
✅ **AirLLM Integration:** Basic runner code exists and looks correct  
✅ **Build Script:** `build-rocm.sh` has the right approach  
✅ **Opencode Integration:** Code exists to configure opencode automatically  

---

## Documentation Created

I've created three documents for the other agent:

1. **AGENT_REVIEW.md** - Comprehensive technical review with all issues
2. **CRITICAL_FIXES_NEEDED.md** - Quick reference with code examples
3. **AGENT_NOTES.md** - Summary and guidance

**Read order:**
1. This file (REVIEW_SUMMARY.md) - Quick overview
2. AGENT_NOTES.md - Detailed guidance
3. CRITICAL_FIXES_NEEDED.md - What to fix with examples
4. AGENT_REVIEW.md - Deep technical analysis

---

## Opencode Integration Notes

From `cmd/config/opencode.go`, I can see that:
- Opencode uses `envconfig.Host().String() + "/v1"` to connect to ollama
- Default is `http://localhost:11434/v1`
- Opencode configures itself via `~/.config/opencode/opencode.json`
- The `ollama launch opencode` command sets this up automatically

**For automatic integration:**
- ✅ Binary at `/usr/bin/ollama` (already planned)
- ✅ Service at `/usr/lib/systemd/system/ollama.service` (already in build script)
- ✅ Service should start on boot (needs `systemctl enable`)

---

## Verification Checklist

After the other agent fixes the issues, verify:

- [ ] `makepkg -s` completes successfully
- [ ] Package installs with `pacman -U`
- [ ] `ollama --version` works
- [ ] ROCM libraries are linked: `ldd /usr/lib/ollama/libggml-hip.so | grep rocm`
- [ ] Service starts: `systemctl start ollama && systemctl status ollama`
- [ ] AirLLM can load models
- [ ] Large models (>24GB) offload to RAM
- [ ] Inference streams correctly
- [ ] `ollama launch opencode` works
- [ ] Opencode can use ollama models

---

## Next Steps for Other Agent

**Priority 1 (Blocking):**
1. Rewrite PKGBUILD to build from source
2. Remove all hardcoded paths
3. Add device parameter to AirLLM

**Priority 2 (Important):**
4. Add Python dependencies to PKGBUILD
5. Test ROCM compilation
6. Verify VRAM offloading

**Priority 3 (Polish):**
7. Improve error handling
8. Add build verification
9. Document requirements

---

## Key Insights

1. **PyTorch + ROCM:** PyTorch uses the CUDA API even with ROCM, so `device="cuda:0"` is correct
2. **Arch Package:** May need AUR package for ROCM-enabled PyTorch if official repos don't have it
3. **Build Environment:** `makepkg` runs in clean chroot - all deps must be listed
4. **AirLLM Offloading:** Should work automatically, but needs testing with large models

---

## Questions for Other Agent

1. Does Arch's `python-torch` have ROCM support, or do we need AUR?
2. Have you tested the build script (`build-rocm.sh`) manually?
3. What's the expected model format? GGUF? Safetensors? Both?
4. Should this replace the standard `ollama` package or be separate?

---

## Files to Review

**Critical:**
- `PKGBUILD` - Needs complete rewrite
- `build-rocm.sh` - Has logic but not integrated
- `runner/airllmrunner/airllm_runner.py` - Missing device config

**Important:**
- `CMakeLists.txt` - Verify ROCM is actually used
- `runner/airllmrunner/runner.go` - Fix hardcoded paths
- `build_ollama_airllm_rocm/etc/default/ollama` - Review config

**Reference:**
- `cmd/config/opencode.go` - Understand opencode integration
- `docs/integrations/opencode.mdx` - User documentation

---

## Success Criteria

The build is successful when:
- ✅ `makepkg -s` builds from source without errors
- ✅ Package installs and runs on clean Arch system
- ✅ ROCM GPU acceleration works
- ✅ AirLLM loads models and offloads to RAM when needed
- ✅ Opencode can use ollama automatically
- ✅ All hardcoded paths removed

---

**End of Summary**

For detailed information, see the other review documents.
