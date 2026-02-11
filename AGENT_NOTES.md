# Notes for Other Agent Working on Ollama-AirLLM-ROCM

## Current Status Summary

✅ **What's Done:**
- Basic AirLLM integration code exists in `runner/airllmrunner/`
- Build script (`build-rocm.sh`) has ROCM detection and CMake setup
- CMakeLists.txt has ROCM/HIP support configured
- Hardware correctly detected as gfx1100 (RX 7900 XTX)
- Package structure exists (but incomplete)

❌ **What's Missing:**
- PKGBUILD doesn't build from source (just packages pre-built binaries)
- Hardcoded user-specific paths throughout codebase
- AirLLM device configuration not set for ROCM
- Python dependencies not in PKGBUILD
- No verification that ROCM compilation actually happens

---

## Key Files to Review

1. **PKGBUILD** - Needs complete rewrite to build from source
2. **build-rocm.sh** - Has build logic but not integrated with PKGBUILD
3. **runner/airllmrunner/airllm_runner.py** - Missing device parameter for ROCM
4. **runner/airllmrunner/runner.go** - Has hardcoded paths
5. **CMakeLists.txt** - Looks good, but verify ROCM is actually used

---

## Critical Path to Success

1. **Fix PKGBUILD** - This is blocking everything else
   - Add `prepare()` and `build()` functions
   - Integrate build-rocm.sh logic
   - Actually compile ollama with ROCM

2. **Remove Hardcoded Paths**
   - Replace `/run/media/piotro/CACHE1/...` with package paths
   - Use `/usr/share/ollama/airllm` for AirLLM
   - Use `$OLLAMA_MODELS` env var for model paths

3. **Configure AirLLM for ROCM**
   - Set `device="cuda:0"` (PyTorch uses CUDA API for ROCM)
   - Verify PyTorch has ROCM support

4. **Add Dependencies**
   - Python packages (torch, transformers, etc.)
   - Verify Arch repos have ROCM-enabled PyTorch

5. **Test Build**
   - Run `makepkg -s` and verify it compiles
   - Check that ROCM libraries are linked
   - Test with actual model

---

## Opencode Integration

The requirement says "when I use opencode for using ollama models, this kicks in automatically."

I found opencode integration code in:
- `cmd/config/opencode.go`
- `docs/integrations/opencode.mdx`

**Action:** Review these files to understand how opencode detects/uses ollama, and ensure the package installs in a way that opencode can find it.

Likely needs:
- Binary at `/usr/bin/ollama` ✓ (already in PKGBUILD)
- Service at `/usr/lib/systemd/system/ollama.service` ✓ (already in build script)
- Config at `/etc/default/ollama` ✓ (already in build script)

---

## ROCM Architecture Details

**Your Hardware:** AMD RX 7900 XTX
- **Architecture:** gfx1100 ✓
- **ROCM Version:** 4.0.0+unknown
- **HSA Runtime:** 1.18

**CMakeLists.txt filter (line 127):**
```cmake
list(FILTER AMDGPU_TARGETS INCLUDE REGEX "^gfx(94[012]|101[02]|1030|110[012]|120[01])$")
```

gfx1100 is in the allowed list, so compilation should work.

**Verification:**
```bash
rocm_agent_enumerator  # Should output: gfx1100
hipconfig -R           # Should output: /opt/rocm
```

---

## AirLLM VRAM Offloading

**Requirement:** "if the model is bigger than available vram, we start offloading to ram and streaming back on inference"

**How AirLLM Works:**
- AirLLM loads models layer-by-layer
- When a layer is processed, it's unloaded from GPU
- Activations can be stored in RAM
- This should automatically handle VRAM overflow

**Testing:**
- Use a model larger than 24GB (your VRAM)
- Monitor with `rocm-smi` during inference
- Verify RAM usage increases
- Check that inference still works

---

## Build Environment Notes

When `makepkg` runs, it:
- Uses a clean chroot (if using `makepkg -s`)
- Only has packages listed in `depends`/`makedepends`
- Runs as non-root user
- Needs network access for git clones

**Important:** All build tools must be in `makedepends`:
- `cmake`
- `go`
- `git`
- `rocm-hip-sdk`
- `python` (for AirLLM)

---

## Quick Test Commands

After building, test with:

```bash
# Install package
sudo pacman -U ollama-airllm-rocm-*.pkg.tar.zst

# Check ROCM
rocm-smi
rocm_agent_enumerator

# Check ollama
ollama --version

# Check if ROCM libraries are linked
ldd /usr/lib/ollama/libggml-hip.so | grep rocm

# Test service
sudo systemctl start ollama
sudo systemctl status ollama

# Test with model (if you have one)
OLLAMA_MODELS=/path/to/models ollama run <model>
```

---

## Common Pitfalls to Avoid

1. **Don't assume pre-built binaries exist** - PKGBUILD must build from source
2. **Don't use hardcoded user paths** - Use package-relative paths
3. **Don't forget Python dependencies** - AirLLM needs them
4. **Don't assume PyTorch has ROCM** - May need special build
5. **Don't skip testing** - Verify ROCM is actually used

---

## Questions to Answer

1. Does Arch's `python-torch` have ROCM support, or do we need AUR package?
2. How does opencode detect ollama? Does it check `/usr/bin/ollama`?
3. Should we create a separate `ollama-airllm-rocm` package or replace `ollama`?
4. What's the expected model format? GGUF? Safetensors? Both?

---

## Files Created for You

1. **AGENT_REVIEW.md** - Comprehensive review with all issues
2. **CRITICAL_FIXES_NEEDED.md** - Quick reference with code examples
3. **AGENT_NOTES.md** - This file (summary and guidance)

Read these in order:
1. AGENT_NOTES.md (this file) - Overview
2. CRITICAL_FIXES_NEEDED.md - What to fix
3. AGENT_REVIEW.md - Detailed analysis

---

## Success Criteria

The build is successful when:
- ✅ `makepkg -s` completes without errors
- ✅ Package installs with `pacman -U`
- ✅ `ollama --version` works
- ✅ ROCM libraries are linked (check with `ldd`)
- ✅ AirLLM can load models
- ✅ Large models offload to RAM
- ✅ Inference streams correctly
- ✅ Opencode can use it automatically

---

**Good luck! The foundation is there, you just need to wire it all together properly.**
