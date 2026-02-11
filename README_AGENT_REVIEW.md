# ⚠️ AGENT REVIEW - READ THIS FIRST

**For the agent working on Ollama-AirLLM-ROCM build**

## 🚨 CRITICAL ISSUE FOUND

**The PKGBUILD does NOT build from source!** It only packages pre-built binaries.

This means the package cannot be built on a clean Arch system.

---

## 📋 Quick Action Items

1. **Fix PKGBUILD** - Add `prepare()` and `build()` functions (see `CRITICAL_FIXES_NEEDED.md`)
2. **Remove hardcoded paths** - Replace `/run/media/piotro/CACHE1/...` with package paths
3. **Configure AirLLM device** - Add `device="cuda:0"` parameter for ROCM

---

## 📚 Review Documents Created

I've created detailed review documents for you:

1. **REVIEW_SUMMARY.md** ← Start here for overview
2. **AGENT_NOTES.md** ← Detailed guidance and context
3. **CRITICAL_FIXES_NEEDED.md** ← Code examples and fixes
4. **AGENT_REVIEW.md** ← Comprehensive technical analysis

**Read in this order for best understanding.**

---

## ✅ What's Working

- ROCM detection (gfx1100) ✓
- CMake configuration for ROCM ✓
- AirLLM integration code exists ✓
- Build script structure ✓

## ❌ What's Missing

- PKGBUILD build functions
- Hardcoded paths removed
- AirLLM device configuration
- Python dependencies in PKGBUILD

---

## 🎯 Goal Reminder

Build a valid PKGBUILD that:
- ✅ Builds ollama from source with ROCM
- ✅ Integrates AirLLM automatically
- ✅ Offloads to RAM when VRAM is full
- ✅ Works with opencode automatically
- ✅ Creates a working Pacman package

---

**Status:** ~40% complete - Foundation exists, but critical build steps missing.

**See REVIEW_SUMMARY.md for full details.**
