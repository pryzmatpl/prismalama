# Build Issue Analysis - CMakeLists.txt Missing

## Problem

The build fails because `CMakeLists.txt` doesn't exist in the `v0.5.7` tag of ollama:

```
CMake Error: The source directory "/run/media/developer/CACHE/prismalama/src/ollama" does not appear to contain CMakeLists.txt.
```

## Root Cause

1. **CMakeLists.txt exists in workspace root** - The current workspace has CMakeLists.txt
2. **CMakeLists.txt NOT in v0.5.7 tag** - The specific tag being cloned doesn't have it
3. **Version mismatch** - v0.5.7 might be too old or the build system changed

## Solutions

### Option 1: Use a newer version/tag that has CMakeLists.txt

Check which tags have CMakeLists.txt:
```bash
cd src/ollama
git fetch --all --tags
git tag | xargs -I {} sh -c 'git show {}:CMakeLists.txt >/dev/null 2>&1 && echo {}' | head -5
```

Then update PKGBUILD and build-all.sh to use that version.

### Option 2: Use main/master branch instead of tag

Update build-all.sh to clone main branch:
```bash
git clone --depth 1 https://github.com/ollama/ollama.git "$SRC_DIR/ollama"
cd "$SRC_DIR/ollama"
# Then checkout specific commit if needed
```

### Option 3: Use the workspace root's ollama source

If the workspace root already has a working ollama with CMakeLists.txt, use that:
```bash
# In build-all.sh, check if workspace root has ollama
if [ -f "${SCRIPT_DIR}/CMakeLists.txt" ] && [ -d "${SCRIPT_DIR}/llama" ]; then
    log_info "Using workspace root ollama source"
    cp -r "${SCRIPT_DIR}" "$SRC_DIR/ollama"  # Or symlink
fi
```

### Option 4: Check if CMakeLists.txt is in a submodule

The CMakeLists.txt might be in a submodule that needs to be initialized:
```bash
cd src/ollama
git submodule update --init --recursive
find . -name "CMakeLists.txt" -type f
```

## Recommended Fix

**Update PKGBUILD to use a version that has CMakeLists.txt**, or use the main branch:

1. Check what version the workspace root is using
2. Update `pkgver` in PKGBUILD to match
3. Or change the source to use main branch with a specific commit

## Quick Test

```bash
# Check if workspace root ollama works
cd /run/media/developer/CACHE/prismalama
if [ -f "CMakeLists.txt" ]; then
    echo "Workspace has CMakeLists.txt - consider using this source"
    # Test if it builds
    mkdir -p test-build && cd test-build
    cmake .. -DLLAMA_HIPBLAS=ON
fi
```

## Next Steps

1. Determine which ollama version actually has CMakeLists.txt
2. Update PKGBUILD pkgver to match
3. Or modify build script to use main branch
4. Test the build
