# Build Fix Needed - CMakeLists.txt Not Found

## 🚨 Issue

The build is failing because CMake cannot find `CMakeLists.txt` in the ollama source directory:

```
CMake Error: The source directory "/run/media/developer/CACHE/prismalama/src/ollama" does not appear to contain CMakeLists.txt.
```

## Root Cause

The git clone is checking out a specific commit (a420a453b4783841e3e79c248ef0fe9548df6914) which puts the repository in a detached HEAD state. The CMakeLists.txt might not be present at that commit, or the submodules haven't been initialized properly.

## Solutions

### Option 1: Check if CMakeLists.txt exists at a different location

```bash
cd /run/media/developer/CACHE/prismalama/src/ollama
find . -name "CMakeLists.txt" -type f
```

### Option 2: Ensure submodules are initialized BEFORE building

The build script initializes submodules, but CMakeLists.txt should be at the root. Check if it needs to be checked out:

```bash
cd /run/media/developer/CACHE/prismalama/src/ollama
git checkout v0.5.7  # Or the correct branch/tag
git submodule update --init --recursive
```

### Option 3: Fix the build script to handle detached HEAD

Update `build-all.sh` to ensure we're on the correct branch/tag:

```bash
# In prepare_sources() function, after cloning:
cd "$SRC_DIR/ollama"
# Ensure we're on the correct tag
git fetch --tags
git checkout "v${PKG_VERSION}" || git checkout "tags/v${PKG_VERSION}"
# Then initialize submodules
git submodule update --init --recursive
```

### Option 4: Check if CMakeLists.txt is in a subdirectory

Some ollama versions might have CMakeLists.txt in a subdirectory. Check:

```bash
cd /run/media/developer/CACHE/prismalama/src/ollama
ls -la ml/backend/ggml/ggml/src/CMakeLists.txt
```

If it's there, the build script needs to run CMake from that directory instead.

## Recommended Fix

Update `build-all.sh` prepare_sources() function:

```bash
prepare_sources() {
    log_step "Preparing sources..."
    
    mkdir -p "$SRC_DIR"
    
    # Clone Ollama if not present
    if [ ! -d "$SRC_DIR/ollama" ]; then
        log_info "Cloning Ollama repository (v${PKG_VERSION})..."
        git clone --depth 1 --branch "v${PKG_VERSION}" https://github.com/ollama/ollama.git "$SRC_DIR/ollama"
    else
        log_info "Ollama source already exists"
        cd "$SRC_DIR/ollama"
        # Ensure we're on the correct tag
        git fetch --tags
        git checkout "v${PKG_VERSION}" 2>/dev/null || git checkout "tags/v${PKG_VERSION}" 2>/dev/null || true
    fi
    
    # Initialize submodules BEFORE checking for CMakeLists.txt
    cd "$SRC_DIR/ollama"
    log_info "Initializing git submodules..."
    git submodule update --init --recursive
    
    # Verify CMakeLists.txt exists
    if [ ! -f "CMakeLists.txt" ]; then
        log_error "CMakeLists.txt not found in ollama source directory"
        log_info "Checking for CMakeLists.txt in subdirectories..."
        find . -name "CMakeLists.txt" -type f | head -5
        exit 1
    fi
    
    cd "$SCRIPT_DIR"
    
    # ... rest of the function
}
```

## Quick Test

Run this to diagnose:

```bash
cd /run/media/developer/CACHE/prismalama/src/ollama
echo "Current commit: $(git rev-parse HEAD)"
echo "Current branch: $(git branch --show-current 2>/dev/null || echo 'detached HEAD')"
echo "CMakeLists.txt exists: $([ -f CMakeLists.txt ] && echo 'YES' || echo 'NO')"
find . -name "CMakeLists.txt" -type f | head -5
```
