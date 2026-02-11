# Build Fix - Symlink Instead of Copy

## Problem
The build script was trying to copy the entire workspace directory to itself:
```
cp: nie można skopiować katalogu '/run/media/developer/CACHE/prismalama' do siebie samego
```

## Solution
Changed the build script to use a **symlink** instead of copying when using the workspace root as the ollama source.

## Changes Made

1. **Create symlink instead of copy:**
   ```bash
   # Before: cp -r "${SCRIPT_DIR}" "$SRC_DIR/ollama"
   # After:  ln -sf "${SCRIPT_DIR}" "$SRC_DIR/ollama"
   ```

2. **Resolve symlink in build functions:**
   - Added symlink resolution in `build_ollama()` function
   - Added symlink resolution in AirLLM integration setup
   - Skip submodule init if it's a symlink (workspace root)

3. **Benefits:**
   - No recursive copy issues
   - Uses workspace root directly
   - Faster (no copying needed)
   - Changes in workspace are immediately available

## Test

```bash
./build-all.sh
```

The build should now work correctly using the workspace root as the ollama source.
