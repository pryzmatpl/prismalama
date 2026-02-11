# CMakeLists.txt Path Fix

## Problem
The build script can't find CMakeLists.txt because:
1. `src/ollama` exists as a directory (not a symlink)
2. The directory doesn't contain CMakeLists.txt
3. CMakeLists.txt exists in the workspace root

## Solution
Updated the build script to:
1. **Check if src/ollama is incomplete** - If it's a directory without CMakeLists.txt, remove it
2. **Create symlink to workspace root** - Use workspace root which has CMakeLists.txt
3. **Better path resolution** - Properly resolve symlinks and check workspace root

## Changes Made

### build-all.sh
- Added check for incomplete directory
- Remove and recreate as symlink if needed
- Better CMakeLists.txt path resolution

## Test

```bash
# Clean and rebuild
rm -rf src/ollama
./build-all.sh
```

Or the script will automatically fix it:
```bash
./build-all.sh
# It will detect incomplete directory and create symlink
```

The build should now find CMakeLists.txt in the workspace root.
