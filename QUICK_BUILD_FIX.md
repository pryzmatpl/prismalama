# Quick Build Fix

## Problem
`CMakeLists.txt` doesn't exist in ollama v0.5.7 tag.

## Solution
The build script has been updated to:
1. **First check if workspace root has ollama** - If CMakeLists.txt exists in workspace root, use that (it's a newer version)
2. **Clone full repository** - Instead of shallow clone, get full history
3. **Fallback to main branch** - If tag doesn't have CMakeLists.txt, use main branch

## Try This

```bash
# Clean and rebuild
rm -rf src/ollama
./build-all.sh
```

## Alternative: Use Workspace Root Directly

If the workspace root already has a working ollama, you can modify the build to use it:

```bash
# In build-all.sh, the fix now checks for workspace root first
# If CMakeLists.txt exists in workspace root, it will use that
```

## If Still Failing

Check what version actually has CMakeLists.txt:
```bash
cd src/ollama
git fetch --all --tags
# Find a tag with CMakeLists.txt
git tag | while read tag; do
    if git show "$tag:CMakeLists.txt" >/dev/null 2>&1; then
        echo "$tag has CMakeLists.txt"
    fi
done | head -5
```

Then update PKGBUILD pkgver to use that version.
