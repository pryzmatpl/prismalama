# Submodule Fix - Missing airllm Submodule

## Problem
The build fails when initializing git submodules:
```
fatal: Nie znaleziono adresu dla ścieżki podmodułu „airllm” w .gitmodules
```

This happens because the workspace root has a reference to an "airllm" submodule in `.gitmodules`, but that submodule doesn't exist in the actual ollama repository.

## Solution
Updated the build script to:
1. **Ignore missing submodule errors** - Filter out errors about missing submodules
2. **Continue if critical submodules exist** - Check for `ml/backend/ggml/ggml` or `llama` directories
3. **Handle both symlink and regular cases** - Works for workspace root (symlink) and cloned repos

## Changes Made

### build-all.sh
- Added error filtering for submodule initialization
- Check for critical submodules before proceeding
- Handle symlinked workspace root case

## Test

```bash
./build-all.sh
```

The build should now:
1. Initialize submodules
2. Ignore the missing "airllm" submodule error
3. Continue if critical submodules (ggml, llama) are present
4. Proceed with the build

## Alternative: Remove airllm from .gitmodules

If you want to fix it permanently, you can remove the airllm entry from `.gitmodules`:

```bash
# Edit .gitmodules and remove the airllm entry
# Or use git to remove it:
git config --file .gitmodules --remove-section submodule.airllm 2>/dev/null || true
```

But the build script fix should handle this automatically now.
