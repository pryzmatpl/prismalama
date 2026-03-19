# Fix Primalama: Add Qwen3.5 Support and PKGBUILD Build from Source

## Objective
Fix the ollama-airllm-rocm package in /sda2/prismalama to properly build from source, add Qwen3.5 architecture support, and ensure ROCm integration. This will allow the built package to load and run Qwen3.5 models with ollama.

## Context
- Current PKGBUILD only packages pre-built binaries, so source changes aren't applied.
- Model architecture 'qwen35' is unknown in the GGML backend.
- AirLLM integration needs device configuration for ROCm.
- Hardcoded paths need replacement.

## Steps to Execute

### 1. Fix PKGBUILD to Build from Source
Modify /sda2/prismalama/PKGBUILD to instead of copying pre-built files, build ollama from the GitHub source with ROCm support.

- Add prepare() and build() functions as shown in CRITICAL_FIXES_NEEDED.md.
- Use Git source from https://github.com/ollama/ollama.git.
- Enable ROCM compilation with CMake flags:
  -DLLAMA_CURL=ON
  -DLLAMA_HIPBLAS=ON
  -DLLAMA_CUDA=OFF
- Build Go binary with proper flags.
- Detect ROCM architecture automatically.

### 2. Add Qwen3.5 Architecture to GGML Backend
In the ollama source code:
- Find where model architectures are defined (likely in llama.h or gguf.h).
- Add "qwen35" to the list of supported architectures, similar to "qwen2".

### 3. Add Qwen3.5 Support to AirLLM
Create /sda2/prismalama/airllm-clean/air_llm/airllm/airllm_qwen35.py by copying airllm_qwen.py.
- Modify the class to handle qwen35 models.
- Ensure it can use ROCm device.

### 4. Fix Device Configuration for AirLLM
Edit /sda2/prismalama/runner/airllmrunner/airllm_runner.py:
- Add device="cuda:0" parameter for ROCm support (PyTorch uses CUDA API for ROCM).

### 5. Remove Hardcoded Paths
Update:
- build-rocm.sh: Replace hardcoded /run/media/piotro/CACHE1/ with $OLLAMA_MODELS
- runner/airllmrunner/runner.go: Use /usr/share/ollama/airllm path
- airllm_runner.py: Remove hardcoded paths, use /usr/share/ollama/airllm

### 6. Install Python Dependencies in PKGBUILD
Add python dependencies to depends array:
- python-torch (ROCM enabled if available)
- python-transformers, python-accelerate, etc.

### 7. Rebuild Package
Run makepkg -s in /sda2/prismalama to build the package.

### 8. Test Build
- Install the package: pacman -U .pkg.tar.zst
- Start ollama and test with a Qwen3.5 model
- Verify ROCm usage with rocm-smi

## Verification
- After build, ollama should load 'qwen35' architecture without error.
- OpenCode should load the model successfully.
- Model inference should use ROCm GPU.

## Output
Provide summarized completion status for each step. Report any errors found and potential fixes.