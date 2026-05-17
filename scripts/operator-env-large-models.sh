#!/usr/bin/env bash
# Recommended environment for Prismalama when models are larger than VRAM and you want
# GPU-first scheduling with conservative KV defaults.
#
# Usage (before starting the server in the same shell):
#   set -a
#   source /path/to/prismalama/scripts/operator-env-large-models.sh
#   set +a
#   ollama serve
#
# Or merge these into systemd Environment= lines / /etc/default/ollama (distribution-specific).

# Conservative default context tiers + parallel KV clamp (see server/memory_policy.go)
export OLLAMA_MEMORY_POLICY="${OLLAMA_MEMORY_POLICY:-balanced}"

# GGUF layer streaming path (off when unset in bare binary — see docs/GOAL-GAPS.md)
export OLLAMA_LAYER_STREAMING="${OLLAMA_LAYER_STREAMING:-1}"

# Streaming buffer pool for qwen35moe / >VRAM models on 12GB GPUs (default 4 GiB is conservative)
export OLLAMA_STREAMING_BUDGET="${OLLAMA_STREAMING_BUDGET:-6442450944}"

# Ollama-engine path (required for qwen35moe / qwen3.6:35b)
export OLLAMA_NEW_ENGINE="${OLLAMA_NEW_ENGINE:-1}"

# Linux: allow mmap when resident RAM < model size so weights can page from NVMe
export OLLAMA_MMAP_ALLOW_LOW_RAM="${OLLAMA_MMAP_ALLOW_LOW_RAM:-1}"

# Linux: Vulkan GGML backends are skipped in discovery unless enabled (discover/runner.go).
# CUDA/HIP are still enumerated separately when libraries exist.
export OLLAMA_VULKAN="${OLLAMA_VULKAN:-1}"
