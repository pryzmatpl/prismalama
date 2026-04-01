# Prismalama principle (north star)

Prismalama is a **large, long-horizon** project: **Ollama-compatible UX** plus **Vulkan/ROCm GGML** for GGUF, and an **optional AirLLM path** for Hugging Face–style layouts and **true layer / NVMe–oriented streaming** where AirLLM applies.

## The key technical fact

**AirLLM-style weight streaming is not implemented inside llama.cpp / GGML.**  
The native GGUF path uses **mmap, partial GPU offload, and KV management** — different semantics and limits than PyTorch AirLLM’s layer-wise execution on huge HF trees.

Treating those as the same product feature causes **wrong expectations** and **wrong debugging** (GPU %, RAM, “streaming” wording). The project must be **explicit** about **two engines** and **how dispatch works**.

## What we ship to make that explicit

| Mechanism | Purpose |
|-----------|---------|
| **`runner.DecideEngine`** (`runner/dispatch.go`) | Single, testable decision: **GGML** vs **AirLLM** for a model directory + reason string. |
| **`GET /api/prismalama/capabilities`** | Operator-facing JSON: **GGML vs AirLLM semantics**, opt-in env, pointers to **`docs/RUNTIME_DISPATCH.md`**. |
| **Integration + unit tests** | Ship bar covers **dispatch opt-out** and **engine kind** so routing regressions are caught. |

## Roadmap alignment

- **Near term:** Harden **dispatch**, **docs**, **observability**, and **tests** so enterprise and power users know **which engine ran** and **why**.  
- **Long term:** **prismallama.cpp** / GGML evolution for better GGUF behavior on huge models; **AirLLM** for HF layouts that will never be a single GGUF — see **`GOAL-GAPS.md`**, **`docs/WEIGHT_STREAMING_STRATEGY.md`**.

This document is the **architectural key** for Prismalama: **honest dual-engine design**, **tested dispatch**, **no conflation** of GGML mmap with AirLLM streaming.
