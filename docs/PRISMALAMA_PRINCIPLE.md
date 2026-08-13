# Prismalama principle (north star)

![Prismalama Logo](../logo.jpg)

Prismalama is a **large, long-horizon** project: **Ollama-compatible UX** plus **Vulkan/ROCm GGML** for GGUF, and an **optional AirLLM path** for Hugging Face–style layouts and **true layer / NVMe–oriented streaming** where AirLLM applies.

## The key technical fact

**AirLLM-style weight streaming is not implemented inside llama.cpp / GGML.**  
The native GGUF path uses **mmap, partial GPU offload, and KV management** — different semantics and limits than PyTorch AirLLM’s layer-wise execution on huge HF trees.

Treating those as the same product feature causes **wrong expectations** and **wrong debugging** (GPU %, RAM, “streaming” wording). The project must be **explicit** about **two engines** and **how dispatch works**.

## What we ship

| Mechanism                                                 | Purpose                                                                                                                                                                                                                                                                                                                    |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`runner.DecideEngine`** (`runner/dispatch.go`)          | Single, testable decision: **GGML** vs **AirLLM** for a model directory + reason string.                                                                                                                                                                                                                                   |
| **`GET /api/prismalama/capabilities`**                    | Operator-facing JSON: **GGML vs AirLLM vs layer streaming** semantics, env state, doc pointers.                                                                                                                                                                                                                            |
| **`ml/streaming`** package                                | **Layer map**, **budget tracker**, **NVMe prefetcher**, **streamer orchestrator** — the infrastructure for AirLLM-like GGUF streaming inside the native Go + GGML stack. Controlled by **`OLLAMA_LAYER_STREAMING`** and **`OLLAMA_STREAMING_BUDGET`**.                                                                     |
| **`ml.StreamingBackend`** + **`Backend.LoadStreaming`**   | Optional backend interface: when `OLLAMA_LAYER_STREAMING=1`, the GGML backend loads weights **block-by-block** (sequential NVMe, format transforms, per-layer progress) instead of the default concurrent all-at-once load. The runner detects the interface at load time.                                                 |
| **`ml.StreamingComputeBackend`** + **GGML eval callback** | Streaming inference: the GGML scheduler's **eval callback** pauses compute at each block boundary (`ScanBlockBoundaries`), invokes `InferenceStreamer.OnBlockDone` to **load the next block's weights** from NVMe and **evict the previous block's**. Peak memory during inference = **1 block + KV cache + activations**. |
| **`ml/streaming.InferenceStreamer`**                      | Runtime coordinator: opens the GGUF file, loads initial block + output weights, then drives block-by-block weight rotation via `OnBlockDone`. The runner creates it at model-load time and installs it around every `ComputeWithNotify` call.                                                                              |
| **Integration + unit tests**                              | Ship bar covers dispatch, streaming env, budget defaults, orchestrator behavior, GGUF tensor name fidelity, backend interface detection, inference streamer lifecycle, and streaming compute interface.                                                                                                                    |

## Target: GGUF models, AirLLM-like streaming, prismallama compute

The **product key** is not “pick GGUF _or_ AirLLM forever.” It is:

1. **Keep GGUF as the on-disk and primary inference format** (single stack with Ollama UX, ROCm/Vulkan **prismallama** backends for **fast compute** once tensors are resident).
2. **Evolve the GGUF path** so weights can be **streamed and time-multiplexed** in the same _sense_ as AirLLM: load a **layer (or block)**, run **HIP/Vulkan GGML** ops, **evict / prefetch** from NVMe, repeat — without requiring PyTorch for models that are already GGUF.
3. **Use data transformations only where needed** (layout, quant, sharding) as **explicit, testable steps** — not a second runtime unless unavoidable.

That work lives primarily in **prismallama.cpp / GGML** (graph lifetime, buffer pools, async I/O, eviction), not in the Go server alone. AirLLM remains the **reference implementation** and the path for **HF safetensors** trees that are not GGUF.

## Roadmap alignment

- **Near term:** Harden **dispatch**, **docs**, **observability**, and **tests** so enterprise and power users know **which engine ran** and **why**.
- **Medium / long term:** Implement **GGUF streaming semantics** in **prismallama.cpp** toward AirLLM-like behavior on **NVMe + constrained VRAM**, reusing **fast prismallama** compute; **AirLLM** for HF layouts that will never be a single GGUF — see **`GOAL-GAPS.md`**, **`docs/WEIGHT_STREAMING_STRATEGY.md`**.

This document is the **architectural key** for Prismalama: **honest status today**, **clear target** (GGUF + streaming in-engine), **tested dispatch**, **no conflation** of current GGML mmap with the end-state streaming story.
