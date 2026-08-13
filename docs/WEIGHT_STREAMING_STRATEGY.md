# Weight streaming strategy (Prismalama)

**Product principle (Arch and default installs):** the **primary** stack is **Go + prismallama.cpp / GGML** (ROCm HIP, Vulkan, CPU). End users should not need PyTorch, `transformers`, or manual `pip` installs for normal **GGUF** workflows. **AirLLM** (Python + optional heavy deps) is an **opt-in** path for Hugging Face–style checkpoints and experiments — ship and document it as secondary to the native engine.

This note compares **high-probability** directions for “weight streaming” and large-model inference. It complements `docs/DEVELOPER.md` (runner selection, AirLLM vs llama.cpp). For **runtime dispatch** (what actually runs for a given path: llama vs ollama engine vs AirLLM, logs), see **`docs/RUNTIME_DISPATCH.md`**.

“Weight streaming” in this repo names **two different problems**:

1. **HF / PyTorch models** that may never live as a single GGUF — AirLLM’s layer-by-layer execution domain.
2. **GGUF + GGML** — the engine already uses **mmap**, **partial GPU offload**, and **layer layout** (`LoadOperation` fit → alloc → commit, `UseMmap`, Vulkan-specific behavior in `llm/server.go`). That is **not** the same semantics as AirLLM’s time-multiplexed weights inside one PyTorch forward.

---

## Option 1 — Double down on GGUF + prismallama.cpp / GGML (Vulkan + HIP), extend the engine

**Idea:** Treat “streaming” as **better memory discipline inside llama.cpp**: mmap, partial offload, optional block-at-a-time load/evict _inside the same GGML stack_ (your fork), not a second Python runtime.

| Pros                                                                            | Cons                                                                                                                                       |
| ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| One **Go + GGML** artifact (matches a slim Docker story).                       | **HF / safetensors** layouts still need conversion to GGUF or another path.                                                                |
| Aligns with **Vulkan as a first-class GGML backend** (vendor-agnostic GPU API). | **True** “weights larger than VRAM” sequential execution is **not** what stock llama.cpp optimizes for; large design and performance work. |
| Reuses existing load/offload and scheduler code paths.                          | Long **research/engineering** cycle; easy to underperform PyTorch until tuned.                                                             |

**Probability of success:** **High** for “large GGUF runs well on modest VRAM **with** mmap/offload/quant”; **medium** for “replace AirLLM for all huge non-GGUF models”; **lower** if the bar is **SOTA** vs AirLLM on arbitrary HF trees without GGUF.

---

## Option 2 — Pragmatic split: GGML for GGUF + Vulkan; keep AirLLM only where unavoidable

**Idea:** Do **not** fold HF mega-models into GGML yet. **Harden** the GGUF path (defaults, mmap, offload, multi-part GGUF handling, docs, tests) and **reserve** AirLLM for **safetensors / non-GGUF** or experimental routes.

| Pros                                                                               | Cons                                                                   |
| ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **Highest short-term success**: fewer moving parts per release.                    | **Two** runtimes (Go+GGML vs Python+Torch).                            |
| Docker **GPU image** stays lean for the **common** case (GGUF inference).          | Large **HF-only** models still depend on PyTorch/ROCm stack.           |
| Clear product story: “GGUF in container; AirLLM optional on host or fatter image.” | Not a single “pure Vulkan streaming layer” for **all** weight formats. |

**Probability of success:** **Highest** overall for **shipping** and **reliability** in the near term.

---

## Option 3 — New minimal “streaming layer” in Go on top of GGML (orchestration only)

**Idea:** A **thin** Go layer that loads **one block / tensor group at a time**, copies to GPU, runs the op, evicts — **without** PyTorch, “bare essentials” only.

| Pros                                                          | Cons                                                                                                     |
| ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Theoretical** fit with “non-vendor Vulkan + Go only.”       | **Hardest** to make **fast**: the **runner + ggml graph** already own tensor lifetime, graphs, KV cache. |
| Full control over I/O and scheduling.                         | High risk of **slower than AirLLM** for a long time; many **Vulkan buffer** and **sync** footguns.       |
| Could target **one** model family / quant first to cap scope. | **Generality** (all architectures) is expensive; easy to become a one-off.                               |

**Probability of success:** **Medium–low** for **performance + generality**; **medium** for a **narrow** demo (fixed arch, fixed quant, fixed GPU).

---

## Summary

- **Maximum probability of a working, maintainable Prismalama:** **Option 2** (GGUF+GGML as primary; AirLLM as escape hatch).
- **Long-term one stack and Docker purity:** invest in **Option 1** on **prismallama.cpp**, not a parallel custom Go streaming layer first.
- **Option 3** is only worth it if scope is **explicitly** narrow (one model class, one backend) and the team accepts a long performance chase.

---

## Related

- `docs/DEVELOPER.md` — runner selection, AirLLM vs llama.cpp, env vars.
- `22-03-2026-DEV-PLAN.md` — roadmap and agent handoff.
- `llm/server.go` — load operations, mmap, GPU layout.
