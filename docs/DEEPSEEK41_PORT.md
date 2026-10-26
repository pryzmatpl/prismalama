# DeepSeek-V4.1-Flash (`deepseek41`) — Prismalama compute path

> We own the full stack: `piotroxp/prismallama.cpp` → vendored `llama/llama.cpp` →
> HIP/Vulkan/CUDA GGML → Go runner → Ollama API → broker.
> This is **not** a Strata wrap. Strata's `strata-dsv41` is a parallel engine.

## Status (2026-10-25)

| Piece                                            | Status                                                                             |
| ------------------------------------------------ | ---------------------------------------------------------------------------------- |
| `LLM_ARCH_DEEPSEEK41` / `models/deepseek41.cpp`  | **In tree** (merge `8273a09b8`)                                                    |
| Engram / PLE (`LLM_KV_PLE_*`, `LLM_KV_ENGRAM_*`) | **In tree**                                                                        |
| CPU dry-build `libllama.so`                      | **OK** — contains deepseek41 + Engram symbols                                      |
| GGUF `general.architecture`                      | **`deepseek41`** (ISTA IQ2_XXS shard1 verified)                                    |
| `ggml-cuda/dsv4-hc.*`                            | Present                                                                            |
| HIP dsv4-hc                                      | **Via CUDA glob** (`ggml-hip` compiles `../ggml-cuda/*.cu`) — no separate HIP file |
| Vulkan `dsv4_hc_*.comp`                          | Present (optional RADV fallback)                                                   |
| Live `prismalama:rocm` image                     | **Shipped** CPU deepseek41 libllama (+ pre Go binary); HIP optional-fail           |
| Weights IQ2_XXS+Engram                           | **Verified + registered** as `deepseek-v4-flash-local`                             |
| ISTA KV aliases                                  | **In tree** — `kv_source_layer_ids` / `index_source_layer_ids` / candidate_*       |

ISTA-DASLab GGUFs use `general.architecture = deepseek41` (not `deepseek4`).
**Do not alias `deepseek41` → `deepseek4`.**
ISTA also uses flat KV names (`*.kv_source_layer_ids`) vs ggml-org convert
(`*.attention.kv_source_layers`); `deepseek41.cpp` accepts both.

## Upstream source of truth

- ggml-org/llama.cpp **master** carries `deepseek41` + Engram
  (`src/models/deepseek41.cpp`, arch enums, PLE/ENGRAM KVs).
- Tracking: [ggml-org/llama.cpp#30817](https://github.com/ggml-org/llama.cpp/pull/30817).
- Vendored pin: `Makefile.sync FETCH_HEAD=8273a09b8071944b573d91d7a6cbc7b023660bf6`
  (merge commit on branch `fix/deepseek41-port`; keeps JAISIU-2319 block-callback + TurboQuant).

## Controlled bring-up

1. ~~Merge ggml-org master into prismallama.cpp~~ **done** (`8273a09b8`).
2. Rebuild image: `GPU_VENDOR=rocm docker build -f packages/prismalama/Dockerfile.rocm -t prismalama:rocm packages/prismalama`
   (Dockerfile uses `-DOLLAMA_LLAMA_BACKENDS=rocm_v7_2 -DCMAKE_HIP_ARCHITECTURES=gfx1100`).
3. Redeploy `jaisiu-prismalama-local` with the new image.
4. When weights verify: `make deepseek4-register`.
5. Smoke: `ollama run deepseek-v4-flash-local` then broker allowlist
   (already wired: `deepseek-v4-flash-local`).

## Engram file layout (ISTA GSQ-RCO)

- Backbone: four `…-IQ2_XXS-0000N-of-00004.gguf` → `/models/deepseek4/backbone/` (**hardlinks**;
  ollama rejects relative `../` symlinks as insecure paths)
- Engram: `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` beside them (lazy PLE)
- Modelfile `FROM /models/deepseek4/backbone` — Engram not ingested at create time

## Host reality (this box)

- RX 7900 XTX 24 GiB + 110 GiB RAM → experts + Engram mostly mmap/SSD.
- Strata `strata-dsv41` has **no** `hip-rx7900` package — prismalama HIP path is the plan.

## Dry validation (no full weights)

```bash
make deepseek4-port-status
# CPU compile smoke (optional):
#   cmake -B packages/prismalama/llama/llama.cpp/build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release
#   ninja -C .../build-cpu libllama.so
#   strings .../libllama.so | grep deepseek41
# GGUF arch (shard1 sufficient):
#   general.architecture must be deepseek41
```

## Status check

```bash
make deepseek4-port-status
```

## Performance / potato contract (JAISIU-2156)

- **Score:** `scripts/deepseek4-bench.sh` → `/tmp/ds-perf/bench.jsonl` (warm gen tok/s from llama-server `eval_duration`, not broker queue time).
- **ROCm defaults:** `LLAMA_ARG_MOE_CACHE_MIB=3072`, `LLAMA_ARG_FIT_TARGET=768`, `num_gpu=-1`, KV `q4_0`, flash-attn on (see `docker-compose.llm-local.rocm.yml`).
- **VRAM:** active experts + MoE SLRU + KV + dense layers must fit on-card; IQ2 expert bulk and Engram tables stay host/NVMe mmap.
- **BC4 / weightimage:** out of scope for IQ2_XXS — do not route this model through the Go texture streamer.
- **Broker:** thin OpenAI proxy to `:11434`; use `stream: true` without buffering the full completion before the first token.
- **Ready check:** `curl /api/tags` + `ollama ps` shows ~23 GiB GPU resident; avoid health-checking with a full generate (evicts warm state).
