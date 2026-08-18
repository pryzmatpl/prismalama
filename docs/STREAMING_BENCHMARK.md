# STREAMING_BENCHMARK — RX 7900 XTX (host `prizm`)

JAISIU-2299. Measured 2026-08-18 on the live `jaisiu-prismalama` container.
No llama.cpp numbers are invented: this image does not ship `llama-server`.

## Host

| Item | Value |
| --- | --- |
| GPU | AMD Radeon RX 7900 XTX (`0x744c`), gfx1100 |
| PCI | `0000:0b:00.0` |
| VRAM | 25 753 026 560 B (24.0 GiB) |
| Host RAM (Prismalama log) | 110.0 GiB |
| Serve binary | `ollama-xtx-engine` (Ubuntu, `/usr/bin/ollama`) |
| Backend | ROCm / HIP overlay + `rocr_bdf_shim.so` |
| Env | `OLLAMA_LAYER_STREAMING=1`, `OLLAMA_STREAMING_BUDGET=4GiB` (default), `OLLAMA_MAX_LOADED_MODELS=1`, `HIP_VISIBLE_DEVICES=0` |

`rocm-smi` (container `/opt/rocm/bin/rocm-smi`) with `qwen3:0.6b` loaded after generate:

```
GPU[0]  GPU use (%): 4
VRAM Total Memory (B): 25753026560
VRAM Total Used Memory (B): 3471298560   # ~3.23 GiB
```

After a failed >VRAM load, idle used VRAM dropped to 1 980 350 464 B; reloading `qwen3:0.6b` used 3 389 014 016 B (~3.16 GiB).

## Pins

| Model | On-disk size | Role |
| --- | --- | --- |
| `qwen3:0.6b` | 0.52 GB | Fits VRAM; keep-resident |
| `qwen35-uncensored:latest` | 36.90 GB | Larger than 24 GiB VRAM; must stream |

## `qwen3:0.6b` (keep-resident)

`POST /api/generate` `stream=false` `num_predict=32` `temperature=0` prompt `Write a haiku about GPUs.` (model already loaded).

| Metric | Value |
| --- | --- |
| `prompt_eval_count` | 15 |
| `prompt_eval_duration` | 22 438 761 ns → **668.5 tok/s** prompt |
| `eval_count` | 32 |
| `eval_duration` | 272 906 130 ns → **117.3 tok/s** generate |
| `load_duration` | 382 313 469 ns (already resident) |
| `total_duration` | 703 364 358 ns |
| GPU offload | 29/29 layers (`ggml GPU layer offload` 100%) |
| Streamer | `streaming inference: initialized` blocks=28 `total_weight_bytes=516688896` (~493 MiB) |

Logs since the 10:00 UTC binary swap (`docker logs --since 2026-08-18T10:00:00Z`):

- `streaming inference: kept block` = **1904**
- `streaming inference: evicted block` = **0**
- budget in those lines: `budget_bytes=4294967296` (4 GiB)

Warm 16-token generate on the same binary earlier the same morning: **106.1 tok/s** (`eval_count=16`, `eval_duration=150833404`).

## `qwen35-uncensored:latest` (>VRAM)

Architecture `qwen35moe` (Qwen3.5-35B-A3B Q8_0, 36.90 GB on disk, 40 repeating layers + output).

`gpuLayersForEngine` packs a VRAM-fitting tail of **routed-expert** tensors (llama.cpp `-ngl` convention) and pins attn/GDN/shared-expert of every repeating layer on GPU (`moe_split=true`). Evict-by-zero is disabled when `OLLAMA_STREAMING_BUDGET>0` because it does not free HIP buffers.

Measured 2026-08-18T11:49Z after MoE split offload:

| Item | Value |
| --- | --- |
| GPU compute | all 40 repeating layers (GDN/attn) + output |
| GPU routed experts | **24/41** (`moe_pin_non_expert_bytes=1585932800` ≈ 1.48 GiB) |
| CPU routed experts | 17 repeating layers (`mul_mat_id` on host; batch < 32 so no HIP op-offload) |
| Load | ~12 s (HTTP `load_duration`) |
| Streamer | `keep-resident after full load` `budget_bytes=4294967296` |
| Cold `num_predict=8` | **9.44 tok/s** (`eval_count=8`, `eval_duration=0.847 s`) |
| Warm `num_predict=16` | **7.68 tok/s** (`eval_count=16`, `eval_duration=2.082 s`) |
| Warm prompt | 6 tokens → **20.7 tok/s** |
| `rocm-smi` VRAM used | 25 672 425 472 B (~24.7 GiB of 24.0 GiB advertised) |

Previous same-day full-layer tail (26/41, GDN of layers 0–14 on CPU) was **~2 tok/s**. Split layout is ~4× on this host. Decoded text is still not coherent.

Q8 MoE expert tensors are ~31.9 GiB of the 36.9 GB file; non-expert repeating weights are ~1.48 GiB. Full-layer HIP paging cannot beat PCIe; the next win is gathering the **8 active experts** (~25 MiB/layer) onto GPU rather than running those 17 CPU GEMMs.

### Earlier 2026-08-18T10:32Z (full-layer tail, before MoE split)

| Item | Value |
| --- | --- |
| GPU offload | **26/41 layers** (`15..40` + output) |
| Warm 16-token generate | **1.54–2.12 tok/s** |
| `rocm-smi` VRAM used | 25 431 359 488 B |

### Follow-up 2026-08-18T10:52Z (IMRoPE + GDN gate)

`qwen3next` now matches llama.cpp `LLAMA_ROPE_TYPE_IMROPE`: `rope.dimension_sections=[11,11,10,0]`, `rope_dim=64`, 4-axis position IDs. GDN `ssm_alpha` gate is reshaped to `[1, H, T, B]` before chunked delta-net (the previous `[H,T,B]` permute scrambled the time axis on prefill).

Log: `qwen3next: IMRoPE sections="[11 11 10 0]" rope_dim=64 head_dim=256`.

Warm-ish 16-token generate after reload (`eval_duration` 7.53 s): **2.12 tok/s**. Decoded text changed vs NeoX (so RoPE is live) but is still not English. Next suspects: remaining GDN chunked-vs-llama.cpp differences, mixed CPU/GPU residual dtypes, MoE routing.

`ollama-engine` `Tokenize` no longer returns dummy `[0,1,2,…]`; it calls the runner `/tokenize` using the model BPE.

Earlier the same morning (10:08Z) the same tag returned **insufficient memory** in ~2 s because auto offload assigned every layer to GPU 0.

## vs stock llama.cpp

**Not measured.** `FindLlamaServer()` fails on this image (no `llama-server` under `/usr/lib/ollama`). Generate uses `runner --ollama-engine`. Do not treat the 117 tok/s figure as a llama.cpp number.

## vs always-evict on the same tiny model

Not re-measured on this binary with `OLLAMA_STREAMING_BUDGET=0` (that needs a container recreate). The keep-resident logs (`evicted=0` with a 4 GiB budget and ~493 MiB weights) are the evidence that P1-2 is active for models that fit.

## How to reproduce

```sh
# GPU
docker exec jaisiu-prismalama /opt/rocm/bin/rocm-smi --showproductname --showmeminfo vram --showuse

# tiny, keep-resident
curl -sS http://127.0.0.1:11434/api/generate -d \
  '{"model":"qwen3:0.6b","prompt":"Write a haiku about GPUs.","stream":false,"options":{"num_predict":32,"temperature":0}}'

# larger than VRAM — partial GPU offload (tail layers)
curl -sS -m 1800 http://127.0.0.1:11434/api/generate -d \
  '{"model":"qwen35-uncensored:latest","prompt":"Say hi in one word.","stream":false,"options":{"num_predict":8,"num_ctx":256,"temperature":0}}'

docker logs --since 10m jaisiu-prismalama 2>&1 | grep -c 'kept block'
docker logs --since 10m jaisiu-prismalama 2>&1 | grep -c 'evicted block'
```
