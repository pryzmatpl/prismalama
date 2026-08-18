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

`POST /api/generate` `num_predict=8` `num_ctx=256` at 2026-08-18T10:08:21Z returned **insufficient memory** in ~2 s. The ollama-engine generate client (`NewOllamaEngineServer`) currently assigns **every** repeating layer plus output to GPU 0 (`gpuLayersForEngine` auto = 100% offload) and `POST /load` commit fails before `LoadStreaming` / keep-evict can run.

This is a real gap, not a skipped test: a 36.9 GB GGUF cannot start on this 24 GiB card until load layout + streaming alloc reserve less than full weight VRAM.

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

# larger than VRAM (expected OOM until layout/streaming alloc is fixed)
curl -sS http://127.0.0.1:11434/api/generate -d \
  '{"model":"qwen35-uncensored:latest","prompt":"Say hi in one word.","stream":false,"options":{"num_predict":8,"num_ctx":256}}'

docker logs --since 10m jaisiu-prismalama 2>&1 | grep -c 'kept block'
docker logs --since 10m jaisiu-prismalama 2>&1 | grep -c 'evicted block'
```
