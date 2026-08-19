# AirLLM — Layer-by-Layer HF Inference Engine

> Vendored AirLLM fork for streaming inference of Hugging Face safetensors models.
> Layer-by-layer loading keeps peak VRAM to **one transformer block + KV cache**.
>
> Dispatch: `OLLAMA_USE_AIRLLM=1` enables AirLLM routing; `0` (Arch default) disables it.
> See [`docs/RUNTIME_DISPATCH.md`](../../docs/RUNTIME_DISPATCH.md).

---

## Architecture

```
airllm_runner.py (Ollama HTTP server, port+1)
        │
        ▼
AutoModel.from_pretrained() ──► architecture detection
        │
        ├─ macOS ──► AirLLMLlamaMlx (MLX backend)
        └─ Linux ──► AirLLMBaseModel subclass (PyTorch backend)
                          │
                          ├─ find_or_create_local_splitted_path()
                          ├─ ModelPersister (safetensors / MLX .npz)
                          └─ forward(): layer-by-layer streaming
                               │
                               ├─ load_layer_to_cpu() ← ThreadPoolExecutor (prefetch)
                               ├─ move_layer_to_device() ← accelerate
                               ├─ layer(input) ← PyTorch forward
                               └─ layer.to("meta") + clean_memory()
```

---

## Directory layout

```
src/airllm/
└── air_llm/
    └── airllm/
        ├── __init__.py                 — Platform detection (macOS → MLX, Linux → PyTorch)
        ├── auto_model.py               — AutoModel factory: architecture → class mapping
        ├── airllm_base.py              — AirLLMBaseModel: core streaming engine (27KB)
        ├── airllm.py                   — AirLLMLlama2 (default, no overrides)
        ├── airllm_qwen.py              — AirLLMQWen (custom layer names, RoPE args)
        ├── airllm_qwen2.py             — AirLLMQWen2
        ├── airllm_qwen35.py            — AirLLMQWen35
        ├── airllm_chatglm.py           — AirLLMChatGLM (custom embed/norm/head paths)
        ├── airllm_baichuan.py          — AirLLMBaichuan (custom SentencePiece tokenizer)
        ├── airllm_internlm.py          — AirLLMInternLM
        ├── airllm_mistral.py           — AirLLMMistral
        ├── airllm_mixtral.py           — AirLLMMixtral
        ├── airllm_llama_mlx.py         — AirLLMLlamaMlx (macOS MLX backend, 17KB)
        ├── utils.py                    — Utilities: compression, loading, memory cleanup (18KB)
        ├── profiler.py                 — LayeredProfiler (timing per operation)
        ├── tokenization_baichuan.py    — Baichuan SentencePiece tokenizer (HF bug workaround)
        └── persist/
            ├── model_persister.py      — Abstract ModelPersister base + singleton factory
            ├── safetensor_model_persister.py — PyTorch safetensors backend
            └── mlx_model_persister.py  — MLX/numpy backend (macOS)
```

---

## AirLLMBaseModel (core engine)

### Constructor

```python
AirLLMBaseModel(
    model_local_path_or_repo_id,    # Local path or HF repo ID
    device="cuda:0",                 # Target GPU device
    dtype=torch.float16,             # Weight dtype
    max_seq_len=512,                 # Maximum sequence length
    compression=None,                # "4bit" or "8bit" (bitsandbytes)
    prefetching=True,                # Async layer prefetch via ThreadPoolExecutor
    hf_token=None,                   # HuggingFace API token
    delete_original=False,           # Delete original after splitting
)
```

### Forward pass (layer streaming)

For each layer in the model:

1. **Prefetch** — `ThreadPoolExecutor` loads next layer's safetensors to CPU (async)
2. **Wait** — block until current layer is ready (`future.result()`)
3. **GPU transfer** — `move_layer_to_device()` via `accelerate.set_module_tensor_to_device()`
4. **Compute** — `layer(input)` (embed / transformer block / norm / lm_head)
5. **Cleanup** — `layer.to("meta")` + `clean_memory()` (gc + malloc_trim + cuda.empty_cache)

Peak VRAM: **one transformer block + KV cache + activations**.

### Layer names (customizable per architecture)

```python
{
    "embed": "model.embed_tokens",       # Token embeddings
    "layer_prefix": "model.layers",      # Transformer blocks (0..N)
    "norm": "model.norm",                # Final layer norm
    "lm_head": "lm_head",               # Output projection
    "rotary_pos_emb": (optional),        # Cached rotary embeddings (ChatGLM)
}
```

---

## Model implementations

| Class | Architecture | Customizations |
|-------|-------------|----------------|
| `AirLLMLlama2` | Llama 2/3 | None (default base) |
| `AirLLMQWen` | QWen | Custom layer names, rotary_pos_emb_list, layer_past args |
| `AirLLMQWen2` | QWen2 | Disables BetterTransformer |
| `AirLLMQWen35` | QWen3.5 | Disables BetterTransformer |
| `AirLLMChatGLM` | ChatGLM/GLM4 | Custom embed/norm/head paths, rotary_pos_emb, kv_cache args |
| `AirLLMBaichuan` | Baichuan | Custom SentencePiece tokenizer |
| `AirLLMInternLM` | InternLM | Disables BetterTransformer |
| `AirLLMMistral` | Mistral | Disables BetterTransformer |
| `AirLLMMixtral` | Mixtral | Disables BetterTransformer |
| `AirLLMLlamaMlx` | Llama (macOS) | Entirely MLX-based (no PyTorch) |

### AutoModel factory

```python
from airllm import AutoModel
model = AutoModel.from_pretrained("/path/to/model", compression="4bit")
```

Detection: reads `config.architectures[0]` and maps to the appropriate class.

---

## Persistence layer

| Backend | File format | Platform | Marker |
|---------|------------|----------|--------|
| `SafetensorModelPersister` | `{layer}.safetensors` | Linux | `.safetensors.done` |
| `MlxModelPersister` | `{layer}.mlx.npz` | macOS | `.mlx.done` |

Platform auto-selected via `ModelPersister.get_model_persister()`.

---

## Quantization

When `compression="4bit"` or `"8bit"`:

- Requires `bitsandbytes` installed
- Layers compressed on disk via `compress_layer_state_dict()`
- Decompressed on CPU before GPU transfer via `uncompress_layer_state_dict()`
- Prefetching disabled (incompatible with compression)

---

## NVMe striping (HC-52)

Multi-NVMe parallel reads for workstations with 4+ drives.

**File:** `src/prismalama/nvme_striping.py`

| Component | Purpose |
|-----------|---------|
| `discover_nvme_mounts()` | Read `AIRLLM_NVME_MOUNTS` env var, validate paths |
| `_build_shard_to_nvme_map()` | Map shard files → NVMe mounts (env, manifest, or infer) |
| `NVMEStripingModelPersister` | Wraps SafetensorModelPersister with parallel reads |
| `inject_striped_persister()` | Monkey-patch into airllm.persist before AutoModel import |
| `_parallel_read_test()` | Benchmark concurrent read throughput |

**NUMA-aware routing (HC-50):** `AIRLLM_NUMA_PROXIMITY_MAP` (JSON) maps GPU index → NVMe
mounts sorted by NUMA closeness. Avoids cross-socket memory bandwidth on multi-socket systems.

**Environment:**
- `AIRLLM_NVME_MOUNTS` — colon-separated mount paths (required for striping)
- `AIRLLM_SHARD_MOUNTS` — explicit shard → mount mappings
- `AIRLLM_NUMA_PROXIMITY_MAP` — GPU → NVMe proximity (JSON)

---

## HTTP server (`airllm_runner.py`)

Ollama-compatible HTTP server spawned by the Go `airllmrunner` on `port+1`.

### Endpoints

| Route | Method | Purpose |
|-------|--------|---------|
| `/load` | POST | Load model (runs in daemon thread) |
| `/completion` | POST | Stream token generation (chunked JSON) |
| `/embedding` | POST | Dummy 768-dim embeddings |
| `/health` | GET | Server status + progress |

### Data models

```python
LoadRequest:    { operation, model_path, compression, main_gpu, kv_size, ... }
CompletionRequest: { prompt, options: { num_predict, temperature, top_p, stop } }
CompletionResponse: { content, done, done_reason, eval_count, eval_duration }
```

### Memory cleanup

Post-inference: `finalize_inference_memory()` calls `torch.cuda.synchronize()`,
`torch.cuda.empty_cache()`, `gc.collect()`, and `malloc_trim()`.
Controlled by `AIRLLM_POST_INFER_CLEANUP` (default: `1`).

---

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `AIRLLM_COMPRESSION` | `4bit` | Weight compression (`4bit`, `8bit`, `none`) |
| `AIRLLM_DEVICE` | `cuda:0` | PyTorch device (ROCm uses same API) |
| `AIRLLM_POST_INFER_CLEANUP` | `1` | `0` to skip post-inference GPU cache flush |
| `AIRLLM_NVME_MOUNTS` | unset | Colon-separated NVMe mount paths for striping |
| `AIRLLM_SHARD_MOUNTS` | unset | Shard → mount mappings |
| `AIRLLM_NUMA_PROXIMITY_MAP` | unset | JSON GPU → NVMe proximity map |
| `PRISMALAMA_AIRLLM_PYTHONPATH` | unset | Prepend to PYTHONPATH for dev-tree detection |

---

## Profiling

```python
model = AutoModel.from_pretrained(path, profiling_mode=True)
model(input_ids)
model.profiler.print_profiling_time()
```

Tracks: `load_safe_tensor`, `compression_time`, `pin_memory_to_trigger_load`,
`create_layer_from_state_dict`, `kick_off_load_cpu`.
