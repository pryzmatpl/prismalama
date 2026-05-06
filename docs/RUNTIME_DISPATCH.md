# Runtime dispatch (which engine runs?)

When a model loads, the server spawns a subprocess: `ollama runner --model <path> --port …`. The **`runner`** package chooses:

| Engine | Process | Typical GPU stack |
|--------|---------|-------------------|
| **llama** | GGML / llama.cpp in-process | ROCm HIP, CUDA, Vulkan, Metal, CPU |
| **airllm** | Python `airllm_runner.py` + PyTorch | `AIRLLM_DEVICE` (e.g. `cuda:0` on ROCm) |
| **ollama** | New Ollama engine path | As configured |

Selection is implemented in **`runner/dispatch.go`** (`DecideEngine`). The **Modelfile name** (e.g. `qwopus`) does not select the engine; **on-disk layout** and **env** do (see table below). The **Prismalama Arch package** defaults to **`OLLAMA_USE_AIRLLM=0`**: GGUF inference uses **GGML** without PyTorch/transformers unless the user opts in. **Product principle:** **`docs/PRISMALAMA_PRINCIPLE.md`** (GGML vs AirLLM is the key architectural fact). **Operators:** **`GET /api/prismalama/capabilities`** documents semantics on a running server.

## AirLLM runner: two ports (Go proxy + Python)

The AirLLM subprocess is **`runner/airllmrunner`**: a **Go** HTTP server receives **`llm.LoadRequest`** JSON from the main binary on **`port`**, and forwards **snake_case** JSON to **`airllm_runner.py`** on **`port+1`**. The Python process must **not** share the Go port (older bugs posted the proxy payload back into the Go `/load` handler and caused **`bad request`**). **`waitForReady`** polls **Python** `/health` on `port+1`.

**Completion gating:** the Go proxy uses **`sync.WaitGroup`**: **`NewServer`** does **`Add(1)`** so **`completion`** blocks until the **first** **`LoadOperationCommit`** finishes (**`Done()`** on every path, including Python **`success: false`**). **Reload** (a **second** Commit on the same process) must **`Add(1)`** before **`Done()`** or the WaitGroup **panics** (`negative WaitGroup counter`), the **runner subprocess exits**, and the server looks **stuck** / **no GPU** — handled in **`runner/airllmrunner/runner.go`** with **`commitReturnedOnce`** + **`commitMu`**.

## Logs (after recent changes)

On each runner start you should see a line like:

```text
runner dispatch engine=llama model=/path/to/...
runner dispatch engine=airllm model=/path/to/... reason=OLLAMA_USE_AIRLLM
```

**`reason`** (AirLLM only) is one of:

- `OLLAMA_USE_AIRLLM` — env set to `1` or `true` (see `/etc/default/ollama` in the Arch package).
- `OLLAMA_MULTI_GGUF=1`
- `model.safetensors.index.json` / `safetensors_shards` / `config.json_hf_heuristic`
- `multipart_gguf` — `*-00001-of-*.gguf` present

If **`OLLAMA_USE_AIRLLM=1`** is set globally, **every** model path is routed to AirLLM unless you clear it — even for a single-file `.gguf` tree.

**Disabling AirLLM:** **`OLLAMA_USE_AIRLLM=0`**, **`false`**, or **`no`** turns **off** all AirLLM routing, including **`multipart_gguf`** and **`OLLAMA_MULTI_GGUF`**. (Previously, multi-shard GGUF could still select AirLLM even when you thought AirLLM was disabled.) After that, loads use **GGML** (`engine=llama`) so ROCm/Vulkan can drive the GPU.

## Manual checks

1. **Process list** — AirLLM: `python3` with `airllm_runner.py`. Llama: no Python child; `ollama runner` only.
2. **Server log** — `starting runner` plus `runner dispatch` (see above).
3. **`curl http://127.0.0.1:11434/api/ps`** — loaded model names (not engine type; use logs for engine).

## Load failures and API messages

- **`model load failed: …`** — `llm.LoadResponse` included an **`error`** string from the runner (AirLLM/Python, HTTP body from Python, or **`ErrNoMem`** text from the GGML runner). Prefer this over the generic messages below when present.
- **`failed to allocate memory for model`** / **`failed to commit memory for model`** — commit returned **`success: false`** without a more specific **`error`** field (often OOM; see logs **`load commit rejected by runner`** / **`runner_error`**).

## HTTP 500 and “bad request”

Runners (`runner/*/runner.go`) return **400** with plain text **`bad request`** when the runner fails to **JSON-decode** a `/load` or `/completion` body (malformed request). The main server maps that to **`api.StatusError`** with the same status code.

If you see **`500`** with message **`bad request`**, check **`journalctl -u ollama`** for **`llm load error:`** / **`llm predict error:`** / **`airllm load JSON decode`** (raw bytes). Common causes: **version skew** between server and model blobs, **OOM** / runner crash (**EOF**), or a **too-large / invalid** prompt. The server also coerces **`status`** on `gin.H` errors so numeric types (e.g. `float64`) do not incorrectly force **500**.

## Ollama engine models (e.g. `qwen3next`, `qwen3-coder-next`)

Architectures listed in `fs/ggml/ggml.go` **`OllamaEngineRequired()`** use the **`--ollama-engine`** runner. The server must still **estimate per-layer sizes from GGUF** before the first `createLayout`; otherwise the scheduler sees **all-zero** layer weights and **GPU offload can be wrong or empty** (no VRAM growth). That path is handled in **`llm/server.go`** via **`prepareMemoryEstimateFromGGML`** shared with the llama.cpp loader.

## GPU usage (AirLLM vs GGML)

Two different stacks:

| Path | What uses the GPU | If VRAM stays idle |
|------|---------------------|---------------------|
| **llama** / **ollama** (GGML) | `OLLAMA_LIBRARY_PATH` → HIP/Vulkan `.so` under `/usr/lib/ollama/rocm` | The scheduler packs **as many transformer layers as fit** into reported free VRAM (minus **`OLLAMA_GPU_OVERHEAD`**). Logs include **`ggml GPU layer offload`** with **`offload_percent`** and **`use_mmap`**. If VRAM looks **tiny** but the model runs (slowly), offload is likely **low** — check **`num_gpu`** in the Modelfile, discovery, and **`HIP_VISIBLE_DEVICES`**. On **Linux**, mmap is **off** when free system RAM is below the estimated model size (avoids thrashing); for large GGUF on **fast NVMe**, set **`OLLAMA_MMAP_ALLOW_LOW_RAM=1`** so weights can **mmap** from disk (`llm/server.go`). **Vulkan** backends are skipped unless **`OLLAMA_VULKAN=1`**; HIP does not need it. |
| **airllm** (Python) | **PyTorch** with ROCm; **`AIRLLM_DEVICE`** (default **`cuda:0`** — ROCm uses the CUDA API name) | **CPU-only PyTorch** (no `torch.cuda.is_available()`). On Arch install **`python-pytorch-rocm`** (or your distro’s ROCm build), not plain **`python-pytorch`**. After restart, logs should show **`PyTorch: cuda.is_available=True`**. |

`/etc/default/ollama` from the package sets **`HIP_VISIBLE_DEVICES`** and **`AIRLLM_DEVICE=cuda:0`**; override only if you know your layout.

## Rebuild

After changing **`runner/airllmrunner`**, **`llm/`**, or **`runner/runner.go`**, rebuild and reinstall (**e.g. `sudo makepkg -sfi`**) and **`sudo systemctl restart ollama`** so **`/usr/bin/ollama`** and packaged **`airllm_runner.py`** match the tree.

## Related

- `docs/WEIGHT_STREAMING_STRATEGY.md` — GGML vs AirLLM product tradeoffs.
- `docs/GOAL-GAPS.md` — shipped defaults vs goals (routing, streaming env).
- `llm/server.go` — mmap defaults; Vulkan disables mmap unless overridden; Linux may disable mmap when RAM is tight unless **`OLLAMA_MMAP_ALLOW_LOW_RAM`**.
