# Runtime dispatch (which engine runs?)

When a model loads, the server spawns a subprocess: `ollama runner --model <path> --port …`. The **`runner`** package chooses:

| Engine | Process | Typical GPU stack |
|--------|---------|-------------------|
| **llama** | GGML / llama.cpp in-process | ROCm HIP, CUDA, Vulkan, Metal, CPU |
| **airllm** | Python `airllm_runner.py` + PyTorch | `AIRLLM_DEVICE` (e.g. `cuda:0` on ROCm) |
| **ollama** | New Ollama engine path | As configured |

Selection is implemented in `runner/runner.go` (`airLLMModelAndReason`). The **Modelfile name** (e.g. `qwopus`) does not select the engine; **on-disk layout** and **env** do (see table below).

## AirLLM runner: two ports (Go proxy + Python)

The AirLLM subprocess is **`runner/airllmrunner`**: a **Go** HTTP server receives **`llm.LoadRequest`** JSON from the main binary on **`port`**, and forwards **snake_case** JSON to **`airllm_runner.py`** on **`port+1`**. The Python process must **not** share the Go port (older bugs posted the proxy payload back into the Go `/load` handler and caused **`bad request`**). **`waitForReady`** polls **Python** `/health` on `port+1`.

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

## Rebuild

After changing **`runner/airllmrunner`**, **`llm/`**, or **`runner/runner.go`**, rebuild the installable artifact (**`makepkg -sf`** / **`./build-rocm.sh`**) and **`sudo systemctl restart ollama`** so `/usr/bin/ollama` matches the tree.

## Related

- `docs/WEIGHT_STREAMING_STRATEGY.md` — GGML vs AirLLM product tradeoffs.
- `llm/server.go` — Vulkan disables mmap by default (`UseMmap`).
