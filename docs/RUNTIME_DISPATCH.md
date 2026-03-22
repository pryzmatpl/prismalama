# Runtime dispatch (which engine runs?)

When a model loads, the server spawns a subprocess: `ollama runner --model <path> --port …`. The **`runner`** package chooses:

| Engine | Process | Typical GPU stack |
|--------|---------|-------------------|
| **llama** | GGML / llama.cpp in-process | ROCm HIP, CUDA, Vulkan, Metal, CPU |
| **airllm** | Python `airllm_runner.py` + PyTorch | `AIRLLM_DEVICE` (e.g. `cuda:0` on ROCm) |
| **ollama** | New Ollama engine path | As configured |

Selection is implemented in `runner/runner.go` (`airLLMModelAndReason`).

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

## Related

- `docs/WEIGHT_STREAMING_STRATEGY.md` — GGML vs AirLLM product tradeoffs.
- `llm/server.go` — Vulkan disables mmap by default (`UseMmap`).
