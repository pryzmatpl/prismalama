# runner/ — Engine Dispatch and Runner Implementations

> Routes model requests to the correct inference engine and manages subprocess lifecycle.
> Runtime dispatch docs: [`docs/RUNTIME_DISPATCH.md`](../docs/RUNTIME_DISPATCH.md).

## Dispatch (`dispatch.go`)

Single testable decision point: GGML (llama.cpp) vs AirLLM (Python).

```go
type EngineKind int
const (
    EngineGGML   EngineKind = iota  // llama.cpp / prismallama.cpp
    EngineAirLLM                     // Python PyTorch layer streaming
)

func DecideEngine(modelPath string) (EngineKind, string)
func DecideEngineDetailed(modelPath string) EngineDecision  // Full trace with reasons
```

**Decision rules (evaluated in order):**

1. Empty path → GGML
2. `OLLAMA_USE_AIRLLM=0/false/no` → GGML (explicit opt-out)
3. `OLLAMA_MULTI_GGUF=1` → AirLLM
4. Path missing → GGML
5. `model.safetensors.index.json` exists → AirLLM
6. `*.safetensors` files exist → AirLLM
7. `config.json` with safetensors/torch_dtype hints → AirLLM
8. `*-00001-of-*.gguf` multipart pattern → AirLLM
9. `OLLAMA_USE_AIRLLM=1/true` → AirLLM (explicit opt-in)
10. Default → GGML

Arch package default: `OLLAMA_USE_AIRLLM=0` (GGML-only unless opt-in).

---

## Runners

### `llamarunner/` — llama.cpp GGUF Runner

Standard GGUF inference via CGo llama.cpp bindings.

| File                  | Purpose                                                      |
| --------------------- | ------------------------------------------------------------ |
| `runner.go`           | Sequence-based inference loop, batch processing, KV cache    |
| `streaming_policy.go` | `useMmapWithLayerStreaming()` — LoRA constraint, mmap policy |

**Key types:**

- `Sequence` — per-request state: inputs, sampling context, stop sequences, metrics
- `NewSequence()` — validates input length, handles truncation, numKeep calculation

```shell
./runner -model <model.gguf>

curl -X POST -H "Content-Type: application/json" \
  -d '{"prompt": "hi"}' http://localhost:8080/completion

curl -X POST -H "Content-Type: application/json" \
  -d '{"prompt": "turn me into an embedding"}' http://localhost:8080/embedding
```

### `ollamarunner/` — Ollama Engine Runner

New-architecture runner using `runner --ollama-engine` path.

| File            | Purpose                  |
| --------------- | ------------------------ |
| `runner.go`     | Ollama-engine subprocess |
| `multimodal.go` | Multimodal cache         |

### `airllmrunner/` — AirLLM Python Proxy (1,188 lines)

Go HTTP proxy coordinating the Python AirLLM subprocess.

```
Go proxy (port N) ←→ Python airllm_runner.py (port N+1)
```

**GPU/NUMA discovery:**

```go
func discoverGPUTopology() (*GPUTopology, error)       // rocm-smi + nvidia-smi
func discoverNUMANodes() (map[int]NUMANodeInfo, error)  // numactl + /sys/
func mapNVMEToNUMA() (map[string]int, error)            // /sys/block/nvme*/device/numa_node
func buildGPUToNVMEProximityMap() (map[int][]string, error) // Score by NUMA distance
```

**Server lifecycle:**

1. `startPythonRunner()` — discover topology, build env, spawn `python3 airllm_runner.py`
2. `waitForReady()` — poll `/health` for 30s
3. `load()` — proxy `LoadRequest` to Python `/load` (snake_case JSON)
4. `completion()` — proxy `CompletionRequest`, stream chunked JSON responses
5. `health()` — structured health with PID, consecutive errors, ReadyForInference

**Environment set by runner:**

- `ROCR_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` — GPU visibility
- `AIRLLM_NUMA_PROXIMITY_MAP` — GPU → NVMe proximity (HC-50)
- `AIRLLM_MULTI_GPU` — multi-GPU flag (HC-49)
- `PYTHONPATH` — dev checkout or `/usr/share/ollama/airllm`

**Circuit breaker:** after 5 consecutive errors, runner is marked unhealthy.

### `common/` — Shared Utilities

Stop sequence detection, Unicode handling.

---

## Subprocess entry

`cmd/runner/main.go` → `runner.Execute()` — spawned by the main server process.

The server dispatches based on `DecideEngine()`:

- `EngineGGML` → `llamarunner`
- `EngineAirLLM` → `airllmrunner`
- Ollama-engine models → `ollamarunner` (via `--ollama-engine` flag)
