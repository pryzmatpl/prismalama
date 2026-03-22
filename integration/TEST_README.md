# Prismalama Test Suite

This directory contains comprehensive tests for the Prismalama project, which unifies ROCm, AirLLM (sharding/streaming), and Ollama for high-performance inference.

## Test Categories

### Integration Tests

Integration tests are organized by build tags to allow targeted testing:

| Tag | Description | Example Command |
|-----|-------------|-----------------|
| `integration` | Basic integration tests | `go test -tags=integration ./integration` |
| `integration,airllm` | AirLLM-specific tests | `go test -tags=integration,airllm ./integration` |
| `integration,gpu` | GPU utilization tests | `go test -tags=integration,gpu ./integration` |
| `integration,sharding` | Large model sharding tests | `go test -tags=integration,sharding ./integration` |
| `integration,hardware` | Hardware detection tests | `go test -tags=integration,hardware ./integration` |
| `integration,minimax` | MiniMax model tests | `go test -tags=integration,minimax ./integration` |
| `integration,perf` | Performance benchmarks | `go test -tags=integration,perf ./integration` |

## Coverage

See **`docs/DEVELOPER.md`** (“Integration tests and coverage”) for commands. Example:

```bash
go test -tags=integration ./integration -coverprofile=/tmp/integration.cov -covermode=atomic -timeout 10m
go tool cover -func=/tmp/integration.cov | tail -20
```

Coverage is **partial** when tests skip (missing models, GPU, or env flags).

## Ship gate (release bar)

After a feature lands, run integration proof then build **`prismalama-ollama`**: **`make ship-check`** (or **`scripts/ship-check.sh`**). Quick loop without packaging: **`make ship-check-fast`**. See **`docs/DEVELOPER.md`** (§ Ship gate).

## Running Tests

### Prerequisites

1. Build the project first:
   ```bash
   go build .
   ```

2. For ROCm/GPU tests, ensure ROCm is installed:
   ```bash
   rocm-smi  # Verify ROCm is working
   ```

3. For AirLLM tests, set the environment variable:
   ```bash
   export OLLAMA_TEST_AIRLLM=1
   ```

### Quick Test

Run basic integration tests:
```bash
go test -tags=integration ./integration -v -timeout 5m -run TestBlueSky
```

### AirLLM Tests

```bash
# Enable AirLLM testing
export OLLAMA_TEST_AIRLLM=1
export OLLAMA_AIRLLM_MODEL="Qwen2.5-Coder-32B-Instruct"

# Run AirLLM tests
go test -tags=integration,airllm ./integration -v -timeout 30m
```

### GPU Utilization Tests

```bash
# Run GPU tests
go test -tags=integration,gpu ./integration -v -timeout 15m

# Run with specific model
export OLLAMA_TEST_DEFAULT_MODEL="llama3.2:3b"
go test -tags=integration,gpu ./integration -v -timeout 15m
```

### Sharding Tests

```bash
# Run sharding tests (requires large RAM)
go test -tags=integration,sharding ./integration -v -timeout 30m
```

### Hardware Detection Tests

```bash
go test -tags=integration,hardware ./integration -v -timeout 10m
```

### MiniMax Model Tests

```bash
# Ensure MiniMax model is available
ls -la "/nvme3/AI Models/MiniMaxM2.5"

# Run MiniMax tests
go test -tags=integration,minimax ./integration -v -timeout 60m
```

### All Integration Tests

```bash
go test -tags=integration,airllm,gpu,sharding,hardware ./integration -v -timeout 90m
```

## Test Output Format

Tests output performance metrics in a parseable format:

```
GPU_PERF: model=llama3.2:1b tps=45.23 duration=2.3s vram_used=2.1GiB
AIRLLM_PERF: model=Qwen2.5-Coder-32B total_time=45s eval_tokens=200 eval_tps=4.44
KV_CACHE: type=f16 tps=42.15 total_time=1.2s response_len=256
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OLLAMA_TEST_AIRLLM` | Set to "1" to enable AirLLM tests |
| `OLLAMA_AIRLLM_MODEL` | Specify the AirLLM model to test |
| `OLLAMA_TEST_EXISTING` | Set to use existing running server |
| `OLLAMA_TEST_DEFAULT_MODEL` | Override default test model |
| `OLLAMA_MULTI_GPU` | Set to "1" to enable multi-GPU tests |
| `AIRLLM_COMPRESSION` | Set compression mode (4bit, 8bit, none) |
| `OLLAMA_USE_AIRLLM` | Force AirLLM mode for models |
| `OLLAMA_MAX_VRAM` | Maximum VRAM for tests |

## Test Files

### airllm_test.go
Tests for AirLLM integration:
- Basic generation
- Code generation
- Chat completion
- Long context handling
- Streaming responses
- Compression modes (4bit, 8bit)
- Model detection
- Performance metrics

### gpu_test.go
Tests for GPU utilization:
- GPU utilization monitoring
- Layer offloading
- Model fitting
- numGPU option testing
- Multi-batch processing
- Flash attention
- Large model sharding
- NVMe model loading

### sharding_test.go
Tests for large model sharding:
- Large GGUF model offloading
- Layer-by-layer inference
- Hybrid CPU offload
- Context window scaling
- KV cache optimization
- Multi-GPU layer distribution
- Batch size optimization

### hardware_test.go
Tests for hardware detection:
- ROCm detection
- VRAM detection
- System memory detection
- CPU info
- NVMe performance
- Optimal settings recommendation
- GPU layer allocation
- Flash attention support

### minimax_test.go
Tests for MiniMax M2.5 model:
- Model detection
- Model size verification
- Basic inference
- Chat completion
- Long context handling
- Code generation
- Memory efficiency with 4-bit

### runner_test.go (unit tests)
Tests for runner logic:
- AirLLM model detection
- Environment variable handling
- Model path extraction

## Performance Testing

Run performance benchmarks:
```bash
go test -tags=integration,perf ./integration -v -timeout 90m -run TestModelsPerf
```

Output can be captured and analyzed:
```bash
go test -tags=integration,perf ./integration -v -timeout 90m 2>&1 | tee perf.log
grep "MODEL_PERF_DATA" perf.log | cut -f2- -d: > perf.csv
```

## Hardware Requirements

### Minimum
- 16GB RAM
- 8GB VRAM (for GPU tests)
- 50GB free disk space

### Recommended
- 64GB+ RAM (for sharding tests)
- 24GB VRAM (for large model tests)
- 500GB+ NVMe storage (for large models)

## Troubleshooting

### Tests timeout
Increase timeout: `go test -timeout 60m ...`

### ROCm not detected
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
rocm-smi
```

### Memory issues
For large model tests, ensure sufficient swap:
```bash
free -h
```

### Model not found
Check model paths:
```bash
ls -la /nvme3/ollama-models/
ls -la "/nvme3/AI Models/"
```
