# Prismalama

Powered-up Ollama with Vulkan-accelerated weight streaming for GGUF models.

## Overview

Prismalama is an enhanced version of Ollama designed to stream weights of local LLM models in GGUF format to the GPU using Vulkan for fast, optimal inference. It implements weight streaming inspired by AirLLM to handle models larger than VRAM while providing multiple runner interfaces to support diverse architectures.

The system uses Vulkan to avoid fragmentation issues between CUDA and ROCm, providing hardware-agnostic GPU acceleration. Prismalama is packaged as a single Arch Linux package for easy deployment.

## Key Features

### Core Capabilities
- **Vulkan-based weight streaming** for efficient GPU utilization across AMD, NVIDIA, and Intel GPUs
- **GGUF format support** with intelligent weight streaming from storage to GPU
- **Multiple runner interfaces** supporting different inference backends
- **Hardware-agnostic inference** using Vulkan to avoid CUDA/ROCm fragmentation
- **Inspired by AirLLM** for streaming weights larger than VRAM capacity
- **Arch Linux packaging** for streamlined installation and updates

### For Developers
- Technical layout, runners, tests, and memory behavior: [docs/DEVELOPER.md](docs/DEVELOPER.md); for automation, see [AGENTS.md](AGENTS.md).
- **GGUF engine:** [prismallama.cpp](https://github.com/piotroxp/prismallama.cpp) is the maintained fork of llama.cpp/ggml; sync into this repo via `Makefile.sync` and [llama/README.md](llama/README.md).
- Vulkan backend with compute shader optimizations
- Modular runner system supporting Llama.cpp, AirLLM, and custom backends
- Comprehensive test coverage for core components (attention mechanisms, device capabilities, quantization)
- Clear separation between model loading, weight streaming, and inference execution
- Extensible architecture for adding new model architectures and backends

### For Users
- Single Arch Linux package (`prismalama-ollama`)
- Runs large language models exceeding VRAM capacity (tested with MiniMax2.5 Q4)
- Optimized performance for both large and standard models
- Reduced complexity compared to managing CUDA/ROCm dependencies
- Designed for robustness and high-throughput inference
- Compatible with existing Ollama workflows and tools

## Architecture

Prismalama enhances Ollama by integrating Vulkan-accelerated weight streaming:

1. **Weight Streaming Layer**: Inspired by AirLLM, streams model weights from storage to GPU as needed
2. **Vulkan Backend**: Hardware-agnostic GPU compute using Vulkan API with compute shaders
3. **Runner Interface**: Pluggable system supporting multiple inference methods (llama.cpp, AirLLM, etc.)
4. **Memory Management**: Efficient VRAM utilization with dynamic weight loading/offloading
5. **Model Handling**: GGUF format parsing with intelligent sharding for multi-part models

## Installation

Prismalama is distributed as an Arch Linux package:

```bash
# Install the prismalama package
pacman -U prismalama-ollama-*.pkg.tar.zst

# Start the service
systemctl start prismalama-ollama

# Or run directly
prismalama serve
```

## Usage

Standard Ollama commands work with Prismalama:

```bash
# Pull a model
prismalama pull minimax2.5:q4

# Run a model
prismalama run minimax2.5:q4 "Explain quantum computing"

# API access (same as Ollama)
curl http://localhost:11434/api/generate -d '{
  "model": "minimax2.5:q4",
  "prompt": "Explain quantum computing"
}'
```

## Performance & Testing

- **Test Coverage**: Unit tests for attention mechanisms (`ml/attention/paged_attention_test.go`), device capabilities (`ml/device_capability_test.go`), quantization (`ml/quantization/quantization_test.go`), and weight streaming integration (`integration/weight_streaming_test.go`)
- **Benchmarks**: Performance validated against Llama, Qwen, and MiniMax model families
- **Hardware Compatibility**: Tested on AMD (RADV), NVIDIA, and Intel GPUs via Vulkan
- **Memory Efficiency**: Demonstrated ability to run models larger than available VRAM through weight streaming
- **Vulkan Optimization**: Compute shader optimizations for attention and MLP operations

## Development

### Building from Source

```bash
# Clone repository
git clone https://github.com/yourorg/prismalama.git
cd prismalama

# Build package
makepkg -s

# Install locally
sudo pacman -U prismalama-ollama-*.pkg.tar.zst
```

### Running Tests

```bash
# Run unit tests
go test ./...

# Run specific test suites
go test ./ml/attention/...
go test ./ml/nn/...
go test ./integration/weight_streaming_test.go
go test ./ml/device_capability_test.go
```

### Vulkan Development

Prismalama includes a Vulkan backend located in `ml/backend/vulkan/`:
- `kernels.go`: Core Vulkan backend implementation
- Vulkan compute pipelines for attention, MLP, and flash attention operations
- Memory pooling for efficient GPU memory management
- Kernel caching for reusable compute pipelines

## Roadmap

- [x] Vulkan-based weight streaming prototype
- [x] Basic GGUF model support
- [x] Arch Linux packaging
- [x] Weight streaming integration tests
- [ ] MiniMax2.5 Q4 optimization and validation
- [ ] Multi-GPU scaling with Vulkan
- [ ] Additional quantization format support (Q2, Q3, Q5, Q6, Q8)
- [ ] Vulkan compute shader optimizations
- [ ] Runner interface expansion for specialized backends
- [ ] Cross-platform Vulkan validation layers integration

## Model Support

Prismalama supports GGUF-format models including:
- Llama series (Llama 2, 3, 3.1, 3.2, 3.3, 4)
- Qwen series (Qwen, Qwen2, Qwen2.5, Qwen3, Qwen3.5)
- MiniMax series (MiniMax, MiniMax2, MiniMax2.5)
- Mistral series
- Phi series
- Gemma series
- And many others through GGUF compatibility

## License

Prismalama is released under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Built upon [Ollama](https://github.com/ollama/ollama)
- Inspired by [AirLLM](https://github.com/AIR-ML/AirLLM) for weight streaming concepts
- Uses [Vulkan](https://www.vulkan.org/) for cross-platform GPU acceleration
- GGUF format support via [prismallama.cpp](https://github.com/piotroxp/prismallama.cpp) (fork of [llama.cpp](https://github.com/ggml-org/llama.cpp))
- Vulkan compute shader foundations from [ggml-vulkan](https://github.com/piotroxp/prismallama.cpp/tree/master/ggml/src/ggml-vulkan) (upstream: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp))