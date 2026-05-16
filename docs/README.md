# Prismalama documentation

![Prismalama Logo](../logo.jpg)

This tree extends upstream Ollama with **Prismalama-specific** runtime dispatch (GGML vs AirLLM), **layer streaming** hooks for GGUF, Vulkan/ROCm-focused packaging, and **`GET /api/prismalama/capabilities`**.

## Prismalama — read first

| Document | Audience |
|----------|----------|
| [PRISMALAMA_PRINCIPLE.md](./PRISMALAMA_PRINCIPLE.md) | Architecture north star: two engines, honest semantics |
| [RUNTIME_DISPATCH.md](./RUNTIME_DISPATCH.md) | Which runner runs; logs; mmap/Vulkan/HIP; AirLLM two-port proxy |
| [WEIGHT_STREAMING_STRATEGY.md](./WEIGHT_STREAMING_STRATEGY.md) | Product tradeoffs for “streaming” |
| [GOAL-GAPS.md](./GOAL-GAPS.md) | Goals vs current gaps and defaults |
| [DEVELOPER.md](./DEVELOPER.md) | Repo layout, env vars, tests, Docker, prismallama.cpp sync |

## User / operator

| Document | Notes |
|----------|-------|
| [../README.md](../README.md) | Quick start, configuration |
| [../README-PKGBUILD.md](../README-PKGBUILD.md) | Arch package build and `/etc/default/ollama` |
| [../INSTALL.md](../INSTALL.md) | Install pointers (Arch vs source) |
| [../SECURITY.md](../SECURITY.md) | Network exposure and reporting |

## API

| Document | Notes |
|----------|-------|
| [api.md](./api.md) | REST reference (upstream-shaped); includes Prismalama capabilities endpoint |
| [examples.md](./examples.md) | Examples |
| [modelfile.mdx](./modelfile.mdx) | Modelfile syntax |

Upstream API reference (compatibility): [https://docs.ollama.com/api](https://docs.ollama.com/api)

## Optional upstream guides

These describe stock Ollama; Prismalama behavior may differ where **`docs/RUNTIME_DISPATCH.md`** or **`OLLAMA_*`** defaults apply.

- [Quickstart](https://docs.ollama.com/quickstart)
- [Importing models](https://docs.ollama.com/import)
- [Linux](https://docs.ollama.com/linux) · [macOS](https://docs.ollama.com/macos) · [Windows](https://docs.ollama.com/windows) · [Docker](https://docs.ollama.com/docker)
- [OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)
- [Anthropic compatibility](./api/anthropic-compatibility.mdx)

## Development

| Document | Notes |
|----------|-------|
| [development.md](./development.md) | Generic build notes (upstream); prefer **DEVELOPER.md** for Prismalama |
| [TECHNICAL_DOCUMENTATION.md](./TECHNICAL_DOCUMENTATION.md) | Deep component survey (may lag; verify against code) |

## Troubleshooting

- [https://docs.ollama.com/troubleshooting](https://docs.ollama.com/troubleshooting)
- [https://docs.ollama.com/faq](https://docs.ollama.com/faq)
