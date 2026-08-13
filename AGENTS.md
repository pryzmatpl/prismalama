# Agent and automation notes

![Prismalama Logo](logo.jpg)

When working in this repository:

1. Read **`NEXT.md`** first — living roadmap (current phase + in-flight tickets + Phase 1 first actions). Then read **`docs/DEVELOPER.md`**, **`docs/PRISMALAMA_PRINCIPLE.md`**, and **`docs/GOAL-GAPS.md`** first: **GGML vs AirLLM** is the key architectural fact; runner dispatch is **`runner/dispatch.go`** (`DecideEngine`). AirLLM vs llama.cpp, environment variables, GPU memory behavior, integration test tags, and **prismallama.cpp** as the vendored GGUF engine (`Makefile.sync`, `llama/README.md`). For **which engine actually ran** (logs, multipart vs safetensors, AirLLM ports), **`docs/RUNTIME_DISPATCH.md`** and **`GET /api/prismalama/capabilities`**.
2. Prefer **evidence from code** over assumptions: large-model paths differ between **Hugging Face safetensors** (AirLLM) and **GGUF** (llama.cpp); multi-part GGUF + env flags are documented in `docs/DEVELOPER.md`.
3. Do **not** change `.env` or secret-bearing config files unless the user explicitly asks.
4. Integration tests live under `integration/` and require `-tags=integration` (and often extra tags); many tests `Skip` when models or hardware are missing.
5. **Ship bar:** each promised feature is **integration-tested**, then the **`prismalama-ollama`** package is **built**. Use **`make ship-check`** (or **`scripts/ship-check.sh`**); **`make ship-check-fast`** for **`TestBlueSky`** only without packaging. See **`README-PKGBUILD.md`**, **`docs/DEVELOPER.md` § Ship gate**. Bump **`PKGBUILD`** `pkgrel`/`pkgver` when the installable artifact changes.
6. **Docker:** **`docker/test`** is CPU-only (`prismalama-test`). **`docker/gpu`** builds **`prismalama-gpu`** (ROCm HIP + Vulkan GGML, Ubuntu base). **`docker/arch`** builds **`prismalama-arch`** from the root **`PKGBUILD`** (Arch base — same layout as **`pacman`** install); see **`docker/arch/README.md`**.

For Cursor-specific rule authoring, see the skill at `~/.cursor/skills-cursor/create-rule/SKILL.md`.

## Building

For a full build from the repository root:

```sh
cmake -B build .
cmake --build build --parallel 8
./ollama serve
```

For quick Go-only iteration against an existing native payload:

```sh
go build .
go run . serve
```

See `docs/development.md` for prerequisites, platform notes, GPU backends, and
the full development workflow.
