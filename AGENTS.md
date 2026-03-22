# Agent and automation notes

When working in this repository:

1. Read **`docs/DEVELOPER.md`** first for runner selection (`runner/runner.go`), AirLLM vs llama.cpp, environment variables, GPU memory behavior, integration test tags, and **prismallama.cpp** as the vendored GGUF engine (`Makefile.sync`, `llama/README.md`). For **which engine actually ran** (logs, multipart vs safetensors, AirLLM ports), **`docs/RUNTIME_DISPATCH.md`**.
2. Prefer **evidence from code** over assumptions: large-model paths differ between **Hugging Face safetensors** (AirLLM) and **GGUF** (llama.cpp); multi-part GGUF + env flags are documented in `docs/DEVELOPER.md`.
3. Do **not** change `.env` or secret-bearing config files unless the user explicitly asks.
4. Integration tests live under `integration/` and require `-tags=integration` (and often extra tags); many tests `Skip` when models or hardware are missing.
5. **Ship bar:** each promised feature is **integration-tested**, then the **`prismalama-ollama`** package is **built**. Use **`make ship-check`** (or **`scripts/ship-check.sh`**); **`make ship-check-fast`** for **`TestBlueSky`** only without packaging. See **`README-PKGBUILD.md`**, **`docs/DEVELOPER.md` § Ship gate**. Bump **`PKGBUILD`** `pkgrel`/`pkgver` when the installable artifact changes.
6. **Docker:** **`docker/test`** is CPU-only (`prismalama-test`). **`docker/gpu`** builds **`prismalama-gpu`** (ROCm HIP + Vulkan GGML, k8s-oriented); see **`docker/gpu/README.md`**.

For Cursor-specific rule authoring, see the skill at `~/.cursor/skills-cursor/create-rule/SKILL.md`.
