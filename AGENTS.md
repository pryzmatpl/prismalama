# Agent and automation notes

When working in this repository:

1. Read **`docs/DEVELOPER.md`** first for runner selection (`runner/runner.go`), AirLLM vs llama.cpp, environment variables, GPU memory behavior, integration test tags, and **prismallama.cpp** as the vendored GGUF engine (`Makefile.sync`, `llama/README.md`).
2. Prefer **evidence from code** over assumptions: large-model paths differ between **Hugging Face safetensors** (AirLLM) and **GGUF** (llama.cpp); multi-part GGUF + env flags are documented in `docs/DEVELOPER.md`.
3. Do **not** change `.env` or secret-bearing config files unless the user explicitly asks.
4. Integration tests live under `integration/` and require `-tags=integration` (and often extra tags); many tests `Skip` when models or hardware are missing.

For Cursor-specific rule authoring, see the skill at `~/.cursor/skills-cursor/create-rule/SKILL.md`.
