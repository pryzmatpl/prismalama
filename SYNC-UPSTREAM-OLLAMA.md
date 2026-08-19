# SYNC-UPSTREAM-OLLAMA — Merge ollama/ollama into prismalama

> Agent instruction for syncing `pryzmatpl/prismalama` (develop) with `ollama/ollama` (main).
> Last audited: 2026-08-19. Merge base: `099a0f18`. Fork: 156 commits ahead, 595 behind.

---

## Situation

Prismalama is forked from `ollama/ollama`. The fork adds:

- **AirLLM engine** — Python layer-by-layer HF inference (`runner/airllmrunner/`, `runner/dispatch.go`)
- **GGUF weight streaming** — block-by-block load/evict via eval callback (`ml/streaming/`)
- **BC4/DCT weight compression** — GPU texture codec (`ml/weightimage/`)
- **prismallama.cpp patches** — 32+ local patches on vendored llama.cpp (`llama/patches/`)
- **Prismalama capabilities API** — `GET /api/prismalama/capabilities` (`server/prismalama_capabilities.go`, `api/prismalama.go`)
- **ROCm/Vulkan focus** — HIP dlopen, FA tile drops, ROCR BDF shim
- **Arch Linux packaging** — `PKGBUILD`, `/etc/default/ollama` defaults
- **Qwen3-Next DeltaNet** — fused `ggml_gated_delta_net` (`model/models/qwen3next/`)
- **MoE split offload** — VRAM-fitting attn/GDN + routed-expert tail
- **Documentation** — `CODEMAP.md`, `NEXT.md`, `docs/AIRLLM.md`, `docs/DEVELOPER.md`, etc.

The upstream (ollama/ollama) has evolved with 595 new commits (new models, API changes, UI, MLX imagegen, server improvements).

---

## Prerequisites

```bash
cd prismalama   # or wherever the prismalama checkout lives

# Ensure upstream remote exists
git remote add upstream https://github.com/ollama/ollama.git 2>/dev/null || true
git fetch upstream --tags
```

---

## Conflict zones (as of 2026-08-19)

### Zone 1: Vendored C/C++ (~335 files) — SKIP manual resolution

Files under `llama/llama.cpp/` and `ml/backend/ggml/ggml/` are vendored via `Makefile.sync` from `piotroxp/prismallama.cpp`, NOT from ollama's vendored copy. These will always conflict because both sides vendor different llama.cpp commits.

**Strategy:** After merging Go-level code, re-run `Makefile.sync` to re-vendor from prismallama.cpp upstream. Accept the upstream (ollama) side for these paths during merge, then overwrite via sync.

```bash
# During merge conflict resolution for vendored paths:
git checkout --theirs llama/llama.cpp/ ml/backend/ggml/ggml/
git add llama/llama.cpp/ ml/backend/ggml/ggml/

# After merge completes, re-vendor from prismallama.cpp:
make -f Makefile.sync clean apply-patches sync
git add llama/ ml/backend/ggml/ggml/
```

### Zone 2: Go-level Prismalama features (~170 files) — CAREFUL merge

These are the files where fork-only code interacts with upstream changes. Each needs manual review.

**Critical files (Prismalama-owned, upstream also changed):**

| File | Prismalama adds | Risk |
|------|----------------|------|
| `runner/runner.go` | AirLLM dispatch, `isAirLLMModel()` | Upstream refactored runner subprocess |
| `runner/llamarunner/runner.go` | Streaming policy, mmap layer streaming | Upstream added new sequence features |
| `runner/ollamarunner/runner.go` | Ollama-engine streaming hooks | Upstream active development |
| `llm/server.go` | `ollama_engine_server.go` hooks, mmap policy | Upstream scheduler changes |
| `ml/backend.go` | `StreamingBackend` / `StreamingComputeBackend` interfaces | Upstream added new backend methods |
| `ml/backend/ggml/ggml.go` | `ggml_streaming.go` (LoadStreaming) | Upstream GGML API changes |
| `ml/device.go` | GPU layer streaming budget | Upstream device management |
| `llama/llama.go` | CGo bindings (sampling, vision, grammar) | Upstream added new bindings |
| `llama/sampling_ext.h` / `.cpp` | Sampling bridge | Upstream refactored sampling |
| `server/sched.go` | Scheduler streaming hooks | Upstream scheduler overhaul |
| `server/routes.go` | Prismalama capabilities route | Upstream route additions |
| `envconfig/config.go` | `OLLAMA_LAYER_STREAMING`, `OLLAMA_STREAMING_BUDGET` | Upstream new env vars |
| `go.mod` / `go.sum` | Fork module path stays `github.com/ollama/ollama` | Accept upstream, re-add fork deps |
| `model/models/qwen3next/` | DeltaNet, GDN fused op | Upstream may add their own qwen3next |
| `CMakeLists.txt` | GGML build flags, local cmake | Upstream build system changes |
| `Makefile.sync` | FETCH_HEAD pin, audit targets | Fork-only — keep ours |

**Prismalama-only files (no upstream equivalent — safe):**

These should survive the merge without conflict:
- `runner/dispatch.go` — engine dispatch
- `runner/airllmrunner/` — entire AirLLM proxy
- `ml/streaming/` — entire streaming infrastructure
- `ml/weightimage/` — weight compression
- `server/prismalama_capabilities.go` — capabilities API
- `api/prismalama.go` — capabilities types
- `airllm_runner.py` — Python AirLLM server
- `src/airllm/` — AirLLM submodule
- `src/prismalama/` — NVMe striping
- `docs/AIRLLM.md`, `docs/DEVELOPER.md`, `docs/PRISMALAMA_PRINCIPLE.md`, etc.
- `CODEMAP.md`, `NEXT.md`, `AGENTS.md`
- `PKGBUILD`, `Makefile`, build scripts
- `Dockerfile.rocm`, `docker/arch/`, `docker/gpu/`
- `integration/ship_*.go`, `integration/prismalama_*.go`

### Zone 3: Docs, UI, tests, CI (~50 files) — ACCEPT upstream

Files in `docs/`, `app/ui/`, `.github/workflows/`, `x/imagegen/`, `x/` — accept upstream versions. Prismalama docs are in separate files that don't conflict.

---

## Merge procedure

### Step 1: Create merge branch

```bash
git checkout develop
git pull origin develop
git checkout -b sync/ollama-upstream-$(date +%Y%m%d)
```

### Step 2: Start merge

```bash
git merge upstream/main --no-commit --no-ff
```

This will report conflicts. Do NOT commit yet.

### Step 3: Resolve vendored C/C++ (bulk — ~335 files)

```bash
# Accept upstream for all vendored paths (will be re-synced later)
git checkout --theirs -- llama/llama.cpp/ ml/backend/ggml/ggml/ || true
git add llama/llama.cpp/ ml/backend/ggml/ggml/
```

### Step 4: Resolve docs/UI/CI (bulk — accept upstream)

```bash
git checkout --theirs -- docs/ app/ui/ .github/workflows/ x/ || true
git add docs/ app/ui/ .github/workflows/ x/
```

### Step 5: Resolve Go-level conflicts (manual — one by one)

For each remaining conflict, open the file and resolve:

**Guiding principle:** Keep Prismalama additions (AirLLM dispatch, streaming, capabilities) while incorporating upstream's structural changes.

```bash
# List remaining conflicts
git diff --name-only --diff-filter=U

# For each file: resolve, then
git add <resolved-file>
```

**Per-file guidance:**

- **`runner/runner.go`**: Keep `isAirLLMModel()` / `airLLMModelAndReason()` calls. Upstream may have refactored `Execute()` — preserve both the new structure and the AirLLM dispatch branch.
- **`llm/server.go`**: Keep `ollama_engine_server.go` references and mmap policy. Accept upstream scheduler/load changes around it.
- **`ml/backend.go`**: Keep `StreamingBackend` / `StreamingComputeBackend` interfaces. Upstream may have added new `Backend` methods — add those too.
- **`llama/llama.go`**: Upstream added new CGo functions — accept those. Keep any Prismalama-specific bindings.
- **`server/routes.go`**: Keep Prismalama capabilities route registration. Accept upstream's new routes.
- **`envconfig/config.go`**: Keep `OLLAMA_LAYER_STREAMING` and `OLLAMA_STREAMING_BUDGET`. Accept upstream's new env vars.
- **`go.mod`**: Accept upstream's dependency updates. Verify module path stays `github.com/ollama/ollama`.
- **`.gitignore`**: Merge both sides (Prismalama adds `vendor/`, `.prismalama_cuda`, etc.).
- **`Makefile.sync`**: Keep ours entirely (fork-only file).

### Step 6: Commit merge

```bash
git commit -m "sync(upstream): merge ollama/ollama main ($(git log upstream/main --oneline -1 | cut -d' ' -f1))

Merge 595 upstream commits into prismalama develop.
Preserves: AirLLM dispatch, GGUF streaming, weight compression,
Prismalama capabilities API, ROCm/Vulkan focus, Arch packaging.

Vendored llama.cpp paths accepted from upstream and will be
re-synced via Makefile.sync in follow-up commit."
```

### Step 7: Re-vendor prismallama.cpp

```bash
make -f Makefile.sync clean apply-patches sync
git add llama/ ml/backend/ggml/ggml/
git commit -m "chore(sync): re-vendor prismallama.cpp after upstream merge"
```

### Step 8: Build and test

```bash
# Go build
go build .

# Unit tests
go test ./...

# Integration (if hardware available)
go test -tags=integration ./integration -timeout 10m

# Ship gate (fast)
make ship-check-fast
```

### Step 9: Fix build failures

Upstream may have:
- Added new `Backend` interface methods → implement in `ml/backend/ggml/ggml.go`
- Changed `runner.Execute()` signature → update `airllmrunner` and `llamarunner`
- Added new model architectures → register in `model/models/models.go`
- Changed API types → update `api/prismalama.go` if it embeds upstream types

Iterate until `go build .` and `go test ./...` pass.

### Step 10: Push and PR

```bash
git push origin sync/ollama-upstream-$(date +%Y%m%d)
# Create PR: sync/ollama-upstream-YYYYMMDD → develop
```

---

## Post-merge checklist

- [ ] `go build .` passes
- [ ] `go test ./...` passes (unit tests)
- [ ] `go test -tags=integration ./integration -count=1 -timeout 10m` passes
- [ ] `make ship-check-fast` passes (TestBlueSky)
- [ ] `DecideEngine` still routes correctly (check `runner/dispatch.go` unchanged)
- [ ] `GET /api/prismalama/capabilities` returns valid JSON
- [ ] `OLLAMA_LAYER_STREAMING=1` env var still recognized
- [ ] `Makefile.sync` re-vendor succeeded (llama/patches applied cleanly)
- [ ] `llama/patches/README.md` updated if FETCH_HEAD changed
- [ ] `NEXT.md` updated with sync note
- [ ] No `.env` files modified
- [ ] Fork-only files intact: `airllm_runner.py`, `src/airllm/`, `src/prismalama/`, `ml/streaming/`, `ml/weightimage/`, `runner/dispatch.go`, `runner/airllmrunner/`

---

## Rollback

If the merge is too broken to fix:

```bash
git merge --abort   # if still in merge state
# or
git reset --hard origin/develop   # if already committed
```

---

## Scheduling

This sync should be done:
- **Before** starting new Prismalama features that touch runner/server/ml
- **After** a stable upstream release (check ollama tags: `git tag -l 'v*' --sort=-v:refname | head -5`)
- **Frequency:** every 4–8 weeks, or when upstream adds features Prismalama needs

---

## References

- Upstream: https://github.com/ollama/ollama
- Fork: https://github.com/pryzmatpl/prismalama
- prismallama.cpp: https://github.com/piotroxp/prismallama.cpp
- AirLLM: https://github.com/piotroxp/airllm
- Prismalama principle: `docs/PRISMALAMA_PRINCIPLE.md`
- Runtime dispatch: `docs/RUNTIME_DISPATCH.md`
- Developer guide: `docs/DEVELOPER.md`
