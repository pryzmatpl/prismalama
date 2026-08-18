# NEXT — Prismalama roadmap (living document)

> Picked up by the next agent (or human) immediately on session start.
> Mirrors [JAISIU-2156](https://pryzmat.youtrack.cloud/issue/JAISIU-2156) phase plan.
> Updated on every Phase 0/1/2/3/4 commit.

## Current phase

**Phase 1 — Streaming Core** (opened 2026-08-18 on the RX 7900 XTX host).
Live serve is the Ubuntu `ollama-xtx-engine` binary (`~/.ollama/rocm-overlay/ollama-xtx-engine`
copied to `/usr/bin/ollama`) + HIP overlay + ROCR BDF shim. Image backup remains
`/usr/bin/ollama.image-rocm`. `NewLlamaServer` falls back to
`runner --ollama-engine` when `llama-server` is missing: GPU discovery, load,
and generate all use that path. Proven 2026-08-18: `qwen3:0.6b` on ROCm /
7900 XTX / 24 GiB, 29/29 layers offloaded, keep-resident
(`OLLAMA_STREAMING_BUDGET` 4 GiB), warm ~106 tok/s for 16 tokens.

## In-flight tickets

| ID                                                              | Title                                                                                 | Owner | Status      |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ----- | ----------- |
| [JAISIU-2156](https://pryzmat.youtrack.cloud/issue/JAISIU-2156) | Epic — Prismalama North-Star (2026–2027)                                              | agent | Submitted   |
| [JAISIU-2295](https://pryzmat.youtrack.cloud/issue/JAISIU-2295) | Epic — Phase 1 Streaming Core                                                         | agent | In Progress |
| [JAISIU-2296](https://pryzmat.youtrack.cloud/issue/JAISIU-2296) | P1-1: HIP `libggml-hip.so` dlopen on gfx1100 — drop FA tiles D≥576                    | agent | In Progress |
| [JAISIU-2297](https://pryzmat.youtrack.cloud/issue/JAISIU-2297) | P1-2: Keep GGUF blocks resident when they fit `OLLAMA_STREAMING_BUDGET`               | agent | In Progress |
| [JAISIU-2298](https://pryzmat.youtrack.cloud/issue/JAISIU-2298) | P1-3: GPU discovery fallback when `llama-server` is missing                           | agent | In Progress |
| [JAISIU-2299](https://pryzmat.youtrack.cloud/issue/JAISIU-2299) | P1-4: `docs/STREAMING_BENCHMARK.md` on this 7900 XTX                                  | tbd   | Submitted   |
| [JAISIU-2300](https://pryzmat.youtrack.cloud/issue/JAISIU-2300) | P1-5: Wire streaming compute into `runner/llamarunner`                                | tbd   | Submitted   |

## Phase 1 — first actions (post Phase 0 close)

1. Pick a single architecture (Llama 3.x or Qwen2.5/3) and implement block-level
   streaming + eviction inside the GGML path end-to-end with benchmark curves.
2. Publish `docs/STREAMING_BENCHMARK.md` with VRAM-vs-tokens/s numbers vs stock llama.cpp.
3. Wire the streaming compute path into `runner/llamarunner` (or
   `runner/ollamarunner` for ollama-engine models).

## Open questions for the operator

- Confirm `OLLAMA_LAYER_STREAMING=1` stays the Arch default in Phase 1.
- Confirm Phase 1 target architecture (Llama vs Qwen vs other).
- Confirm whether `prismallama.cpp` upstream sync is operator-gated or agent-driven.

## Definition of "ready to move to Phase 1"

- [ ] All 6 Phase 0 children **Done**
- [ ] `make ship-check-fast` green on a clean CPU host
- [ ] `GET /api/prismalama/capabilities` reflects every shipped env key + backends array
- [ ] `llama/patches/README.md` is current with `Makefile.sync FETCH_HEAD`
- [ ] `docs/ARCHITECTURE.md` matches shipped code
- [ ] `docs/PACKAGING_DEFAULTS.md` inventory present and matches `make print-defaults`

## Phase 0 → Phase 1 transition

When all six checkboxes above tick:

1. Open a new Epic **JAISIU-2XXX — Phase 1 Streaming Core**, link as child of JAISIU-2156.
2. Re-plan the 4–8 week window into ≤6 implementation-ready subtasks (capped ~300 LOC each).
3. Update this `NEXT.md` "Current phase" → Phase 1.

## Coordination rules (binding)

- One worktree: `/home/node/.openclaw/workspace/prismalama`
- Branch off `devel`, never `main`
- Per-subtask branch: `fix/JAISIU-<N>-<short>`
- Tag every commit that closes a subtask: `Phase-0 / JAISIU-XXXX closes`
- MR per subtask; description lists files-touched + tests-run + screenshot/log evidence
- No `.env` edits, no `git push` without operator confirmation

## Recent decisions (chronological, latest first)

- **2026-08-18** — ollama-engine generate client (`llm/ollama_engine_server.go`)
  landed. `NewLlamaServer` uses `runner --ollama-engine` when `llama-server`
  is absent. Live binary is `ollama-xtx-engine` (89 MiB Ubuntu). Restore path:
  `/usr/bin/ollama.image-rocm`. Keep-resident proven (`kept_block=980`,
  `evicted_block=0` on `qwen3:0.6b` / 4 GiB budget).
- **2026-08-18** — Phase 1 opened on host `prizm` (RX 7900 XTX). Live
  `qwen3:0.6b` generate on ROCm with `OLLAMA_LAYER_STREAMING=1`. HIP dlopen
  required dropping FA tiles D≥512/576/640; ROCm 7.2 rejects PCI BDF in
  `ROCR_VISIBLE_DEVICES` (shim + `rocrVisibleDeviceToken` → `"0"`).
- **2026-08-13** — Phase 0 scoped to 6 subtasks under JAISIU-2156 (Telegram 10:39 UTC brief).
- **2026-08-13** — Decision: Phase 0 work happens on `devel`, MR per subtask.
- **2026-08-13** — Decision: prior agent doc drift snapshotted to
  `chore/snapshot-existing-agent-changes-2026-08-13` (commit `860034c8`) so Phase 0 diffs stay clean.
