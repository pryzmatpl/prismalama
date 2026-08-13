# NEXT — Prismalama roadmap (living document)

> Picked up by the next agent (or human) immediately on session start.
> Mirrors [JAISIU-2156](https://pryzmat.youtrack.cloud/issue/JAISIU-2156) phase plan.
> Updated on every Phase 0/1/2/3/4 commit.

## Current phase

**Phase 0 — Stabilize** (in progress; target close: 2026-08-27).

## In-flight tickets

| ID                                                              | Title                                                                                 | Owner | Status      |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ----- | ----------- |
| [JAISIU-2156](https://pryzmat.youtrack.cloud/issue/JAISIU-2156) | Epic — Prismalama North-Star (2026–2027)                                              | agent | Submitted   |
| [JAISIU-2157](https://pryzmat.youtrack.cloud/issue/JAISIU-2157) | P0-1: Harden `runner/dispatch.go` (typed reasons, structured logging, expanded tests) | tbd   | Submitted   |
| [JAISIU-2158](https://pryzmat.youtrack.cloud/issue/JAISIU-2158) | P0-2: Capabilities operator surface (env keys + backends, schema v2)                  | tbd   | Submitted   |
| [JAISIU-2159](https://pryzmat.youtrack.cloud/issue/JAISIU-2159) | P0-3: `prismallama.cpp` upstream sync audit + `llama/patches/README.md`               | tbd   | Submitted   |
| [JAISIU-2160](https://pryzmat.youtrack.cloud/issue/JAISIU-2160) | P0-4: Packaging defaults audit (PKGBUILD, `/etc/default/ollama`, Docker)              | tbd   | Submitted   |
| [JAISIU-2161](https://pryzmat.youtrack.cloud/issue/JAISIU-2161) | P0-5: NEXT.md living roadmap + CHANGELOG hygiene (this ticket)                        | tbd   | In Progress |
| [JAISIU-2162](https://pryzmat.youtrack.cloud/issue/JAISIU-2162) | P0-6: Verify integration suite + ship-check on a non-GPU host                         | tbd   | Submitted   |

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

- **2026-08-13** — Phase 0 scoped to 6 subtasks under JAISIU-2156 (Telegram 10:39 UTC brief).
- **2026-08-13** — Decision: Phase 0 work happens on `devel`, MR per subtask.
- **2026-08-13** — Decision: prior agent doc drift snapshotted to
  `chore/snapshot-existing-agent-changes-2026-08-13` (commit `860034c8`) so Phase 0 diffs stay clean.
