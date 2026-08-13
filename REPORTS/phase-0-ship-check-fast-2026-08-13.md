# Phase 0 ship-check-fast log — 2026-08-13

> Captured by the Phase 0 audit agent ([JAISIU-2162](https://pryzmat.youtrack.cloud/issue/JAISIU-2162)).
> Markdown placeholder; replace with the actual log once a Go ≥ 1.24 host
> runs `./scripts/ship-check.sh --list` + `SHIP_SKIP_PKG=1 ./scripts/ship-check.sh`.

## Environment

- Container: `/home/node/.openclaw/workspace/prismalama`
- Branch: `fix/JAISIU-2162-ship-check-verify` (this MR)
- Go toolchain: **NOT installed** (`go: not found` in this container)
- `make`: **NOT installed** (`make: not found` in this container)

## What I could verify in this container (no Go required)

```bash
$ bash scripts/ship-check.sh --help
ship-check.sh — Prismalama ship gate
... (full usage block; see ./scripts/ship-check.sh)
```

```bash
$ bash scripts/ship-check.sh --list
ship-check.sh — step list (no execution):
  1. cd ${REPO_ROOT}
  2. echo "== integration (CGO) =="
     CGO_ENABLED=1 go test -tags=integration ./integration \
        -count=1 -timeout ${SHIP_INTEGRATION_TIMEOUT:-15m} \
        ${SHIP_GO_TEST_EXTRA}
  3. [unless SHIP_SKIP_PKG=1]
     echo "== prismalama-ollama package =="
     exec ${REPO_ROOT}/build-rocm.sh
See docs/SHIP_CHECK.md for prerequisites and failure modes.
```

```bash
$ bash scripts/test-operator-env.sh
Source: /home/node/.openclaw/workspace/prismalama/scripts/operator-env-large-models.sh
Asserting defaults after sourcing with no inherited env...
  ok    OLLAMA_MEMORY_POLICY=balanced
  ok    OLLAMA_LAYER_STREAMING=1
  ok    OLLAMA_STREAMING_BUDGET=6442450944
  ok    OLLAMA_NEW_ENGINE=1
  ok    OLLAMA_MMAP_ALLOW_LOW_RAM=1
  ok    OLLAMA_VULKAN=1
OK: operator-env-large-models.sh defaults match docs/PACKAGING_DEFAULTS.md
```

## What MUST be run on a host with Go installed

```bash
# From /home/node/.openclaw/workspace/prismalama with Go >= 1.24
SHIP_SKIP_PKG=1 ./scripts/ship-check.sh 2>&1 | tee REPORTS/phase-0-ship-check-fast-$(date +%Y-%m-%d).log
```

Then commit the log alongside this MR (or a follow-up). The acceptance
criteria for [JAISIU-2162](https://pryzmat.youtrack.cloud/issue/JAISIU-2162)
require this log be green.

## Status

- `--help` / `--list` / unknown-arg handling: **verified** ✓
- operator-env smoke test (`test-operator-env.sh`): **passing** ✓
- `make ship-check-fast` end-to-end: **NOT RUN** (no Go in this container)
- `go test ./runner/ -count=1` for the JAISIU-2157 typed-reason tests: **NOT RUN**
- `go test ./server/ -count=1` for the JAISIU-2158 capabilities v2 tests: **NOT RUN**

## Honest statement

This Phase 0 work was authored on a container with no Go toolchain. The
code was carefully reviewed for syntactic correctness and additive
behavior; existing test signatures and behavior are preserved. **All
test commands listed in the per-ticket scopes MUST be re-run on a Go ≥ 1.24
host before MR close.** No "validated" or "tested" claim is made for any
code that was not actually executed.

— Phase 0 audit agent, 2026-08-13