# Ship-check — what runs, in what order, and how to debug

> Phase 0 / [JAISIU-2162](https://pryzmat.youtrack.cloud/issue/JAISIU-2162).
> Source of truth for `make ship-check` / `make ship-check-fast` /
> `./scripts/ship-check.sh` callers.

## Prerequisites

| Step | Tool / package | Required for | Notes |
|------|----------------|--------------|-------|
| `go test -tags=integration ./integration` | Go ≥ 1.24 + CGO toolchain | both `ship-check` and `ship-check-fast` | `BC4` and other recent work requires Go 1.24+ |
| `./build-rocm.sh` | `rocm-hip-sdk` (or `cuda` for NVIDIA) + `cmake` + `ninja` | `ship-check` only (skipped by `SHIP_SKIP_PKG=1`) | ROCm build is long (15–30 min cold) |
| GPU runtime | HIP / CUDA libs on `LD_LIBRARY_PATH` (or `/usr/lib/ollama/rocm`) | both (test discovery skips silently if absent) | the integration suite is tag-gated; many tests `Skip` when models or hardware are missing |

A clean **CPU host** is enough for `SHIP_SKIP_PKG=1` — that is what CI
and the fast-path use.

## What runs (in order)

`scripts/ship-check.sh` (default invocation):

```
1. cd ${REPO_ROOT}
2. echo "== integration (CGO) =="
3. CGO_ENABLED=1 go test -tags=integration ./integration \
       -count=1 -timeout ${SHIP_INTEGRATION_TIMEOUT:-15m} \
       ${SHIP_GO_TEST_EXTRA}
4. [unless SHIP_SKIP_PKG=1]
   echo "== prismalama-ollama package =="
   exec ${REPO_ROOT}/build-rocm.sh
```

Print the list without running:

```bash
./scripts/ship-check.sh --list
```

## Variants

| Command | What it does | Use when |
|---------|--------------|----------|
| `make ship-check` | runs `./scripts/ship-check.sh` (integration + package) | full pre-merge gate (operator-supplied ROCm/CUDA host) |
| `make ship-check-fast` | `SHIP_GO_TEST_EXTRA` selects only the cheap "Ship*" tests, `SHIP_SKIP_PKG=1` | CPU host, no GPU |
| `./scripts/ship-check.sh` | same as `make ship-check` | direct invocation |
| `SHIP_SKIP_PKG=1 ./scripts/ship-check.sh` | same as `make ship-check-fast` | direct invocation |
| `./scripts/ship-check.sh --list` | print steps, exit 0 | debugging |
| `./scripts/ship-check.sh --help` | print usage | n/a |

## Env vars (all optional)

| Var | Default | Effect |
|-----|---------|--------|
| `SHIP_INTEGRATION_TIMEOUT` | `15m` | `go test` timeout |
| `SHIP_GO_TEST_EXTRA` | empty | extra args to `go test`; quote `\|` characters (e.g. `'-run=TestBlueSky\|TestShipMemoryPolicyEnv'`) |
| `SHIP_SKIP_PKG` | unset | `1` → skip `build-rocm.sh` |

## Invocation examples

```bash
# Full ship gate (CI on an ROCm host)
make ship-check

# Fast path on a CPU box
make ship-check-fast

# Fast path with explicit env override
SHIP_INTEGRATION_TIMEOUT=5m SHIP_GO_TEST_EXTRA='-run=TestShipLayerStreaming' ./scripts/ship-check.sh

# Skip the package build on an ROCm host (testing only)
SHIP_SKIP_PKG=1 make ship-check

# Dry-run
./scripts/ship-check.sh --list
```

## Failure-mode table

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `go: command not found` | Go toolchain not installed | Install Go ≥ 1.24 (see `MOST-RECENT-WORK.md` "Constraints") |
| `CGO_ENABLED=1` build fails on missing `.h` | system C headers missing (`zlib`, `ssl`) | Install distro dev packages (Arch: `base-devel`) |
| `build-rocm.sh` fails on `rocBLAS` not found | `rocm-hip-sdk` not installed or wrong version | Install per `docs/DEVELOPER.md` § "Arch package"; set `PRISMALAMA_AMDGPU_TARGETS` for non-gfx1100 |
| Tests time out at 15m | Cold build; full integration can take 20m first time | Bump `SHIP_INTEGRATION_TIMEOUT=30m` |
| Tests skip silently | Model files / hardware missing | Many tests are tag-gated + `Skip()` when missing; check `go test -v` output |
| `pkgrel`/`pkgver` not bumped after `llm/` change | Operator forgot | Re-run `make update-pkg` then rebuild |
| `next-heartbeat` job fails on missing Go | Same as "go: command not found" | Install Go ≥ 1.24 |

## Capturing a run for the MR

```bash
SHIP_SKIP_PKG=1 ./scripts/ship-check.sh 2>&1 | tee REPORTS/phase-0-ship-check-fast-$(date +%Y-%m-%d).log
```

Commit the log alongside the diff so reviewers can verify the gate
without running it themselves.

## Related

- [JAISIU-2162](https://pryzmat.youtrack.cloud/issue/JAISIU-2162) — Phase 0 ship-check verification
- [JAISIU-2156](https://pryzmat.youtrack.cloud/issue/JAISIU-2156) — North-Star Epic
- `scripts/ship-check.sh` — implementation
- `scripts/docker-test.sh` — runs `ship-check-fast` inside the CPU Docker image
- `Makefile` — `ship-check`, `ship-check-fast`, `docker-test*` targets
- `integration/TEST_README.md` — tag matrix + skip semantics
- `docs/SHIP_CHECK.md` (this file) — operator-facing guide