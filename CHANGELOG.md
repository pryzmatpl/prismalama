# Changelog

All notable changes to Prismalama will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — Phase 0 (in progress, target close 2026-08-27)

### Added
- `NEXT.md` at repo root — living roadmap mirroring the JAISIU-2156 phase plan
- `docs/PACKAGING_DEFAULTS.md` — single inventory of shipped defaults (P0-4)
- `docs/SHIP_CHECK.md` — what `make ship-check` / `ship-check-fast` runs, in order (P0-6)
- `llama/patches/README.md` — upstream patch audit + sync policy (P0-3)
- `POST /api/prismalama/dispatch` — dry-run dispatch endpoint with reason trace (P0-1)
- `GET /api/prismalama/capabilities` v2 schema — build info, backends array, expanded env keys, resolved values (P0-2)
- `make sync-audit-check` — fails if `llama/patches/README.md` is stale (P0-3)
- `make print-defaults` — prints the inventory table for CI (P0-4)

### Changed
- `runner/dispatch.go` — added typed `Reason` enum and `EngineDecision` trace record (P0-1; signature of `DecideEngine` unchanged for back-compat)
- `runner/runner.go` — `slog.Info("runner dispatch", ...)` now logs the typed reason
- `server/prismalama_capabilities.go` — schema v2 with build info, backends, expanded env (P0-2)
- `docs/RUNTIME_DISPATCH.md` — operator hints + schema v2 note
- `docs/ARCHITECTURE.md` — updated env table
- `docs/DEVELOPER.md` — references to `llama/patches/README.md`, `docs/PACKAGING_DEFAULTS.md`, `docs/SHIP_CHECK.md`

### Fixed
- Operator surface: `OLLAMA_KEEP_ALIVE`, `OLLAMA_GPU_OVERHEAD`, `OLLAMA_STREAMING_BUDGET`, `OLLAMA_LIBRARY_PATH`, `HIP_VISIBLE_DEVICES`, `AIRLLM_DEVICE`, `AIRLLM_COMPRESSION`, `PRISMALAMA_AIRLLM_PYTHONPATH` now visible in `capabilities` (P0-2)

### Security
- n/a

## Past releases

Historical releases were tracked informally in commit messages and
`docs/PRISMALAMA_PRINCIPLE.md` plus `REPORTS/`. Phase 0 introduces this
changelog; back-filling historical releases is out of scope for Phase 0
(see [JAISIU-2161](https://pryzmat.youtrack.cloud/issue/JAISIU-2161) non-goals).