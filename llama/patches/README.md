# prismallama.cpp — local patches audit

> Source of truth for which patches Prismalama carries on top of upstream
> [piotroxp/prismallama.cpp](https://github.com/piotroxp/prismallama.cpp).
> Maintained by the Phase 0 sync-audit work ([JAISIU-2159](https://pryzmat.youtrack.cloud/issue/JAISIU-2159)).

## Upstream

| Key | Value |
|-----|-------|
| URL | https://github.com/piotroxp/prismallama.cpp.git |
| Branch (developer) | `main` (refresh on every sync) |
| Pinned commit (`Makefile.sync FETCH_HEAD`) | `20efe75cf1127268cb2ad73accd5ccb6f33064ff` |
| Last full audit | 2026-08-13 |
| Sync cadence | Phase 0: every MR; Phase 1+: weekly + on-demand |
| Audit tool | `make -f Makefile.sync sync` (read-only) |

> **Bumping `FETCH_HEAD`** requires operator sign-off. After a successful
> merge of `ggml-org/llama.cpp` (or Ollama's pinned llama SHA) into
> `piotroxp/prismallama.cpp`, an agent may open a PR that:
> 1. Updates `FETCH_HEAD` in `Makefile.sync`.
> 2. Re-runs `make -f Makefile.sync clean apply-patches sync`.
> 3. Re-stamps this README's "Last full audit" + "Pinned commit".
> 4. Posts the diff to the JAISIU-2159 ticket.

## Local patches (32 total, 2026-08-13 snapshot)

Patches below were inherited from Ollama and re-applied against the
prismallama.cpp fork. Each row is `<filename> | <Subject: header>` (the
`Subject:` line is the only human-friendly description in a `git
format-patch` blob — the patch bodies are unmaintained).

| File | Subject (from patch header) | Bisect policy |
|------|------------------------------|---------------|
| `0002-pretokenizer.patch` | pretokenizer | keep local — model-specific |
| `0003-clip-unicode.patch` | clip-unicode | keep local — model-specific |
| `0004-solar-pro.patch` | solar-pro | keep local — model-specific |
| `0005-fix-deepseek-deseret-regex.patch` | fix deepseek deseret regex | keep local — model-specific |
| `0006-maintain-ordering-for-rules-for-grammar.patch` | maintain ordering for rules for grammar | keep local — ggml behaviour |
| `0007-sort-devices-by-score.patch` | sort devices by score | keep local — discovery |
| `0008-add-phony-target-ggml-cpu-for-all-cpu-variants.patch` | add phony target ggml-cpu for all cpu variants | safe to upstream — build infra |
| `0009-remove-amx.patch` | remove amx | safe to upstream — generic cleanup |
| `0010-fix-string-arr-kv-loading.patch` | fix string arr kv loading | safe to upstream — bug fix |
| `0011-ollama-debug-tensor.patch` | ollama debug tensor | keep local — ollama-specific |
| `0012-add-ollama-vocab-for-grammar-support.patch` | add ollama vocab for grammar support | keep local — ollama-specific |
| `0013-add-argsort-and-cuda-copy-for-i32.patch` | add argsort and cuda copy for i32 | safe to upstream — ggml generic |
| `0014-graph-memory-reporting-on-failure.patch` | graph memory reporting on failure | safe to upstream — diagnostic |
| `0015-ggml-Export-GPU-UUIDs.patch` | ggml: Export GPU UUIDs | safe to upstream — ggml API |
| `0016-add-C-API-for-mtmd_input_text.patch` | add C API for mtmd_input_text | safe to upstream — mtmd |
| `0017-no-power-throttling-win32-with-gnuc.patch` | no power throttling win32 with gnuc | safe to upstream — win32 fix |
| `0018-ggml-Add-batch-size-hint.patch` | ggml: Add batch size hint | safe to upstream — ggml API |
| `0019-fix-mtmd-audio.cpp-build-on-windows.patch` | fix mtmd-audio.cpp build on windows | safe to upstream — windows fix |
| `0020-ggml-No-alloc-mode.patch` | ggml: No-alloc mode | safe to upstream — ggml API |
| `0021-decode-disable-output_all.patch` | decode: disable output_all | keep local — decode path |
| `0022-ggml-Enable-resetting-backend-devices.patch` | ggml: Enable resetting backend devices | safe to upstream — ggml API |
| `0023-harden-uncaught-exception-registration.patch` | harden uncaught exception registration | keep local — Prismalama robustness |
| `0024-GPU-discovery-enhancements.patch` | GPU discovery enhancements | keep local — Prismalama multi-vendor |
| `0025-NVML-fallback-for-unified-memory-GPUs.patch` | NVML fallback for unified memory GPUs | keep local — Prismalama multi-vendor |
| `0026-report-LoadLibrary-failures.patch` | report LoadLibrary failures | safe to upstream — diagnostics |
| `0027-interleave-multi-rope.patch` | interleave multi-rope | keep local — model-specific |
| `0028-Add-memory-detection-using-DXGI-PDH.patch` | Add memory detection using DXGI + PDH | safe to upstream — windows diag |
| `0029-ggml-cuda-skip-large-batches.patch` | ggml-cuda: skip large batches | keep local — Prismalama concurrency |
| `0030-fix-bakllava-regression.patch` | fix bakllava regression | safe to upstream — bug fix |
| `0031-win-exit-instead-of-abort.patch` | win: exit instead of abort | safe to upstream — windows fix |
| `0032-ggml-enable-MLA-flash-attention-for-GLM-4.7-flash.patch` | ggml: enable MLA flash attention for GLM-4.7 flash | keep local — model-specific |
| `0033-ggml-metal-solve_tri.patch` | ggml: metal solve_tri | safe to upstream — metal backend |

### Bisect policy (cheat sheet)

- **safe to upstream** — generic bug fix / API addition that any llama.cpp
  consumer would benefit from. Open a PR to `ggml-org/llama.cpp` (and to
  `piotroxp/prismallama.cpp` if the original fork lives there). Once
  merged upstream, **drop the local patch on the next sync** to keep
  this repo bisectable.
- **keep local** — ollama-specific, Prismalama multi-vendor (HIP/Vulkan
  fixes), Prismalama concurrency, or model-specific. These should never
  be upstreamed as-is; if there's value, split out the generic parts.

## Sync procedure

1. `cd <prismalama repo>`
2. `make -f Makefile.sync clean` — removes `llama/vendor` and rsynced trees.
3. `make -f Makefile.sync apply-patches sync` — re-fetches upstream at
   `FETCH_HEAD`, re-applies every patch in this directory.
4. If a patch fails to apply:
   - Inspect the conflict (the apply-patches target prints the file).
   - Decide: upstream now includes the same fix? → **drop the patch**.
   - Or: rebased upstream, conflict expected → **regenerate the patch**
     with `git format-patch -1 <sha>` from a temporary branch against
     new upstream.
5. Re-run `make print-patches-audit` (see below) and commit the new
   `llama/patches/README.md` snapshot.

## CI / pre-merge hook

The Makefile exposes a read-only audit target. CI MUST call it on every
PR that touches `Makefile.sync` or `llama/patches/`:

```makefile
.PHONY: sync-audit-check
sync-audit-check:
	@echo "Verify llama/patches/README.md references FETCH_HEAD=$(FETCH_HEAD)"
	@grep -q "FETCH_HEAD.*$(FETCH_HEAD)" llama/patches/README.md || \
	  (echo "stale patches README — please re-run sync-audit" && exit 1)
```

Run locally:

```bash
make -f Makefile.sync sync-audit-check
```

If it fails, refresh the README's "Pinned commit" + "Last full audit"
lines (this file) and re-commit.

## Audit history

| Date | Auditor | Action | Notes |
|------|---------|--------|-------|
| 2026-08-13 | Phase 0 agent | Created this README + `sync-audit-check` target | Initial 32-patch inventory; bisect policy stub |

## Related

- `Makefile.sync` — `UPSTREAM`, `WORKDIR`, `FETCH_HEAD`
- `llama/README.md` — vendoring workflow
- `docs/DEVELOPER.md` § "Upstream engine: prismallama.cpp" — operator-facing description
- [JAISIU-2159](https://pryzmat.youtrack.cloud/issue/JAISIU-2159) — Phase 0 audit ticket
- [JAISIU-2156](https://pryzmat.youtrack.cloud/issue/JAISIU-2156) — North-Star Epic