# GPU deployment guide (reconciled)

This guide intentionally avoids fixed TPS guarantees and focuses on verifiable runtime checks.

## Choose a deployment path

1. **Preferred (maintained):** use `docker/gpu/README.md` or `docker/arch/README.md`.
2. **Legacy helpers:** use top-level scripts only if you understand and accept local customization.

## Verify prerequisites

- Docker is available and can run containers.
- GPU runtime is installed for your vendor.
- Model storage path has enough space.

## Bring up a server

Use your selected README path, then verify:

```bash
curl -sS http://127.0.0.1:11434/api/version
curl -sS http://127.0.0.1:11434/api/prismalama/capabilities
```

## Validate Prismalama promises at runtime

The capabilities endpoint should answer:

- whether `OLLAMA_LAYER_STREAMING` is enabled in this process
- what streaming budget bytes are active
- current environment passthrough fields
- operator hints for large-model operation

Helper:

```bash
cd /path/to/prismalama/scripts
./verify-prismalama-runtime.sh
```

## Benchmarking

Use repository benchmark tooling against your running endpoint:

```bash
cd /path/to/prismalama
python3 benchmark-gpu.py
```

Report benchmark results with:

- GPU model + driver
- model + quantization tag
- prompt length and generated tokens
- concurrent request count
- server env (`OLLAMA_*`) used during run

## Troubleshooting references

- Runner/engine selection: `docs/RUNTIME_DISPATCH.md`
- Defaults and gaps: `docs/GOAL-GAPS.md`
- Arch package defaults: `README-PKGBUILD.md`
- Security posture: `SECURITY.md`
