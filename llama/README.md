# Llama

This package provides Go bindings to **llama.cpp** (GGUF + GGML), vendored into this tree.

## Upstream: Prismallama.cpp

Prismalama uses **[prismallama.cpp](https://github.com/piotroxp/prismallama.cpp)** as the **canonical C/C++ engine** for GGUF inference (Vulkan, CUDA, HIP, etc.). That repo is a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp); GGML-native and GGUF workstreams land there first, then we sync here via `Makefile.sync`.

**Reproducible builds:** pin `FETCH_HEAD` in `Makefile.sync` to a full commit **SHA** (not a branch name) before tagging a release or publishing Arch packages.

`LLAMA_CPP_VERSION` pins Ollama's llama.cpp source. An update can change more
than compilation: it can affect model loading, GPU discovery, scheduler inputs,
runtime logs, streaming, and compatibility patches. Validate the upstream diff,
the patched source Ollama actually builds, and the affected local paths.

### Workflow

Record the old ref from the base branch and choose an explicit new llama.cpp
tag or commit. After updating `LLAMA_CPP_VERSION`, materialize the source
through Ollama's normal build path:

```sh
cmake -S llama/server --preset cpu
```

### Updating the base commit

**Pin to a new base commit**

1. Set `FETCH_HEAD` in `Makefile.sync` to the desired commit from [prismallama.cpp](https://github.com/piotroxp/prismallama.cpp) (or merge your fork’s `master` first).
2. Run:

```shell
make -f Makefile.sync apply-patches
```

If patches fail, resolve conflicts in `./vendor/`, then continue the series (commits use **Prism** / **piotr.slupski@pryzmat.pl**):

```shell
GIT_AUTHOR_NAME=Prism GIT_AUTHOR_EMAIL=piotr.slupski@pryzmat.pl git -C llama/vendor am --continue
```

Repeat `apply-patches` until clean.

3. Sync vendored sources into the tree:

### What to review

- Build option and dependency drift: changed `GGML_*` or `LLAMA_*` options,
  new `find_package` calls, generated assets, shader tools, or backend
  dependencies. Compare against `llama/server/CMakeLists.txt`,
  `llama/server/CMakePresets.json`, `cmake/local.cmake`, Dockerfiles, CI, and
  build scripts as needed.
- Backend discovery contracts: GGML symbols used by `discover/native_probe*.go`,
  `ggml_backend_dev_props`, backend device type enums, backend registry loading,
  device ordering, visible-device filtering, and CUDA/ROCm/Vulkan/Metal runtime
  library behavior.
- llama-server contracts: launch args and defaults, status and error payloads,
  memory/offload log lines, `system_info:`, flash-attention logging,
  `--main-gpu`, split-mode behavior, and scheduler-sensitive flags consumed by
  `llm/llama_server.go` or `server/sched.go`.
- Streaming: any new SSE frame shape, heartbeat, keepalive ping, completion
  marker, or response cadence on paths Ollama parses directly.
- Model and conversion surfaces: new architectures, tensor names, GGUF
  metadata, tokenizer behavior, speculative/MTP paths, sampler defaults, and
  server capabilities that may require updates under `convert/`, `model/`,
  `x/create/`, `llm/`, or `llama/compat/`. A model load alone is not enough;
  affected paths should run a real request and assert the expected result.

### Compatibility patches

Patches under `llama/compat/` are applied during configure. If a patch
insertion point moved, regenerate the patch against a fresh checkout of the new
ref rather than editing an already-patched `_deps/` tree.

If compatibility sources, model patches, `llama/server/CMakeLists.txt`, or
`cmake/local.cmake` changed, build the CPU target:

```sh
cmake --build build/llama-server-cpu --target llama-server --parallel 12
```

**Switching the remote for the first time** (e.g. from ggml-org to the fork):

```shell
rm -rf llama/vendor
make -f Makefile.sync checkout apply-patches sync
```

### Generating patches

When changing vendored C/C++ code:

Run the Go tests:

```sh
go test ./...
```

Then proceed to build the full Ollama release and verify.

### End-to-end Testing

For runtime validation, build the full applicable native payload for the
platform using the [developer guide](../docs/development.md): Metal on macOS
arm64, and the available CUDA, ROCm, and Vulkan backends on Linux and Windows.

Then run the [integration tests](../integration/README.md) on the platforms
being validated. Use them to exercise real Ollama requests and inspect logs for
device discovery, offload, memory accounting, flash attention, and
request/response behavior. macOS, Windows, and Linux behavior must be validated
on those platforms.
