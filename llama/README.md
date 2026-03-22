# `llama`

This package provides Go bindings to **llama.cpp** (GGUF + GGML), vendored into this tree.

## Upstream: Prismallama.cpp

Prismalama uses **[prismallama.cpp](https://github.com/piotroxp/prismallama.cpp)** as the **canonical C/C++ engine** for GGUF inference (Vulkan, CUDA, HIP, etc.). That repo is a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp); GGML-native and GGUF workstreams land there first, then we sync here via `Makefile.sync`.

**Reproducible builds:** pin `FETCH_HEAD` in `Makefile.sync` to a full commit **SHA** (not a branch name) before tagging a release or publishing Arch packages.

## Vendoring

We vendor `llama.cpp` and `ggml` from the clone in `./vendor/`, and carry a small set of patches under `llama/patches/`.

To (re)establish the tracking tree and apply patches:

```shell
make -f Makefile.sync apply-patches
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

```shell
make -f Makefile.sync format-patches sync
```

**Switching the remote for the first time** (e.g. from ggml-org to the fork):

```shell
rm -rf llama/vendor
make -f Makefile.sync checkout apply-patches sync
```

### Generating patches

When changing vendored C/C++ code:

```shell
make -f Makefile.sync clean apply-patches
```

Iterate in `./vendor/`, then:

```shell
make -f Makefile.sync format-patches
```

Prefer upstreaming fixes to **prismallama.cpp** so the fork stays the single source of truth; keep patches here only for Prismalama-specific glue if needed.

### Submodule `src/ollama`

The `src/ollama` git submodule has its own `Makefile.sync` (upstream Ollama defaults). For a **single** engine story across the repo, mirror the same `UPSTREAM` / `FETCH_HEAD` policy there when you merge or vendor from this project.
