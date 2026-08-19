# llama/ — prismallama.cpp CGo Bindings

> Go CGo wrappers for the vendored prismallama.cpp fork (GGUF inference engine).
> Upstream: https://github.com/piotroxp/prismallama.cpp
> Sync: `Makefile.sync` (pin `FETCH_HEAD`, apply patches, rsync)

---

## Files

| File                | Lines | Purpose                                                                |
| ------------------- | ----: | ---------------------------------------------------------------------- |
| `llama.go`          |   810 | Primary CGo wrapper — devices, model, context, batch, sampling, vision |
| `llama_test.go`     |     — | Binding tests                                                          |
| `sampling_ext.h`    |    50 | C header bridge for sampling (common_sampler, grammar)                 |
| `build-info.cpp`    |     5 | Build metadata (FETCH_HEAD commit hash)                                |
| `build-info.cpp.in` |     5 | Template for `@FETCH_HEAD@` substitution                               |

---

## CGo API (`llama.go`)

### Devices

```go
func EnumerateGPUs() []DeviceInfo    // Iterates GGML backend devices (GPU/IGPU only)
```

### Model

```go
type Model struct { /* wraps *llama_model */ }

func LoadModelFromFile(path string, params ModelParams) (*Model, error)
func (m *Model) NumVocab() int
func (m *Model) NEmbd() int
func (m *Model) Tokenize(text string, addSpecial bool) []int
func (m *Model) TokenToPiece(token int) string
func (m *Model) TokenIsEog(token int) bool
func (m *Model) AddBOSToken() bool
func (m *Model) ApplyLoraFromFile(path string) error
```

### Context

```go
type Context struct { /* wraps *llama_context + threads */ }

func NewContext(model *Model, params ContextParams) (*Context, error)
func (c *Context) Decode(batch Batch) error   // returns ErrKvCacheFull if code > 0
func (c *Context) GetLogitsIth(i int) []float32
func (c *Context) GetEmbeddingsIth(i int) []float32
func (c *Context) GetEmbeddingsSeq(seq int) []float32

// KV cache management
func (c *Context) KvCacheSeqAdd(seq, p0, p1, delta int)
func (c *Context) KvCacheSeqRm(seq, p0, p1 int) bool
func (c *Context) KvCacheSeqCp(srcSeq, dstSeq, p0, p1 int)
func (c *Context) KvCacheClear()
func (c *Context) KvCacheCanShift() bool
```

### ContextParams

```go
type ContextParams struct {
    NumCtx, NumBatch, NumSeqMax int
    Threads                     int
    FlashAttention              bool
    KVCacheType                 string  // "q8_0", "q4_0", "f16"
}
```

### Batch

```go
type Batch struct { /* wraps llama_batch */ }

func NewBatch(batchSize, maxSeq, embedSize int) Batch
func (b *Batch) Add(token int, embed []float32, pos int, logits bool, seqIds ...int)
func (b *Batch) Clear()
func (b *Batch) Free()
```

### Sampling

```go
type SamplingContext struct { /* wraps *common_sampler */ }
type SamplingParams struct {
    TopK, TopP, MinP, TypicalP, Temp float32
    PenaltyRepeat, PenaltyFreq, PenaltyPresent float32
    Seed int
    Grammar string
}

func NewSamplingContext(model *Model, params SamplingParams) *SamplingContext
func (s *SamplingContext) Sample(ctx *Context, idx int) int
func (s *SamplingContext) Accept(token int)
func SchemaToGrammar(schema string) (string, error)  // JSON schema → GBNF
```

### Grammar

```go
type Grammar struct { /* wraps *llama_grammar, mu sync.Mutex */ }

func NewGrammar(vocabIDs []int, pieces []string, eogTokens []int) *Grammar
func (g *Grammar) Apply(logits []float32)
func (g *Grammar) Accept(token int)
func (g *Grammar) Free()  // thread-safe nil check
```

### Vision (multimodal)

```go
type MtmdContext struct { /* wraps *mtmd_context */ }

func NewMtmdContext(model *Model, params MtmdParams) *MtmdContext
func (m *MtmdContext) MultimodalTokenize(text string, images [][]byte) ([]MtmdChunk, error)
```

---

## Patches (`patches/`)

32 local patches maintained in `patches/README.md` with audit table.

**Categories:**

- **Model-specific (keep local):** pretokenizer, clip-unicode, solar-pro, DeepSeek regex, interleave MRoPE
- **Multi-vendor (keep local):** GPU discovery enhancements, NVML fallback, device sorting
- **Robustness (keep local):** harden exception registration, exit vs abort, skip large CUDA batches
- **Safe to upstream:** argsort, batch size hint, no-alloc mode, DXGI+PDH memory detection, MLA flash attention

**Sync commands:**

```bash
make -f Makefile.sync clean apply-patches sync    # Full sync
make -f Makefile.sync sync-audit-check            # CI guard
make -f Makefile.sync print-patches-audit         # List patches + subjects
```

---

## Vendoring

```
llama/
├── llama.cpp/              — Vendored prismallama.cpp tree (rsync from llama/vendor)
│   └── .rsync-filter       — Selective include (common/, include/, src/, tools/mtmd)
├── vendor/                 — Git clone of upstream (FETCH_HEAD pin)
├── patches/                — 32 local patches
│   └── README.md           — Audit table + bisect policy
└── compat/                 — Compatibility shims
```

`Makefile.sync` flow: `checkout` → `apply-patches` → `sync` (rsync to `llama/llama.cpp/` + `ml/backend/ggml/ggml/`).
