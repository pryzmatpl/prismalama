# ml/streaming — GGUF Weight Streaming Infrastructure

> Block-by-block weight loading and eviction for models larger than VRAM.
> Controlled by `OLLAMA_LAYER_STREAMING=1` and `OLLAMA_STREAMING_BUDGET` (default 4 GiB).
> See [`docs/PRISMALAMA_PRINCIPLE.md`](../../docs/PRISMALAMA_PRINCIPLE.md).

---

## Architecture

```
ComputeWithNotify (GGML eval callback)
    │
    ▼
InferenceStreamer.OnBlockDone(blockIdx)
    ├─ loadBlock(nextBlock)     ← read GGUF tensor data → backend tensors
    └─ evictBlock(blockIdx)     ← zero in-place (HIP buffers not freed)

OR (higher-level):

Streamer.Prepare(layerIdx)
    ├─ Prefetcher.Fetch(layerIdx)  ← async I/O with concurrency semaphore
    ├─ BudgetTracker.NeedEvict()   ← LRU eviction to make room
    └─ waitForLayer()              ← block until ready
```

## Files

### `inference.go` — Eval Callback Streamer (253 lines)

The low-level streamer installed around `ComputeWithNotify` calls.

```go
type InferenceStreamer struct {
    backend     ml.Backend
    modelPath   string
    LayerMap    *LayerMap
    currentBlock int
    loadedWeights map[string]bool
    budgetBytes  int64  // 0 = always evict
}

func NewInferenceStreamer(backend, modelPath, budgetBytes) *InferenceStreamer
func (s *InferenceStreamer) Setup() error          // MapFromFile + PrepareForInference
func (s *InferenceStreamer) MapFromFile(path) error // Decode GGUF, build LayerMap
func (s *InferenceStreamer) PrepareForInference()   // Load block 0 + output layer
func (s *InferenceStreamer) OnBlockDone(blockIdx)   // Eval callback: load next, evict current
```

**Keep-resident logic:** if the full model fits within `budgetBytes`, eviction is skipped
(`keepResident()` returns true). Peak memory: 1 block + KV cache + activations.

### `layermap.go` — GGUF Block Boundary Detection (178 lines)

Parses GGUF tensor names to identify transformer blocks.

```go
type LayerInfo struct {
    Index      int
    Name       string       // "blk.0", "blk.1", ...
    ByteSize   int64
    TensorCount int
    Tensors    []TensorRef  // { Name, Offset, Size }
}

type LayerMap struct {
    Layers           []LayerInfo
    TensorDataBase   int64       // File offset to tensor data section
    TotalWeightBytes int64
    BlockCount       int
}

func BuildLayerMap(g *ggml.GGML) (*LayerMap, error)
func (lm *LayerMap) FitsInBudget(budgetBytes int64) int  // How many layers fit
func (lm *LayerMap) LayerByteRange(idx int) (start, end int64)
func (lm *LayerMap) ReadLayerTensors(r io.ReaderAt, idx int) (map[string][]byte, error)
```

### `budget.go` — LRU Residency Budget (185 lines)

Fixed-byte budget with LRU eviction policy.

```go
type LayerState int
const (
    LayerEvicted LayerState = iota
    LayerLoading
    LayerResident
)

type BudgetTracker struct { /* budgetBytes, usedBytes, states[] */ }

func NewBudgetTracker(layerSizes []int64, budgetBytes int64) *BudgetTracker
func (bt *BudgetTracker) MarkLoading(idx int)
func (bt *BudgetTracker) MarkResident(idx int)
func (bt *BudgetTracker) MarkEvicted(idx int)
func (bt *BudgetTracker) NeedEvict(wantBytes int64, protect []int) []int  // LRU order
func (bt *BudgetTracker) State(idx int) LayerState
func (bt *BudgetTracker) UsedBytes() int64
func (bt *BudgetTracker) AvailableBytes() int64
```

### `prefetch.go` — Async I/O with Concurrency Bounding (103 lines)

```go
type PrefetchResult struct {
    LayerIndex int
    Tensors    map[string][]byte
    Err        error
    ReadTime   time.Duration
}

type Prefetcher struct { /* lm, reader, sem chan, inflight map */ }

func NewPrefetcher(lm *LayerMap, r io.ReaderAt, concurrency int) *Prefetcher
func (p *Prefetcher) Fetch(ctx context.Context, layerIdx int) <-chan PrefetchResult
func (p *Prefetcher) IsInflight(layerIdx int) bool
```

### `streamer.go` — High-Level Orchestrator (217 lines)

Coordinates budget, prefetch, and layer lifecycle.

```go
type StreamerConfig struct {
    BudgetBytes    int64  // VRAM budget
    PrefetchAhead  int    // layers to prefetch (default 1)
    IOConcurrency  int    // concurrent reads (default 2)
}

type Streamer struct { /* lm, budget, prefetch, cfg, residentData, pending */ }

func NewStreamer(g *ggml.GGML, r io.ReaderAt, cfg StreamerConfig) (*Streamer, error)
func (s *Streamer) Prepare(ctx context.Context, layerIdx int) (map[string][]byte, error)
func (s *Streamer) Release(layerIdx int)
func (s *Streamer) Stats() (usedBytes, availableBytes int64)
func (s *Streamer) RunAllLayers(callback func(idx int, tensors map[string][]byte) error) error
```

### `texture_inference.go` — BC4/DCT Compressed Streaming (265 lines)

Extends `InferenceStreamer` with weight compression.

```go
type CompressedInferenceStreamer struct {
    InferenceStreamer
    texBackend VulkanWeightTextureBackend
    format     CompressionFormat
}

type TextureStreamingCache struct { /* layerCache, maxResidents */ }
func (c *TextureStreamingCache) LoadLayer(name string, data []float32, w, h int) error
func (c *TextureStreamingCache) CompressionRatio() float64
```

---

## Related

- [`ml/weightimage/`](../weightimage/) — BC4/DCT compression codec
- [`ml/backend/ggml/ggml_streaming.go`](../backend/ggml/ggml_streaming.go) — LoadStreaming (layer-by-layer model loading)
- [`docs/STREAMING_BENCHMARK.md`](../../docs/STREAMING_BENCHMARK.md) — Performance data
- [`docs/WEIGHT_STREAMING_STRATEGY.md`](../../docs/WEIGHT_STREAMING_STRATEGY.md) — Architecture options
