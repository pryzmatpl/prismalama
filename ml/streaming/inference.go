package streaming

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"sync"
	"time"

	"github.com/ollama/ollama/ml"
)

// InferenceStreamer manages weight loading/eviction during GGML graph compute.
// It works with the GGML backend's eval callback to load the next block's weights
// and evict the previous block's at each block boundary.
type InferenceStreamer struct {
	backend   ml.Backend
	modelPath string
	lm        *LayerMap

	mu            sync.Mutex
	currentBlock  int
	totalBlocks   int
	file          *os.File
	loadedWeights map[int]bool
	// budgetBytes is the VRAM/host residency cap. 0 means always-evict
	// (legacy sliding window). When the full LayerMap fits, completed
	// blocks stay resident so a model smaller than the budget does not
	// reload every block on every token.
	budgetBytes uint64
}

// NewInferenceStreamer creates a streamer for runtime inference weight management.
func NewInferenceStreamer(backend ml.Backend, modelPath string, lm *LayerMap) *InferenceStreamer {
	return &InferenceStreamer{
		backend:       backend,
		modelPath:     modelPath,
		lm:            lm,
		currentBlock:  -1,
		totalBlocks:   lm.BlockCount,
		loadedWeights: make(map[int]bool),
	}
}

// SetBudgetBytes sets the residency budget used by OnBlockDone. 0 = always evict.
func (is *InferenceStreamer) SetBudgetBytes(budgetBytes uint64) {
	is.budgetBytes = budgetBytes
}

// PrepareForInference loads the initial block weights (block 0 + output/embeddings)
// and opens the model file for streaming reads.
func (is *InferenceStreamer) PrepareForInference(ctx context.Context) error {
	f, err := os.Open(is.modelPath)
	if err != nil {
		return fmt.Errorf("open model for streaming: %w", err)
	}
	is.file = f

	outputIdx := len(is.lm.Layers) - 1
	if outputIdx >= 0 && is.lm.Layers[outputIdx].Name == "output" {
		if err := is.loadBlock(outputIdx); err != nil {
			return fmt.Errorf("load output layer: %w", err)
		}
	}

	if is.totalBlocks > 0 {
		if err := is.loadBlock(0); err != nil {
			return fmt.Errorf("load block 0: %w", err)
		}
	}

	is.currentBlock = 0
	return nil
}

// OnBlockDone is the callback for the eval callback mechanism. It is called
// when the last node of block blockIdx has been computed. It loads the next
// block's weights and evicts the current block unless the full model fits
// in the streaming budget (keep-resident).
func (is *InferenceStreamer) OnBlockDone(blockIdx int) bool {
	is.mu.Lock()
	defer is.mu.Unlock()

	nextBlock := blockIdx + 1
	if nextBlock < is.totalBlocks {
		start := time.Now()
		if err := is.loadBlock(nextBlock); err != nil {
			slog.Error("streaming inference: failed to load block", "block", nextBlock, "error", err)
			return false
		}
		slog.Debug("streaming inference: loaded block", "block", nextBlock, "duration", time.Since(start))
	}

	if blockIdx >= 0 && blockIdx < is.totalBlocks {
		if is.keepResident() {
			slog.Debug("streaming inference: kept block", "block", blockIdx, "budget_bytes", is.budgetBytes)
		} else {
			is.evictBlock(blockIdx)
			slog.Debug("streaming inference: evicted block", "block", blockIdx)
		}
	}

	is.currentBlock = nextBlock
	return true
}

// keepResident is true when the streaming budget can hold every layer, so
// evicting a completed block would only add NVMe round-trips.
func (is *InferenceStreamer) keepResident() bool {
	if is.lm == nil || is.budgetBytes == 0 {
		return false
	}
	return is.lm.FitsInBudget(is.budgetBytes) >= len(is.lm.Layers)
}

// Close releases resources.
func (is *InferenceStreamer) Close() {
	if is.file != nil {
		is.file.Close()
		is.file = nil
	}
}

// loadBlock reads a block's tensor data from the GGUF file and writes it
// into the corresponding backend tensors via Backend.Get(name).FromBytes().
func (is *InferenceStreamer) loadBlock(blockIdx int) error {
	if blockIdx < 0 || blockIdx >= len(is.lm.Layers) {
		return fmt.Errorf("block %d out of range", blockIdx)
	}
	if is.loadedWeights[blockIdx] {
		return nil
	}
	if is.backend == nil {
		is.loadedWeights[blockIdx] = true
		return nil
	}

	li := is.lm.Layers[blockIdx]
	for _, ref := range li.Tensors {
		t := is.backend.Get(ref.Name)
		if t == nil {
			slog.Warn("streaming inference: tensor not found in backend", "name", ref.Name)
			continue
		}

		buf := make([]byte, ref.Size)
		absOff := int64(is.lm.TensorDataBase + ref.Offset)
		if _, err := is.file.ReadAt(buf, absOff); err != nil {
			return fmt.Errorf("read tensor %s: %w", ref.Name, err)
		}

		if err := safeFromBytes(t, buf); err != nil {
			slog.Warn("streaming inference: tensor size mismatch, skipping",
				"name", ref.Name, "gguf_size", ref.Size, "error", err)
		}
	}

	is.loadedWeights[blockIdx] = true
	return nil
}

// evictBlock zeros out a block's tensors to release physical pages.
// For mmap'd buffers, the OS reclaims the pages. For malloc'd buffers,
// this at least allows the OS to compress/dedup zero pages.
func (is *InferenceStreamer) evictBlock(blockIdx int) {
	if blockIdx < 0 || blockIdx >= len(is.lm.Layers) || is.backend == nil {
		delete(is.loadedWeights, blockIdx)
		return
	}

	li := is.lm.Layers[blockIdx]
	for _, ref := range li.Tensors {
		t := is.backend.Get(ref.Name)
		if t == nil {
			continue
		}
		zeros := make([]byte, ref.Size)
		_ = safeFromBytes(t, zeros)
	}

	delete(is.loadedWeights, blockIdx)
}

// safeFromBytes writes buf into tensor t, recovering from panics caused by
// size mismatches (e.g., format-transformed tensors like BF16→FP32).
func safeFromBytes(t ml.Tensor, buf []byte) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("%v", r)
		}
	}()
	t.FromBytes(buf)
	return nil
}
