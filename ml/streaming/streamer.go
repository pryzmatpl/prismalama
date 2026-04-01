package streaming

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"sync"
	"time"

	"github.com/ollama/ollama/fs/ggml"
)

// StreamerConfig tunes the layer streaming orchestrator.
type StreamerConfig struct {
	BudgetBytes    uint64 // max bytes for resident layer weights (VRAM or host, depending on backend)
	PrefetchAhead  int    // how many layers to prefetch ahead of the current compute layer (default 1)
	IOConcurrency  int    // max parallel NVMe reads (default 2)
}

// DefaultStreamerConfig returns a sensible baseline: 4 GiB budget, 1 layer prefetch, 2 I/O workers.
func DefaultStreamerConfig() StreamerConfig {
	return StreamerConfig{
		BudgetBytes:   4 * 1024 * 1024 * 1024,
		PrefetchAhead: 1,
		IOConcurrency: 2,
	}
}

// Streamer orchestrates layer-by-layer weight streaming for a GGUF model. It manages a
// BudgetTracker (residency) and a Prefetcher (async NVMe reads), exposing a sequential
// "prepare → compute → advance" API that the runner calls per inference step.
type Streamer struct {
	lm       *LayerMap
	budget   *BudgetTracker
	prefetch *Prefetcher
	cfg      StreamerConfig

	mu           sync.Mutex
	residentData map[int]map[string][]byte // layerIdx → tensor name → bytes
	pending      map[int]<-chan PrefetchResult
}

// NewStreamer builds a Streamer for a decoded GGML model, reading weights from r.
func NewStreamer(g *ggml.GGML, r io.ReaderAt, cfg StreamerConfig) (*Streamer, error) {
	lm, err := BuildLayerMap(g)
	if err != nil {
		return nil, fmt.Errorf("build layer map: %w", err)
	}

	sizes := make([]uint64, len(lm.Layers))
	for i, li := range lm.Layers {
		sizes[i] = li.ByteSize
	}

	if cfg.PrefetchAhead < 0 {
		cfg.PrefetchAhead = 1
	}
	if cfg.IOConcurrency < 1 {
		cfg.IOConcurrency = 2
	}

	return &Streamer{
		lm:           lm,
		budget:       NewBudgetTracker(sizes, cfg.BudgetBytes),
		prefetch:     NewPrefetcher(lm, r, cfg.IOConcurrency),
		cfg:          cfg,
		residentData: make(map[int]map[string][]byte),
		pending:      make(map[int]<-chan PrefetchResult),
	}, nil
}

// LayerMap returns the parsed GGUF layer structure.
func (s *Streamer) LayerMap() *LayerMap { return s.lm }

// Prepare ensures layer layerIdx is resident (loaded into memory) and kicks off prefetch for
// upcoming layers. Blocks until the requested layer is ready or ctx is cancelled.
// Returns the raw tensor data for the layer.
func (s *Streamer) Prepare(ctx context.Context, layerIdx int) (map[string][]byte, error) {
	if layerIdx < 0 || layerIdx >= len(s.lm.Layers) {
		return nil, fmt.Errorf("layer %d out of range", layerIdx)
	}

	s.kickPrefetch(ctx, layerIdx)

	data, err := s.waitForLayer(ctx, layerIdx)
	if err != nil {
		return nil, err
	}

	for ahead := 1; ahead <= s.cfg.PrefetchAhead; ahead++ {
		next := layerIdx + ahead
		if next < len(s.lm.Layers) {
			s.kickPrefetch(ctx, next)
		}
	}

	return data, nil
}

// Release marks a layer as no longer needed, allowing its memory to be reclaimed.
func (s *Streamer) Release(layerIdx int) {
	s.mu.Lock()
	delete(s.residentData, layerIdx)
	s.mu.Unlock()
	s.budget.MarkEvicted(layerIdx)
	slog.Debug("streaming: released layer", "layer", layerIdx)
}

// Stats returns current budget usage.
func (s *Streamer) Stats() (usedBytes, availableBytes uint64) {
	return s.budget.UsedBytes(), s.budget.AvailableBytes()
}

func (s *Streamer) kickPrefetch(ctx context.Context, layerIdx int) {
	s.mu.Lock()
	if _, ok := s.residentData[layerIdx]; ok {
		s.mu.Unlock()
		return
	}
	if _, ok := s.pending[layerIdx]; ok {
		s.mu.Unlock()
		return
	}
	s.mu.Unlock()

	s.evictIfNeeded(layerIdx)
	s.budget.MarkLoading(layerIdx)
	ch := s.prefetch.Fetch(ctx, layerIdx)
	if ch != nil {
		s.mu.Lock()
		s.pending[layerIdx] = ch
		s.mu.Unlock()
	}
}

func (s *Streamer) evictIfNeeded(layerIdx int) {
	wantBytes := s.lm.Layers[layerIdx].ByteSize
	protect := map[int]bool{layerIdx: true}
	toEvict := s.budget.NeedEvict(wantBytes, protect)
	for _, idx := range toEvict {
		s.Release(idx)
		slog.Debug("streaming: evicted for budget", "evicted", idx, "for", layerIdx)
	}
}

func (s *Streamer) waitForLayer(ctx context.Context, layerIdx int) (map[string][]byte, error) {
	s.mu.Lock()
	if data, ok := s.residentData[layerIdx]; ok {
		s.mu.Unlock()
		return data, nil
	}
	ch, hasPending := s.pending[layerIdx]
	s.mu.Unlock()

	if !hasPending {
		s.kickPrefetch(ctx, layerIdx)
		s.mu.Lock()
		ch = s.pending[layerIdx]
		s.mu.Unlock()
		if ch == nil {
			return nil, fmt.Errorf("failed to start prefetch for layer %d", layerIdx)
		}
	}

	start := time.Now()
	select {
	case result := <-ch:
		s.mu.Lock()
		delete(s.pending, layerIdx)
		s.mu.Unlock()

		if result.Err != nil {
			s.budget.MarkEvicted(layerIdx)
			return nil, fmt.Errorf("prefetch layer %d: %w", layerIdx, result.Err)
		}

		s.mu.Lock()
		s.residentData[layerIdx] = result.Tensors
		s.mu.Unlock()

		if err := s.budget.MarkResident(layerIdx); err != nil {
			return nil, err
		}

		slog.Info("streaming: layer ready", "layer", layerIdx, "wait", time.Since(start),
			"read", result.ReadTime, "tensors", len(result.Tensors))
		return result.Tensors, nil

	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// RunAllLayers walks every layer sequentially: prepare → callback → release. This is the
// core "AirLLM-like" loop for GGUF models. The callback receives raw tensor bytes per layer.
func (s *Streamer) RunAllLayers(ctx context.Context, fn func(layerIdx int, tensors map[string][]byte) error) error {
	for i := range s.lm.Layers {
		data, err := s.Prepare(ctx, i)
		if err != nil {
			return fmt.Errorf("prepare layer %d: %w", i, err)
		}
		if err := fn(i, data); err != nil {
			return fmt.Errorf("compute layer %d: %w", i, err)
		}
		if i >= s.cfg.PrefetchAhead {
			s.Release(i - s.cfg.PrefetchAhead)
		}
	}
	for i := len(s.lm.Layers) - s.cfg.PrefetchAhead; i < len(s.lm.Layers); i++ {
		if i >= 0 {
			s.Release(i)
		}
	}
	return nil
}
