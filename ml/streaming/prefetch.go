package streaming

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"sync"
	"time"
)

// PrefetchResult holds the raw tensor data read from disk for one layer.
type PrefetchResult struct {
	LayerIndex int
	Tensors    map[string][]byte // tensor name → raw bytes
	Err        error
	ReadTime   time.Duration
}

// Prefetcher reads layer tensor data from a GGUF file asynchronously. It allows one or more
// layers to be in flight, bounded by concurrency. Callers submit prefetch requests and
// receive results via channels.
type Prefetcher struct {
	lm   *LayerMap
	r    io.ReaderAt
	sem  chan struct{} // bounds concurrent reads
	mu   sync.Mutex
	inflight map[int]bool
}

// NewPrefetcher creates a prefetcher reading from r (must support concurrent ReadAt, e.g. *os.File)
// with at most concurrency parallel layer reads.
func NewPrefetcher(lm *LayerMap, r io.ReaderAt, concurrency int) *Prefetcher {
	if concurrency < 1 {
		concurrency = 1
	}
	return &Prefetcher{
		lm:       lm,
		r:        r,
		sem:      make(chan struct{}, concurrency),
		inflight: make(map[int]bool),
	}
}

// Fetch starts an async read of a layer's tensors and sends the result on the returned channel.
// If the layer is already in flight, returns nil (caller should wait for the existing request).
// Respects ctx cancellation.
func (p *Prefetcher) Fetch(ctx context.Context, layerIdx int) <-chan PrefetchResult {
	p.mu.Lock()
	if p.inflight[layerIdx] {
		p.mu.Unlock()
		return nil
	}
	p.inflight[layerIdx] = true
	p.mu.Unlock()

	ch := make(chan PrefetchResult, 1)
	go func() {
		defer func() {
			p.mu.Lock()
			delete(p.inflight, layerIdx)
			p.mu.Unlock()
		}()

		select {
		case p.sem <- struct{}{}:
			defer func() { <-p.sem }()
		case <-ctx.Done():
			ch <- PrefetchResult{LayerIndex: layerIdx, Err: ctx.Err()}
			return
		}

		start := time.Now()
		tensors, err := p.lm.ReadLayerTensors(p.r, layerIdx)
		elapsed := time.Since(start)

		if err == nil {
			var totalBytes uint64
			for _, b := range tensors {
				totalBytes += uint64(len(b))
			}
			bw := float64(totalBytes) / elapsed.Seconds() / (1024 * 1024 * 1024)
			slog.Debug("prefetch complete", "layer", layerIdx, "bytes", totalBytes,
				"elapsed", elapsed, "throughput_GiBps", fmt.Sprintf("%.2f", bw))
		}

		ch <- PrefetchResult{
			LayerIndex: layerIdx,
			Tensors:    tensors,
			Err:        err,
			ReadTime:   elapsed,
		}
	}()
	return ch
}

// IsInflight reports whether a layer read is currently in progress.
func (p *Prefetcher) IsInflight(layerIdx int) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.inflight[layerIdx]
}
