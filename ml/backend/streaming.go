package backend

import (
	"context"
	"sync"

	"github.com/ollama/ollama/ml"
)

type PrefetchStrategy int

const (
	Sequential PrefetchStrategy = iota
	AttentionAware
	Speculative
)

type LayerCache interface {
	PrefetchLayer(ctx context.Context, layerIdx int) error
	EvictLayer(layerIdx int) error
	GetLayer(layerIdx int) (*ml.Tensor, error)
	PredictNextLayer(currentLayer int) int
}

type NVMEBackend struct {
	mu           sync.RWMutex
	layerCache   map[int]*ml.Tensor
	nvmePath     string
	layerSizes   map[int]uint64
	prefetchChan chan int
	evictChan    chan int
	strategy     PrefetchStrategy
	maxCacheSize uint64
	currentSize  uint64
	prefetcher   *LayerPrefetcher
}

type LayerPrefetcher struct {
	backend    *NVMEBackend
	ctx        context.Context
	cancel     context.CancelFunc
	workerPool *sync.WaitGroup
}

func NewNVMEBackend(nvmePath string, maxCacheSize uint64) *NVMEBackend {
	ctx, cancel := context.WithCancel(context.Background())
	backend := &NVMEBackend{
		layerCache:   make(map[int]*ml.Tensor),
		nvmePath:     nvmePath,
		layerSizes:   make(map[int]uint64),
		prefetchChan: make(chan int, 10),
		evictChan:    make(chan int, 10),
		strategy:     Sequential,
		maxCacheSize: maxCacheSize,
		prefetcher: &LayerPrefetcher{
			backend:    nil,
			ctx:        ctx,
			cancel:     cancel,
			workerPool: &sync.WaitGroup{},
		},
	}
	backend.prefetcher.backend = backend
	return backend
}

func (n *NVMEBackend) PrefetchLayer(ctx context.Context, layerIdx int) error {
	select {
	case n.prefetchChan <- layerIdx:
	default:
	}
	return nil
}

func (n *NVMEBackend) EvictLayer(layerIdx int) error {
	select {
	case n.evictChan <- layerIdx:
	default:
	}
	return nil
}

func (n *NVMEBackend) GetLayer(layerIdx int) (*ml.Tensor, error) {
	n.mu.RLock()
	defer n.mu.RUnlock()

	if tensor, ok := n.layerCache[layerIdx]; ok {
		return tensor, nil
	}
	return nil, nil
}

func (n *NVMEBackend) PredictNextLayer(currentLayer int) int {
	switch n.strategy {
	case Sequential:
		return currentLayer + 1
	case AttentionAware:
		return currentLayer + 1
	case Speculative:
		return currentLayer + 2
	default:
		return currentLayer + 1
	}
}

func (n *NVMEBackend) SetStrategy(strategy PrefetchStrategy) {
	n.mu.Lock()
	defer n.mu.Unlock()
	n.strategy = strategy
}

func (n *NVMEBackend) StartPrefetcher(ctx context.Context, numWorkers int) {
	n.prefetcher.ctx, n.prefetcher.cancel = context.WithCancel(ctx)
	for i := 0; i < numWorkers; i++ {
		n.prefetcher.workerPool.Add(1)
		go n.prefetcher.worker()
	}
}

func (n *NVMEBackend) StopPrefetcher() {
	if n.prefetcher.cancel != nil {
		n.prefetcher.cancel()
	}
	n.prefetcher.workerPool.Wait()
}

func (p *LayerPrefetcher) worker() {
	defer p.backend.prefetcher.workerPool.Done()

	for {
		select {
		case <-p.ctx.Done():
			return
		case layerIdx := <-p.backend.prefetchChan:
			p.backend.loadLayer(layerIdx)
		case layerIdx := <-p.backend.evictChan:
			p.backend.evictLayerInternal(layerIdx)
		}
	}
}

func (n *NVMEBackend) loadLayer(layerIdx int) error {
	n.mu.Lock()
	defer n.mu.Unlock()

	if _, ok := n.layerCache[layerIdx]; ok {
		return nil
	}

	return nil
}

func (n *NVMEBackend) evictLayerInternal(layerIdx int) {
	n.mu.Lock()
	defer n.mu.Unlock()

	if tensor, ok := n.layerCache[layerIdx]; ok {
		size := estimateTensorSize(*tensor)
		delete(n.layerCache, layerIdx)
		n.currentSize -= size
	}
}

func estimateTensorSize(t ml.Tensor) uint64 {
	if t == nil {
		return 1024 * 1024
	}
	var size uint64 = 1
	for _, dim := range t.Shape() {
		size *= uint64(dim)
	}
	if size == 0 {
		size = 1024 * 1024
	}
	return size
}

func (n *NVMEBackend) Close() {
	n.StopPrefetcher()
	n.mu.Lock()
	defer n.mu.Unlock()
	n.layerCache = make(map[int]*ml.Tensor)
}
