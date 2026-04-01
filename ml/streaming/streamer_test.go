package streaming

import (
	"context"
	"testing"
	"time"
)

func TestStreamer_RunAllLayers(t *testing.T) {
	data := make([]byte, 600)
	for i := range data {
		data[i] = byte(i % 256)
	}

	lm := testLayerMap()

	cfg := StreamerConfig{
		BudgetBytes:   300,
		PrefetchAhead: 1,
		IOConcurrency: 2,
	}

	s := &Streamer{
		lm:           lm,
		budget:       NewBudgetTracker([]uint64{200, 200, 200}, cfg.BudgetBytes),
		prefetch:     NewPrefetcher(lm, &fakeReaderAt{data: data}, cfg.IOConcurrency),
		cfg:          cfg,
		residentData: make(map[int]map[string][]byte),
		pending:      make(map[int]<-chan PrefetchResult),
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	var visited []int
	err := s.RunAllLayers(ctx, func(layerIdx int, tensors map[string][]byte) error {
		visited = append(visited, layerIdx)
		if len(tensors) == 0 {
			t.Fatalf("layer %d: no tensors", layerIdx)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}

	if len(visited) != 3 {
		t.Fatalf("expected 3 layers visited, got %d: %v", len(visited), visited)
	}
	for i, v := range visited {
		if v != i {
			t.Fatalf("expected sequential visit, got %v", visited)
		}
	}
}

func TestStreamer_BudgetEviction(t *testing.T) {
	data := make([]byte, 600)
	lm := testLayerMap()

	cfg := StreamerConfig{
		BudgetBytes:   250, // only fits 1 layer at a time
		PrefetchAhead: 0,
		IOConcurrency: 1,
	}

	s := &Streamer{
		lm:           lm,
		budget:       NewBudgetTracker([]uint64{200, 200, 200}, cfg.BudgetBytes),
		prefetch:     NewPrefetcher(lm, &fakeReaderAt{data: data}, cfg.IOConcurrency),
		cfg:          cfg,
		residentData: make(map[int]map[string][]byte),
		pending:      make(map[int]<-chan PrefetchResult),
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	maxResident := uint64(0)
	err := s.RunAllLayers(ctx, func(layerIdx int, tensors map[string][]byte) error {
		used := s.budget.UsedBytes()
		if used > maxResident {
			maxResident = used
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if maxResident > cfg.BudgetBytes {
		t.Fatalf("peak resident %d exceeded budget %d", maxResident, cfg.BudgetBytes)
	}
}

func TestStreamer_PrepareOutOfRange(t *testing.T) {
	data := make([]byte, 600)
	lm := testLayerMap()
	cfg := DefaultStreamerConfig()
	s := &Streamer{
		lm:           lm,
		budget:       NewBudgetTracker([]uint64{200, 200, 200}, cfg.BudgetBytes),
		prefetch:     NewPrefetcher(lm, &fakeReaderAt{data: data}, cfg.IOConcurrency),
		cfg:          cfg,
		residentData: make(map[int]map[string][]byte),
		pending:      make(map[int]<-chan PrefetchResult),
	}

	ctx := context.Background()
	_, err := s.Prepare(ctx, -1)
	if err == nil {
		t.Fatal("expected error for negative layer index")
	}
	_, err = s.Prepare(ctx, 999)
	if err == nil {
		t.Fatal("expected error for out-of-range layer index")
	}
}

func TestStreamer_Stats(t *testing.T) {
	data := make([]byte, 600)
	lm := testLayerMap()
	cfg := DefaultStreamerConfig()
	s := &Streamer{
		lm:           lm,
		budget:       NewBudgetTracker([]uint64{200, 200, 200}, cfg.BudgetBytes),
		prefetch:     NewPrefetcher(lm, &fakeReaderAt{data: data}, cfg.IOConcurrency),
		cfg:          cfg,
		residentData: make(map[int]map[string][]byte),
		pending:      make(map[int]<-chan PrefetchResult),
	}

	used, avail := s.Stats()
	if used != 0 {
		t.Fatalf("initial used: %d", used)
	}
	if avail != cfg.BudgetBytes {
		t.Fatalf("initial available: %d", avail)
	}
}
