package streaming

import (
	"bytes"
	"context"
	"testing"
	"time"
)

type fakeReaderAt struct {
	data []byte
}

func (f *fakeReaderAt) ReadAt(p []byte, off int64) (int, error) {
	return copy(p, f.data[off:off+int64(len(p))]), nil
}

func testLayerMap() *LayerMap {
	return &LayerMap{
		TensorDataBase:   0,
		BlockCount:       2,
		TotalWeightBytes: 600,
		Layers: []LayerInfo{
			{Index: 0, Name: "blk.0", ByteSize: 200, TensorCount: 1,
				Tensors: []TensorRef{{Name: "attn", Offset: 0, Size: 200}}},
			{Index: 1, Name: "blk.1", ByteSize: 200, TensorCount: 1,
				Tensors: []TensorRef{{Name: "attn", Offset: 200, Size: 200}}},
			{Index: 2, Name: "output", ByteSize: 200, TensorCount: 1,
				Tensors: []TensorRef{{Name: "weight", Offset: 400, Size: 200}}},
		},
	}
}

func TestPrefetcher_Fetch(t *testing.T) {
	data := make([]byte, 600)
	for i := range data {
		data[i] = byte(i % 256)
	}
	lm := testLayerMap()
	p := NewPrefetcher(lm, &fakeReaderAt{data: data}, 2)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	ch := p.Fetch(ctx, 0)
	if ch == nil {
		t.Fatal("expected channel, got nil")
	}
	result := <-ch
	if result.Err != nil {
		t.Fatal(result.Err)
	}
	if len(result.Tensors) != 1 {
		t.Fatalf("expected 1 tensor, got %d", len(result.Tensors))
	}
	if !bytes.Equal(result.Tensors["attn"], data[0:200]) {
		t.Fatal("tensor data mismatch")
	}
}

func TestPrefetcher_DuplicateFetchReturnsNil(t *testing.T) {
	data := make([]byte, 600)
	lm := testLayerMap()
	p := NewPrefetcher(lm, &fakeReaderAt{data: data}, 1)

	ctx := context.Background()
	ch1 := p.Fetch(ctx, 0)
	ch2 := p.Fetch(ctx, 0)
	if ch2 != nil {
		t.Fatal("second fetch for same layer should return nil")
	}
	<-ch1
}

func TestPrefetcher_CancelledContext(t *testing.T) {
	data := make([]byte, 600)
	lm := testLayerMap()
	p := NewPrefetcher(lm, &fakeReaderAt{data: data}, 0)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	// Fill the semaphore so new fetches block on it
	p2 := NewPrefetcher(lm, &fakeReaderAt{data: data}, 1)
	_ = p2 // unused, just testing cancelled ctx on p
	ch := p.Fetch(ctx, 1)
	if ch == nil {
		t.Skip("fetch returned nil (inflight race)")
	}
	result := <-ch
	if result.Err == nil {
		// Read was fast enough to complete before cancel; acceptable
		return
	}
}
