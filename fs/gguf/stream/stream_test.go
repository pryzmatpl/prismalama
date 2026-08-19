package stream

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/fs/ggml"
)

// createTestGGUFFile writes a minimal GGUF file and returns its path.
// The caller is responsible for cleaning it up.
func createTestGGUFFile(t *testing.T, tensors []*ggml.Tensor) string {
	t.Helper()

	f, err := os.CreateTemp(t.TempDir(), "test-*.gguf")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	kv := ggml.KV{
		"general.architecture":                   "llama",
		"llama.block_count":                      uint32(2),
		"llama.embedding_length":                 uint32(8),
		"llama.attention.head_count":             uint32(2),
		"llama.attention.head_count_kv":          uint32(2),
		"llama.attention.key_length":             uint32(8),
		"llama.rope.dimension_count":             uint32(8),
		"llama.rope.freq_base":                   float32(10000.0),
		"llama.rope.freq_scale":                  float32(1.0),
		"llama.attention.layer_norm_rms_epsilon": float32(1e-6),
	}

	if err := ggml.WriteGGUF(f, kv, tensors); err != nil {
		t.Fatal(err)
	}

	return f.Name()
}

// makeTestTensor creates a GGUF-compatible test tensor.
// GGUF stores tensors with 32-byte alignment, so the on-disk size may be
// larger than the logical element count * element-size.
// We use a [2,2] shape (4 F32 elements = 16 bytes, aligned to 32 bytes)
// to keep the physical size predictable and small.
func makeTestTensor(name string, shape []uint64, data byte) *ggml.Tensor {
	n := 1
	for _, v := range shape {
		n *= int(v)
	}
	// Kind 0 = TensorTypeF32; F32 NumBytes = n * 4 (4 bytes per float32).
	// GGUF pads to 32-byte alignment, so [2,2] = 16 bytes → 32 bytes on disk.
	buf := bytes.NewBuffer(make([]byte, n*4))
	for i := range buf.Bytes() {
		buf.Bytes()[i] = data
	}
	return &ggml.Tensor{Name: name, Kind: 0, Shape: shape, WriterTo: buf}
}

// TestLoaderOpen verifies that opening a file twice is idempotent.
func TestLoaderOpen(t *testing.T) {
	tensors := []*ggml.Tensor{makeTestTensor("tensor_a", []uint64{2, 2}, 0xAA)}
	path := createTestGGUFFile(t, tensors)

	l := NewLoader(1 << 20) // 1 MiB

	if err := l.Open(path); err != nil {
		t.Fatal(err)
	}
	if err := l.Open(path); err != nil {
		t.Fatal(err)
	}

	l.Close()
}

// TestLoaderCacheBasics verifies that a single tensor is cached after reading.
func TestLoaderCacheBasics(t *testing.T) {
	// F32 [2,2] = 4 elements * 4 bytes = 16 bytes logical, 32 bytes on disk.
	tensors := []*ggml.Tensor{makeTestTensor("tensor_a", []uint64{2, 2}, 0xAB)}
	path := createTestGGUFFile(t, tensors)

	l := NewLoader(1 << 20) // 1 MiB cache

	if err := l.Open(path); err != nil {
		t.Fatal(err)
	}

	data, err := l.GetTensor(path, "tensor_a")
	if err != nil {
		t.Fatalf("GetTensor: %v", err)
	}

	// After a single read the tensor must be in the cache.
	if l.cacheSize == 0 {
		t.Fatal("cacheSize is 0 after reading a tensor — addToCache was not called")
	}
	if l.cacheSize != int64(len(data)) {
		t.Fatalf("cacheSize=%d but len(data)=%d", l.cacheSize, len(data))
	}

	l.Close()
}

// TestLoaderGetTensor_CacheHit verifies that the same tensor read twice
// returns the same data both times (cache hit on second call).
func TestLoaderGetTensor_CacheHit(t *testing.T) {
	tensors := []*ggml.Tensor{makeTestTensor("tensor_a", []uint64{2, 2}, 0xAB)}
	path := createTestGGUFFile(t, tensors)

	l := NewLoader(1 << 20)

	if err := l.Open(path); err != nil {
		t.Fatal(err)
	}

	data1, err := l.GetTensor(path, "tensor_a")
	if err != nil {
		t.Fatalf("first GetTensor: %v", err)
	}
	if len(data1) == 0 {
		t.Fatalf("data1 is empty")
	}

	data2, err := l.GetTensor(path, "tensor_a")
	if err != nil {
		t.Fatalf("second GetTensor (cache hit): %v", err)
	}

	if !bytes.Equal(data1, data2) {
		t.Fatal("cache hit returned different data")
	}

	l.Close()
}

// TestLoaderLRUEviction verifies that when the cache is full, the oldest
// entry is evicted to make room for a new one. We use [2,2] tensors
// (16 bytes) and maxCache=24 so that only one tensor fits.
// This makes the LRU behavior unambiguous.
func TestLoaderLRUEviction(t *testing.T) {
	// maxCache=24, each [2,2] F32 tensor is 16 bytes → only one fits.
	l := NewLoader(24)

	// Create three separate single-tensor files so each GetTensor loads a
	// different key and we can track which tensor survives.
	pathA := createTestGGUFFile(t, []*ggml.Tensor{makeTestTensor("tensor_a", []uint64{2, 2}, 0xAA)})
	pathB := createTestGGUFFile(t, []*ggml.Tensor{makeTestTensor("tensor_a", []uint64{2, 2}, 0xBB)})
	pathC := createTestGGUFFile(t, []*ggml.Tensor{makeTestTensor("tensor_a", []uint64{2, 2}, 0xCC)})

	for _, p := range []string{pathA, pathB, pathC} {
		if err := l.Open(p); err != nil {
			t.Fatal(err)
		}
	}

	// Read A — cache now has A.
	dataA, err := l.GetTensor(pathA, "tensor_a")
	if err != nil {
		t.Fatalf("GetTensor A: %v", err)
	}
	if l.cacheSize != int64(len(dataA)) {
		t.Fatalf("after A: cacheSize=%d, want %d", l.cacheSize, len(dataA))
	}

	// Read B — cache evicts A (A+B would exceed 50), stores B.
	dataB, err := l.GetTensor(pathB, "tensor_a")
	if err != nil {
		t.Fatalf("GetTensor B: %v", err)
	}
	// Only B should be cached (A was evicted).
	if l.cacheSize != int64(len(dataB)) {
		t.Fatalf("after B: cacheSize=%d, want %d (A should be evicted)", l.cacheSize, len(dataB))
	}

	// Read C — cache evicts B (B+C would exceed 50), stores C.
	dataC, err := l.GetTensor(pathC, "tensor_a")
	if err != nil {
		t.Fatalf("GetTensor C: %v", err)
	}
	if l.cacheSize != int64(len(dataC)) {
		t.Fatalf("after C: cacheSize=%d, want %d (B should be evicted)", l.cacheSize, len(dataC))
	}

	// Re-read A — must come from disk (A was evicted), then cache A (C evicted).
	dataA2, err := l.GetTensor(pathA, "tensor_a")
	if err != nil {
		t.Fatalf("GetTensor A after eviction: %v", err)
	}
	// Cache now has A only.
	if l.cacheSize != int64(len(dataA2)) {
		t.Fatalf("after re-read A: cacheSize=%d, want %d (C should be evicted)", l.cacheSize, len(dataA2))
	}
	if !bytes.Equal(dataA, dataA2) {
		t.Fatal("re-read A returned different data")
	}

	l.Close()
}

// TestLoaderOversizedTensor verifies that a single tensor larger than maxCache
// is not stored but does not cause errors.
func TestLoaderOversizedTensor(t *testing.T) {
	// Tensor is 16 bytes, maxCache is 10 bytes — tensor is strictly larger.
	l := NewLoader(10)
	path := createTestGGUFFile(t, []*ggml.Tensor{makeTestTensor("tensor_a", []uint64{2, 2}, 0xDD)})

	if err := l.Open(path); err != nil {
		t.Fatal(err)
	}

	data, err := l.GetTensor(path, "tensor_a")
	if err != nil {
		t.Fatalf("GetTensor: %v", err)
	}
	// Data must still be returned even if not cached.
	if len(data) == 0 {
		t.Fatal("data is empty")
	}
	if l.cacheSize != 0 {
		t.Fatalf("cacheSize=%d, want 0 (oversized tensor not cached)", l.cacheSize)
	}

	l.Close()
}

// TestLoaderCloseThenGet verifies that after Close(), the cache is cleared
// but GetTensor still works after re-Open() (re-reads from disk, repopulates cache).
func TestLoaderCloseThenGet(t *testing.T) {
	tensors := []*ggml.Tensor{makeTestTensor("tensor_a", []uint64{2, 2}, 0xEE)}
	path := createTestGGUFFile(t, tensors)

	l := NewLoader(1 << 20)

	if err := l.Open(path); err != nil {
		t.Fatal(err)
	}

	data1, err := l.GetTensor(path, "tensor_a")
	if err != nil {
		t.Fatalf("GetTensor before close: %v", err)
	}

	if err := l.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Re-open after close (Close clears the file map).
	if err := l.Open(path); err != nil {
		t.Fatalf("re-Open after close: %v", err)
	}

	// GetTensor must re-read from disk without panicking.
	data2, err := l.GetTensor(path, "tensor_a")
	if err != nil {
		t.Fatalf("GetTensor after close+reopen: %v", err)
	}
	if !bytes.Equal(data1, data2) {
		t.Fatalf("data changed after re-read: %x vs %x", data1[:4], data2[:4])
	}

	l.Close()
}

// TestLoaderCloseTwice verifies that calling Close() twice does not panic.
func TestLoaderCloseTwice(t *testing.T) {
	tensors := []*ggml.Tensor{makeTestTensor("tensor_a", []uint64{2, 2}, 0xFF)}
	path := createTestGGUFFile(t, tensors)

	l := NewLoader(1 << 20)

	if err := l.Open(path); err != nil {
		t.Fatal(err)
	}

	if err := l.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := l.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

// TestLoaderListTensors verifies that ListTensors returns all tensor metadata.
func TestLoaderListTensors(t *testing.T) {
	tensors := []*ggml.Tensor{
		makeTestTensor("tensor_a", []uint64{2, 3}, 0x11),
		makeTestTensor("tensor_b", []uint64{5, 5}, 0x22),
	}
	path := createTestGGUFFile(t, tensors)

	l := NewLoader(1 << 20)
	if err := l.Open(path); err != nil {
		t.Fatal(err)
	}

	info, err := l.ListTensors(path)
	if err != nil {
		t.Fatalf("ListTensors: %v", err)
	}
	if len(info) != 2 {
		t.Fatalf("len(ListTensors)=%d, want 2", len(info))
	}

	l.Close()
}

// TestLoaderGetModelFiles verifies that GetModelFiles returns only .gguf files.
func TestLoaderGetModelFiles(t *testing.T) {
	dir := t.TempDir()
	ggufA := filepath.Join(dir, "model-00001-of-00003.gguf")
	ggufB := filepath.Join(dir, "model-00002-of-00003.gguf")
	nonGGUF := filepath.Join(dir, "README.md")

	for _, p := range []string{ggufA, ggufB, nonGGUF} {
		f, err := os.Create(p)
		if err != nil {
			t.Fatal(err)
		}
		f.Close()
	}

	l := NewLoader(1 << 20)
	files, err := l.GetModelFiles(dir)
	if err != nil {
		t.Fatalf("GetModelFiles: %v", err)
	}
	if len(files) != 2 {
		t.Fatalf("len(GetModelFiles)=%d, want 2", len(files))
	}

	l.Close()
}
