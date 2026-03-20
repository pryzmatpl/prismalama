package cache

import (
	"testing"
)

func TestPrefixCache_New(t *testing.T) {
	cache := NewPrefixCache(100)
	if cache == nil {
		t.Fatal("expected non-nil PrefixCache")
	}
	if cache.maxEntries != 100 {
		t.Errorf("expected maxEntries 100, got %d", cache.maxEntries)
	}
}

func TestPrefixCache_ComputeHash(t *testing.T) {
	cache := NewPrefixCache(100)
	hash1 := cache.ComputeHash([]int{1, 2, 3})
	hash2 := cache.ComputeHash([]int{1, 2, 3})
	hash3 := cache.ComputeHash([]int{1, 2, 4})

	if hash1 != hash2 {
		t.Error("expected same hash for same input")
	}
	if hash1 == hash3 {
		t.Error("expected different hash for different input")
	}
}

func TestPrefixCache_Put_Get(t *testing.T) {
	cache := NewPrefixCache(100)

	cache.Put([]int{1, 2, 3}, 0, CacheLocation{LayerIdx: 0, Position: 10})

	loc, ok := cache.Get([]int{1, 2, 3}, 0)
	if !ok {
		t.Error("expected to find cached value")
	}
	if loc.LayerIdx != 0 {
		t.Errorf("expected LayerIdx 0, got %d", loc.LayerIdx)
	}
	if loc.Position != 10 {
		t.Errorf("expected Position 10, got %d", loc.Position)
	}
}

func TestPrefixCache_Get_NotFound(t *testing.T) {
	cache := NewPrefixCache(100)

	_, ok := cache.Get([]int{1, 2, 3}, 0)
	if ok {
		t.Error("expected not to find nonexistent key")
	}
}

func TestPrefixCache_Remove(t *testing.T) {
	cache := NewPrefixCache(100)

	cache.Put([]int{1, 2, 3}, 0, CacheLocation{LayerIdx: 0, Position: 10})
	cache.Remove([]int{1, 2, 3}, 0)

	_, ok := cache.Get([]int{1, 2, 3}, 0)
	if ok {
		t.Error("expected not to find removed value")
	}
}

func TestPrefixCache_GetStats(t *testing.T) {
	cache := NewPrefixCache(100)

	cache.Put([]int{1, 2, 3}, 0, CacheLocation{LayerIdx: 0, Position: 10})
	cache.Get([]int{1, 2, 3}, 0)
	cache.Get([]int{1, 2, 3}, 0)
	cache.Get([]int{1, 2, 4}, 0)

	hits, misses, hitRate := cache.GetStats()
	if hits != 2 {
		t.Errorf("expected 2 hits, got %d", hits)
	}
	if misses != 1 {
		t.Errorf("expected 1 miss, got %d", misses)
	}
	if hitRate != 2.0/3.0 {
		t.Errorf("expected hitRate 2/3, got %f", hitRate)
	}
}

func TestPrefixCache_Clear(t *testing.T) {
	cache := NewPrefixCache(100)

	cache.Put([]int{1, 2, 3}, 0, CacheLocation{LayerIdx: 0, Position: 10})
	cache.Clear()

	_, ok := cache.Get([]int{1, 2, 3}, 0)
	if ok {
		t.Error("expected not to find value after clear")
	}
}

func TestSparseAttention_New(t *testing.T) {
	attn := NewSparseAttention(1024, 8)
	if attn == nil {
		t.Fatal("expected non-nil SparseAttention")
	}
	if attn.windowSize != 1024 {
		t.Errorf("expected windowSize 1024, got %d", attn.windowSize)
	}
	if attn.globalTokens != 8 {
		t.Errorf("expected globalTokens 8, got %d", attn.globalTokens)
	}
}

func TestSparseAttention_SetWindowSize(t *testing.T) {
	attn := NewSparseAttention(1024, 8)
	attn.SetWindowSize(2048)

	if attn.windowSize != 2048 {
		t.Errorf("expected windowSize 2048, got %d", attn.windowSize)
	}
	if !attn.useSlidingWindow {
		t.Error("expected sliding window to be enabled")
	}
}

func TestSparseAttention_IsSlidingWindow(t *testing.T) {
	attn := NewSparseAttention(0, 8)
	if attn.IsSlidingWindow() {
		t.Error("expected no sliding window for windowSize 0")
	}

	attn.SetWindowSize(1024)
	if !attn.IsSlidingWindow() {
		t.Error("expected sliding window for windowSize > 0")
	}
}

func TestSparseAttention_GetWindowSize(t *testing.T) {
	attn := NewSparseAttention(1024, 8)
	if attn.GetWindowSize() != 1024 {
		t.Errorf("expected 1024, got %d", attn.GetWindowSize())
	}
}

func TestSparseAttention_GetGlobalTokens(t *testing.T) {
	attn := NewSparseAttention(1024, 8)
	if attn.GetGlobalTokens() != 8 {
		t.Errorf("expected 8, got %d", attn.GetGlobalTokens())
	}
}

func TestSparseAttention_SetBlockPattern(t *testing.T) {
	attn := NewSparseAttention(1024, 8)
	attn.SetBlockPattern([]int{8, 16, 32})

	if len(attn.blockPattern) != 3 {
		t.Errorf("expected 3 elements, got %d", len(attn.blockPattern))
	}
}

func TestKVCacheCompressor_New(t *testing.T) {
	comp := NewKVCacheCompressor(0.5)
	if comp == nil {
		t.Fatal("expected non-nil KVCacheCompressor")
	}
	if !comp.IsEnabled() {
		t.Error("expected compressor to be enabled")
	}
	if comp.compressionRate != 0.5 {
		t.Errorf("expected 0.5, got %f", comp.compressionRate)
	}
}

func TestKVCacheCompressor_NewDisabled(t *testing.T) {
	comp := NewKVCacheCompressor(0.0)
	if comp.IsEnabled() {
		t.Error("expected compressor to be disabled")
	}
}

func TestKVCacheCompressor_SetCompressionRate(t *testing.T) {
	comp := NewKVCacheCompressor(0.0)
	comp.SetCompressionRate(0.8)

	if !comp.IsEnabled() {
		t.Error("expected compressor to be enabled")
	}
	if comp.compressionRate != 0.8 {
		t.Errorf("expected 0.8, got %f", comp.compressionRate)
	}
}

func TestKVCacheCompressor_Compress(t *testing.T) {
	comp := NewKVCacheCompressor(0.5)
	data := []float32{1.0, 2.0, 3.0}

	result, err := comp.Compress(data)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(result) != 3 {
		t.Errorf("expected 3 elements, got %d", len(result))
	}
}

func TestKVCacheCompressor_Decompress(t *testing.T) {
	comp := NewKVCacheCompressor(0.5)
	data := []float32{1.0, 2.0, 3.0}

	result, err := comp.Decompress(data)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(result) != 3 {
		t.Errorf("expected 3 elements, got %d", len(result))
	}
}

func TestFlashDecode_New(t *testing.T) {
	flash := NewFlashDecode(16, 1024)
	if flash == nil {
		t.Fatal("expected non-nil FlashDecode")
	}
	if flash.blockSize != 16 {
		t.Errorf("expected blockSize 16, got %d", flash.blockSize)
	}
	if flash.maxBlocks != 1024 {
		t.Errorf("expected maxBlocks 1024, got %d", flash.maxBlocks)
	}
}

func TestFlashDecode_AllocateBlock(t *testing.T) {
	flash := NewFlashDecode(16, 1024)

	blockID, err := flash.AllocateBlock()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if blockID < 0 {
		t.Error("expected valid block ID")
	}
}

func TestFlashDecode_FreeBlock(t *testing.T) {
	flash := NewFlashDecode(16, 1024)

	blockID, _ := flash.AllocateBlock()
	err := flash.FreeBlock(blockID)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestFlashDecode_NumAllocated(t *testing.T) {
	flash := NewFlashDecode(16, 1024)

	flash.AllocateBlock()
	flash.AllocateBlock()

	if flash.NumAllocated() != 2 {
		t.Errorf("expected 2 allocated blocks, got %d", flash.NumAllocated())
	}
}

func TestFlashDecode_MaxBlocks(t *testing.T) {
	flash := NewFlashDecode(16, 1024)
	if flash.MaxBlocks() != 1024 {
		t.Errorf("expected 1024, got %d", flash.MaxBlocks())
	}
}

func TestFlashDecode_SetPageTable(t *testing.T) {
	flash := NewFlashDecode(16, 1024)
	blocks := []int{1, 2, 3}

	flash.SetPageTable(10, blocks)

	result, ok := flash.GetPageTable(10)
	if !ok {
		t.Error("expected to find page table")
	}
	if len(result) != 3 {
		t.Errorf("expected 3 blocks, got %d", len(result))
	}
}

func TestFlashDecode_GetPageTable_NotFound(t *testing.T) {
	flash := NewFlashDecode(16, 1024)

	_, ok := flash.GetPageTable(999)
	if ok {
		t.Error("expected not to find nonexistent page table")
	}
}
