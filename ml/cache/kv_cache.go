package cache

import (
	"container/list"
	"crypto/sha256"
	"sync"

	"github.com/ollama/ollama/ml"
)

type CacheLocation struct {
	LayerIdx    int
	Position    int
	BlockOffset int
}

type PrefixCache struct {
	mu             sync.RWMutex
	prefixIndex    map[uint64]map[int]CacheLocation
	accessOrder    *list.List
	sharedPrefixes []uint64
	maxEntries     int
	hits           uint64
	misses         uint64
}

func NewPrefixCache(maxEntries int) *PrefixCache {
	return &PrefixCache{
		prefixIndex:    make(map[uint64]map[int]CacheLocation),
		accessOrder:    list.New(),
		sharedPrefixes: make([]uint64, 0),
		maxEntries:     maxEntries,
	}
}

func (p *PrefixCache) ComputeHash(tokens []int) uint64 {
	hasher := sha256.New()
	for _, t := range tokens {
		hasher.Write([]byte{byte(t & 0xFF), byte((t >> 8) & 0xFF)})
	}
	hashBytes := hasher.Sum(nil)
	return uint64(hashBytes[0]) | uint64(hashBytes[1])<<8 | uint64(hashBytes[2])<<16 | uint64(hashBytes[3])<<24
}

func (p *PrefixCache) Get(prefixTokens []int, layerIdx int) (CacheLocation, bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	hash := p.ComputeHash(prefixTokens)

	if layerMap, ok := p.prefixIndex[hash]; ok {
		if loc, ok := layerMap[layerIdx]; ok {
			p.updateAccessOrder(hash)
			p.hits++
			return loc, true
		}
	}

	p.misses++
	return CacheLocation{}, false
}

func (p *PrefixCache) Put(prefixTokens []int, layerIdx int, location CacheLocation) {
	p.mu.Lock()
	defer p.mu.Unlock()

	hash := p.ComputeHash(prefixTokens)

	if _, ok := p.prefixIndex[hash]; !ok {
		p.prefixIndex[hash] = make(map[int]CacheLocation)
		if len(p.prefixIndex) > p.maxEntries {
			p.evictOldest()
		}
	}

	p.prefixIndex[hash][layerIdx] = location
	p.updateAccessOrder(hash)
}

func (p *PrefixCache) Remove(prefixTokens []int, layerIdx int) {
	p.mu.Lock()
	defer p.mu.Unlock()

	hash := p.ComputeHash(prefixTokens)

	if layerMap, ok := p.prefixIndex[hash]; ok {
		delete(layerMap, layerIdx)
		if len(layerMap) == 0 {
			delete(p.prefixIndex, hash)
		}
	}
}

func (p *PrefixCache) updateAccessOrder(hash uint64) {
	for e := p.accessOrder.Front(); e != nil; e = e.Next() {
		if e.Value == hash {
			p.accessOrder.MoveToBack(e)
			return
		}
	}
	p.accessOrder.PushBack(hash)
}

func (p *PrefixCache) evictOldest() {
	if p.accessOrder.Front() != nil {
		oldest := p.accessOrder.Front().Value.(uint64)
		delete(p.prefixIndex, oldest)
		p.accessOrder.Remove(p.accessOrder.Front())
	}
}

func (p *PrefixCache) GetStats() (hits, misses uint64, hitRate float64) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	total := p.hits + p.misses
	if total == 0 {
		return 0, 0, 0
	}
	return p.hits, p.misses, float64(p.hits) / float64(total)
}

func (p *PrefixCache) Clear() {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.prefixIndex = make(map[uint64]map[int]CacheLocation)
	p.accessOrder = list.New()
}

type SparseAttention struct {
	windowSize       int
	globalTokens     int
	blockPattern     []int
	useSlidingWindow bool
}

func NewSparseAttention(windowSize int, globalTokens int) *SparseAttention {
	return &SparseAttention{
		windowSize:       windowSize,
		globalTokens:     globalTokens,
		blockPattern:     []int{16, 16, 16},
		useSlidingWindow: windowSize > 0,
	}
}

func (s *SparseAttention) Forward(ctx ml.Context, q, k, v *ml.Tensor, attnMask *ml.Tensor) (*ml.Tensor, error) {
	_ = ctx
	_ = q
	_ = k
	_ = v
	_ = attnMask
	return nil, nil
}

func (s *SparseAttention) SetWindowSize(size int) {
	s.windowSize = size
	s.useSlidingWindow = size > 0
}

func (s *SparseAttention) SetGlobalTokens(tokens int) {
	s.globalTokens = tokens
}

func (s *SparseAttention) SetBlockPattern(pattern []int) {
	s.blockPattern = pattern
}

func (s *SparseAttention) IsSlidingWindow() bool {
	return s.useSlidingWindow
}

func (s *SparseAttention) GetWindowSize() int {
	return s.windowSize
}

func (s *SparseAttention) GetGlobalTokens() int {
	return s.globalTokens
}

type KVCacheCompressor struct {
	mu              sync.RWMutex
	compressionRate float32
	enabled         bool
}

func NewKVCacheCompressor(compressionRate float32) *KVCacheCompressor {
	return &KVCacheCompressor{
		compressionRate: compressionRate,
		enabled:         compressionRate > 0,
	}
}

func (c *KVCacheCompressor) Compress(kvData []float32) ([]float32, error) {
	if !c.enabled {
		return kvData, nil
	}

	_ = c.compressionRate
	return kvData, nil
}

func (c *KVCacheCompressor) Decompress(compressedData []float32) ([]float32, error) {
	if !c.enabled {
		return compressedData, nil
	}

	return compressedData, nil
}

func (c *KVCacheCompressor) SetCompressionRate(rate float32) {
	c.compressionRate = rate
	c.enabled = rate > 0
}

func (c *KVCacheCompressor) IsEnabled() bool {
	return c.enabled
}

type FlashDecode struct {
	blockSize    int
	maxBlocks    int
	numAllocated int
	pageTable    map[int][]int
	kvPages      []KVPage
	mu           sync.RWMutex
}

type KVPage struct {
	blockID  int
	isAlloc  bool
	refCount int
}

func NewFlashDecode(blockSize, maxBlocks int) *FlashDecode {
	return &FlashDecode{
		blockSize:    blockSize,
		maxBlocks:    maxBlocks,
		numAllocated: 0,
		pageTable:    make(map[int][]int),
		kvPages:      make([]KVPage, maxBlocks),
	}
}

func (f *FlashDecode) AllocateBlock() (int, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	for i, page := range f.kvPages {
		if !page.isAlloc {
			f.kvPages[i].isAlloc = true
			f.kvPages[i].refCount = 1
			f.numAllocated++
			return i, nil
		}
	}

	return -1, nil
}

func (f *FlashDecode) FreeBlock(blockID int) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	if blockID < 0 || blockID >= len(f.kvPages) {
		return nil
	}

	f.kvPages[blockID].isAlloc = false
	f.kvPages[blockID].refCount = 0
	f.numAllocated--

	return nil
}

func (f *FlashDecode) GetPageTable(tokenPos int) ([]int, bool) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	if pages, ok := f.pageTable[tokenPos]; ok {
		return pages, true
	}

	return nil, false
}

func (f *FlashDecode) SetPageTable(tokenPos int, blocks []int) {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.pageTable[tokenPos] = blocks
}

func (f *FlashDecode) NumAllocated() int {
	f.mu.RLock()
	defer f.mu.RUnlock()

	return f.numAllocated
}

func (f *FlashDecode) MaxBlocks() int {
	return f.maxBlocks
}
