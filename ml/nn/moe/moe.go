package moe

import (
	"context"
	"sync"

	"github.com/ollama/ollama/ml"
)

type MoEConfig struct {
	NumExperts      int
	TopKExperts     int
	RoutingStrategy string
	ExpertsOnDisk   bool
	ExpertCacheSize int
}

type ExpertRouter interface {
	Route(tokens []int) ([]int, []float32)
	LoadExpert(expertID int) error
	EvictExpert(expertID int) error
}

type Expert struct {
	ID         int
	UpWeight   ml.Tensor
	GateWeight ml.Tensor
	DownWeight ml.Tensor
	Loaded     bool
}

type StreamingMoE struct {
	config        MoEConfig
	expertCache   *LRUCache
	expertDisk    string
	activeExperts []int
	experts       []*Expert
	mu            sync.RWMutex
	router        ExpertRouter
	auxLoss       float32
}

type LRUCache struct {
	mu       sync.Mutex
	items    map[int]*cacheItem
	head     *cacheItem
	tail     *cacheItem
	capacity int
	size     int
}

type cacheItem struct {
	key        int
	value      interface{}
	prev       *cacheItem
	next       *cacheItem
	accessTime int64
}

func NewLRUCache(capacity int) *LRUCache {
	return &LRUCache{
		items:    make(map[int]*cacheItem),
		capacity: capacity,
	}
}

func (l *LRUCache) Get(key int) (interface{}, bool) {
	l.mu.Lock()
	defer l.mu.Unlock()

	if item, ok := l.items[key]; ok {
		l.moveToFront(item)
		return item.value, true
	}

	return nil, false
}

func (l *LRUCache) Put(key int, value interface{}) {
	l.mu.Lock()
	defer l.mu.Unlock()

	if item, ok := l.items[key]; ok {
		item.value = value
		l.moveToFront(item)
		return
	}

	if l.size >= l.capacity {
		l.evictOldest()
	}

	newItem := &cacheItem{
		key:        key,
		value:      value,
		accessTime: 0,
	}

	l.items[key] = newItem
	l.addToFront(newItem)
	l.size++
}

func (l *LRUCache) Remove(key int) {
	l.mu.Lock()
	defer l.mu.Unlock()

	if item, ok := l.items[key]; ok {
		l.remove(item)
		delete(l.items, key)
		l.size--
	}
}

func (l *LRUCache) moveToFront(item *cacheItem) {
	l.remove(item)
	l.addToFront(item)
}

func (l *LRUCache) addToFront(item *cacheItem) {
	item.prev = nil
	item.next = l.head
	if l.head != nil {
		l.head.prev = item
	}
	l.head = item
	if l.tail == nil {
		l.tail = item
	}
}

func (l *LRUCache) remove(item *cacheItem) {
	if item.prev != nil {
		item.prev.next = item.next
	} else {
		l.head = item.next
	}

	if item.next != nil {
		item.next.prev = item.prev
	} else {
		l.tail = item.prev
	}
}

func (l *LRUCache) evictOldest() {
	if l.tail != nil {
		oldest := l.tail
		l.remove(l.tail)
		delete(l.items, oldest.key)
		l.size--
	}
}

func NewStreamingMoE(config MoEConfig, expertDisk string) *StreamingMoE {
	moe := &StreamingMoE{
		config:        config,
		expertCache:   NewLRUCache(config.ExpertCacheSize),
		expertDisk:    expertDisk,
		activeExperts: make([]int, 0),
		experts:       make([]*Expert, config.NumExperts),
	}

	for i := 0; i < config.NumExperts; i++ {
		moe.experts[i] = &Expert{
			ID:     i,
			Loaded: false,
		}
	}

	return moe
}

func (s *StreamingMoE) Forward(ctx context.Context, input ml.Tensor) (ml.Tensor, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	expertIDs, auxLoss := s.route(input)
	s.auxLoss = 0
	for _, loss := range auxLoss {
		s.auxLoss += loss
	}

	var output ml.Tensor

	_ = ctx
	_ = expertIDs

	return output, nil
}

func (s *StreamingMoE) route(input ml.Tensor) ([]int, []float32) {
	expertIDs := make([]int, s.config.TopKExperts)
	auxLosses := make([]float32, s.config.TopKExperts)

	for i := 0; i < s.config.TopKExperts; i++ {
		expertIDs[i] = i % s.config.NumExperts
		auxLosses[i] = 0
	}

	return expertIDs, auxLosses
}

func (s *StreamingMoE) LoadExpert(expertID int) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if expertID < 0 || expertID >= len(s.experts) {
		return nil
	}

	if s.experts[expertID].Loaded {
		return nil
	}

	expert := s.experts[expertID]
	expert.Loaded = true

	s.expertCache.Put(expertID, expert)
	s.activeExperts = append(s.activeExperts, expertID)

	return nil
}

func (s *StreamingMoE) EvictExpert(expertID int) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if expertID < 0 || expertID >= len(s.experts) {
		return nil
	}

	expert := s.experts[expertID]
	if expert.Loaded {
		expert.Loaded = false
		s.expertCache.Remove(expertID)

		for i, id := range s.activeExperts {
			if id == expertID {
				s.activeExperts = append(s.activeExperts[:i], s.activeExperts[i+1:]...)
				break
			}
		}
	}

	return nil
}

func (s *StreamingMoE) GetAuxLoss() float32 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.auxLoss
}

func (s *StreamingMoE) GetActiveExperts() []int {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := make([]int, len(s.activeExperts))
	copy(result, s.activeExperts)
	return result
}

type TopKRouter struct {
	config        MoEConfig
	loadBalancing float32
}

func NewTopKRouter(config MoEConfig) *TopKRouter {
	return &TopKRouter{
		config:        config,
		loadBalancing: 0.0,
	}
}

func (r *TopKRouter) Route(tokens []int) ([]int, []float32) {
	expertIDs := make([]int, r.config.TopKExperts)
	auxLosses := make([]float32, r.config.TopKExperts)

	for i := 0; i < r.config.TopKExperts; i++ {
		expertIDs[i] = i % r.config.NumExperts
		auxLosses[i] = 0.0
	}

	return expertIDs, auxLosses
}

func (r *TopKRouter) UpdateLoadBalancing(loss float32) {
	r.loadBalancing = loss
}

type ExpertCache struct {
	diskPath     string
	mmapCache    map[int][]byte
	mu           sync.RWMutex
	maxCacheSize uint64
	currentSize  uint64
}

func NewExpertCache(diskPath string, maxCacheSize uint64) *ExpertCache {
	return &ExpertCache{
		diskPath:     diskPath,
		mmapCache:    make(map[int][]byte),
		maxCacheSize: maxCacheSize,
	}
}

func (e *ExpertCache) LoadExpert(expertID int) ([]byte, error) {
	e.mu.RLock()
	if data, ok := e.mmapCache[expertID]; ok {
		e.mu.RUnlock()
		return data, nil
	}
	e.mu.RUnlock()

	return nil, nil
}

func (e *ExpertCache) CacheExpert(expertID int, data []byte) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	size := uint64(len(data))
	if e.currentSize+size > e.maxCacheSize {
		e.evictOldest(size)
	}

	e.mmapCache[expertID] = data
	e.currentSize += size

	return nil
}

func (e *ExpertCache) evictOldest(neededSize uint64) {
	for id := range e.mmapCache {
		delete(e.mmapCache, id)
		e.currentSize = 0
		break
	}
}

func (e *ExpertCache) Clear() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.mmapCache = make(map[int][]byte)
	e.currentSize = 0
}

type FP8MoEKernels struct {
	enabled bool
}

func NewFP8MoEKernels() *FP8MoEKernels {
	return &FP8MoEKernels{
		enabled: true,
	}
}

func (f *FP8MoEKernels) Forward(ctx context.Context, input ml.Tensor, experts []*Expert, topK int) (ml.Tensor, error) {
	_ = ctx
	_ = input
	_ = experts
	_ = topK
	return nil, nil
}

func (f *FP8MoEKernels) IsEnabled() bool {
	return f.enabled
}

func (f *FP8MoEKernels) SetEnabled(enabled bool) {
	f.enabled = enabled
}
