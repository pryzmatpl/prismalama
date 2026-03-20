package attention

import (
	"context"
	"sync"
	"time"

	"github.com/ollama/ollama/ml"
)

type PagedAttention struct {
	blockSize     int
	maxBlocks     int
	numAllocated  int
	pageTable     map[int][]int
	kvPages       []KVPage
	mu            sync.RWMutex
	maxMemory     uint64
	currentMemory uint64
}

type KVPage struct {
	blockID  int
	isAlloc  bool
	refCount int
}

type Request struct {
	ID            string
	Prompt        string
	Tokens        []int
	MaxTokens     int
	Priority      int
	CreatedAt     time.Time
	ContextLength int
}

type RunningRequest struct {
	*Request
	CurrentTokens []int
	OutputTokens  []int
	IsComplete    bool
}

type ContinuousScheduler struct {
	maxBatchSize      int
	preemptionEnabled bool
	waitingQueue      chan *Request
	runningQueue      map[string]*RunningRequest
	mu                sync.RWMutex
	maxGPUmemory      uint64
	currentGPUmemory  uint64
	batchReady        chan *Batch
	preemptedCount    uint64
	completedCount    uint64
}

type Batch struct {
	Requests []*RunningRequest
	InputIDs []int
	MaxLen   int
}

func NewPagedAttention(blockSize, maxBlocks int, maxMemory uint64) *PagedAttention {
	return &PagedAttention{
		blockSize:    blockSize,
		maxBlocks:    maxBlocks,
		numAllocated: 0,
		pageTable:    make(map[int][]int),
		kvPages:      make([]KVPage, maxBlocks),
		maxMemory:    maxMemory,
	}
}

func (p *PagedAttention) AllocateBlock() (int, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	for i, page := range p.kvPages {
		if !page.isAlloc {
			p.kvPages[i].isAlloc = true
			p.kvPages[i].refCount = 1
			p.numAllocated++
			return i, nil
		}
	}

	return -1, nil
}

func (p *PagedAttention) FreeBlock(blockID int) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if blockID < 0 || blockID >= len(p.kvPages) {
		return nil
	}

	p.kvPages[blockID].isAlloc = false
	p.kvPages[blockID].refCount = 0
	p.numAllocated--

	return nil
}

func (p *PagedAttention) AllocatePages(numTokens int) ([]int, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	numBlocks := (numTokens + p.blockSize - 1) / p.blockSize
	blocks := make([]int, 0, numBlocks)

	for i := 0; i < numBlocks; i++ {
		found := -1
		for j, page := range p.kvPages {
			if !page.isAlloc {
				found = j
				break
			}
		}

		if found == -1 {
			return nil, nil
		}

		p.kvPages[found].isAlloc = true
		p.kvPages[found].refCount = 1
		blocks = append(blocks, found)
		p.numAllocated++
	}

	return blocks, nil
}

func (p *PagedAttention) GetPageTable(tokenPos int) ([]int, bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if pages, ok := p.pageTable[tokenPos]; ok {
		return pages, true
	}

	return nil, false
}

func (p *PagedAttention) SetPageTable(tokenPos int, blocks []int) {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.pageTable[tokenPos] = blocks
}

func (p *PagedAttention) Forward(ctx ml.Context, q, k, v *ml.Tensor, pageTable map[int][]int) (*ml.Tensor, error) {
	_ = ctx
	_ = q
	_ = k
	_ = v
	_ = pageTable

	return nil, nil
}

func (p *PagedAttention) NumAllocated() int {
	p.mu.RLock()
	defer p.mu.RUnlock()

	return p.numAllocated
}

func (p *PagedAttention) MaxBlocks() int {
	return p.maxBlocks
}

func (p *PagedAttention) BlockSize() int {
	return p.blockSize
}

func NewContinuousScheduler(maxBatchSize int, maxGPUmemory uint64) *ContinuousScheduler {
	return &ContinuousScheduler{
		maxBatchSize:      maxBatchSize,
		preemptionEnabled: true,
		waitingQueue:      make(chan *Request, maxBatchSize*2),
		runningQueue:      make(map[string]*RunningRequest),
		maxGPUmemory:      maxGPUmemory,
		batchReady:        make(chan *Batch, 10),
	}
}

func (c *ContinuousScheduler) AddRequest(req *Request) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	runningReq := &RunningRequest{
		Request:       req,
		CurrentTokens: req.Tokens,
		OutputTokens:  make([]int, 0),
		IsComplete:    false,
	}

	if c.estimateMemory(runningReq) > c.maxGPUmemory-c.currentGPUmemory {
		c.preemptedCount++
		return nil
	}

	c.runningQueue[req.ID] = runningReq
	c.currentGPUmemory += c.estimateMemory(runningReq)

	return nil
}

func (c *ContinuousScheduler) Schedule() (*Batch, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	batch := &Batch{
		Requests: make([]*RunningRequest, 0, c.maxBatchSize),
		InputIDs: make([]int, 0),
		MaxLen:   0,
	}

	for _, req := range c.runningQueue {
		if len(batch.Requests) >= c.maxBatchSize {
			break
		}

		if !req.IsComplete {
			batch.Requests = append(batch.Requests, req)
			batch.InputIDs = append(batch.InputIDs, req.CurrentTokens...)
			if len(req.CurrentTokens) > batch.MaxLen {
				batch.MaxLen = len(req.CurrentTokens)
			}
		}
	}

	if len(batch.Requests) > 0 {
		return batch, nil
	}

	return nil, nil
}

func (c *ContinuousScheduler) UpdateRequest(reqID string, outputTokens []int, isComplete bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if req, ok := c.runningQueue[reqID]; ok {
		req.OutputTokens = append(req.OutputTokens, outputTokens...)
		req.CurrentTokens = append(req.CurrentTokens, outputTokens...)

		if isComplete {
			req.IsComplete = true
			c.currentGPUmemory -= c.estimateMemory(req)
			c.completedCount++
		}
	}
}

func (c *ContinuousScheduler) RemoveRequest(reqID string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if req, ok := c.runningQueue[reqID]; ok {
		c.currentGPUmemory -= c.estimateMemory(req)
		delete(c.runningQueue, reqID)
	}
}

func (c *ContinuousScheduler) Preempt() *Request {
	if !c.preemptionEnabled {
		return nil
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	for _, req := range c.runningQueue {
		if req.Priority == 0 {
			c.currentGPUmemory -= c.estimateMemory(req)
			delete(c.runningQueue, req.ID)
			c.preemptedCount++
			return req.Request
		}
	}

	return nil
}

func (c *ContinuousScheduler) GetStats() (running, waiting, preempted, completed uint64) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	running = uint64(len(c.runningQueue))
	preempted = c.preemptedCount
	completed = c.completedCount

	return
}

func (c *ContinuousScheduler) estimateMemory(req *RunningRequest) uint64 {
	return uint64(len(req.CurrentTokens) * 4 * 1024)
}

type SpeculativeDecoder struct {
	draftModel   ml.Backend
	targetModel  ml.Backend
	speculateLen int
	mu           sync.RWMutex
}

func NewSpeculativeDecoder(draft, target ml.Backend, speculateLen int) *SpeculativeDecoder {
	return &SpeculativeDecoder{
		draftModel:   draft,
		targetModel:  target,
		speculateLen: speculateLen,
	}
}

func (s *SpeculativeDecoder) Generate(ctx context.Context, input []int) ([]int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	_ = ctx

	draftTokens := make([]int, s.speculateLen)
	for i := 0; i < s.speculateLen; i++ {
		draftTokens[i] = input[i%len(input)]
	}

	return draftTokens, nil
}

func (s *SpeculativeDecoder) Verify(input, draft []int) ([]int, error) {
	_ = input
	_ = draft
	return draft, nil
}

func (s *SpeculativeDecoder) SetSpeculateLen(len int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.speculateLen = len
}

func (s *SpeculativeDecoder) GetSpeculateLen() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.speculateLen
}
