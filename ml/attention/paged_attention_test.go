package attention

import (
	"context"
	"testing"
	"time"
)

func TestPagedAttention_New(t *testing.T) {
	paged := NewPagedAttention(16, 1024, 1024*1024*1024)
	if paged == nil {
		t.Fatal("expected non-nil PagedAttention")
	}
	if paged.blockSize != 16 {
		t.Errorf("expected blockSize 16, got %d", paged.blockSize)
	}
	if paged.maxBlocks != 1024 {
		t.Errorf("expected maxBlocks 1024, got %d", paged.maxBlocks)
	}
}

func TestPagedAttention_AllocateBlock(t *testing.T) {
	paged := NewPagedAttention(16, 1024, 1024*1024*1024)

	blockID, err := paged.AllocateBlock()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if blockID < 0 {
		t.Error("expected valid block ID")
	}
	if paged.numAllocated != 1 {
		t.Errorf("expected 1 allocated, got %d", paged.numAllocated)
	}
}

func TestPagedAttention_FreeBlock(t *testing.T) {
	paged := NewPagedAttention(16, 1024, 1024*1024*1024)

	blockID, _ := paged.AllocateBlock()
	err := paged.FreeBlock(blockID)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestPagedAttention_AllocatePages(t *testing.T) {
	paged := NewPagedAttention(16, 1024, 1024*1024*1024)

	blocks, err := paged.AllocatePages(32)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(blocks) != 2 {
		t.Errorf("expected 2 blocks for 32 tokens, got %d", len(blocks))
	}
}

func TestPagedAttention_SetPageTable(t *testing.T) {
	paged := NewPagedAttention(16, 1024, 1024*1024*1024)
	blocks := []int{1, 2, 3}

	paged.SetPageTable(10, blocks)

	result, ok := paged.GetPageTable(10)
	if !ok {
		t.Error("expected to find page table")
	}
	if len(result) != 3 {
		t.Errorf("expected 3 blocks, got %d", len(result))
	}
}

func TestPagedAttention_NumAllocated(t *testing.T) {
	paged := NewPagedAttention(16, 1024, 1024*1024*1024)

	paged.AllocateBlock()
	paged.AllocateBlock()

	if paged.NumAllocated() != 2 {
		t.Errorf("expected 2, got %d", paged.NumAllocated())
	}
}

func TestPagedAttention_MaxBlocks(t *testing.T) {
	paged := NewPagedAttention(16, 1024, 1024*1024*1024)
	if paged.MaxBlocks() != 1024 {
		t.Errorf("expected 1024, got %d", paged.MaxBlocks())
	}
}

func TestPagedAttention_BlockSize(t *testing.T) {
	paged := NewPagedAttention(16, 1024, 1024*1024*1024)
	if paged.BlockSize() != 16 {
		t.Errorf("expected 16, got %d", paged.BlockSize())
	}
}

func TestRequest(t *testing.T) {
	req := &Request{
		ID:            "req1",
		Prompt:        "test prompt",
		Tokens:        []int{1, 2, 3},
		MaxTokens:     100,
		Priority:      1,
		CreatedAt:     time.Now(),
		ContextLength: 2048,
	}

	if req.ID != "req1" {
		t.Errorf("expected req1, got %s", req.ID)
	}
	if len(req.Tokens) != 3 {
		t.Errorf("expected 3 tokens, got %d", len(req.Tokens))
	}
}

func TestRunningRequest(t *testing.T) {
	req := &Request{
		ID:        "req1",
		Tokens:    []int{1, 2, 3},
		MaxTokens: 100,
	}

	running := &RunningRequest{
		Request:       req,
		CurrentTokens: []int{1, 2, 3},
		OutputTokens:  []int{4},
		IsComplete:    false,
	}

	if running.IsComplete {
		t.Error("expected not complete")
	}
	if len(running.OutputTokens) != 1 {
		t.Errorf("expected 1 output token, got %d", len(running.OutputTokens))
	}
}

func TestContinuousScheduler_New(t *testing.T) {
	scheduler := NewContinuousScheduler(8, 1024*1024*1024)
	if scheduler == nil {
		t.Fatal("expected non-nil ContinuousScheduler")
	}
	if scheduler.maxBatchSize != 8 {
		t.Errorf("expected maxBatchSize 8, got %d", scheduler.maxBatchSize)
	}
	if scheduler.maxGPUmemory != 1024*1024*1024 {
		t.Errorf("expected maxGPUmemory 1GB, got %d", scheduler.maxGPUmemory)
	}
}

func TestContinuousScheduler_AddRequest(t *testing.T) {
	scheduler := NewContinuousScheduler(8, 1024*1024*1024)

	req := &Request{
		ID:        "req1",
		Tokens:    []int{1, 2, 3},
		MaxTokens: 100,
	}

	err := scheduler.AddRequest(req)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestContinuousScheduler_Schedule(t *testing.T) {
	scheduler := NewContinuousScheduler(8, 1024*1024*1024)

	req := &Request{
		ID:        "req1",
		Tokens:    []int{1, 2, 3},
		MaxTokens: 100,
	}
	scheduler.AddRequest(req)

	batch, err := scheduler.Schedule()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if batch == nil {
		t.Fatal("expected non-nil batch")
	}
	if len(batch.Requests) != 1 {
		t.Errorf("expected 1 request in batch, got %d", len(batch.Requests))
	}
}

func TestContinuousScheduler_UpdateRequest(t *testing.T) {
	scheduler := NewContinuousScheduler(8, 1024*1024*1024)

	req := &Request{
		ID:        "req1",
		Tokens:    []int{1, 2, 3},
		MaxTokens: 100,
	}
	scheduler.AddRequest(req)

	scheduler.UpdateRequest("req1", []int{4}, false)

	running, waiting, preempted, completed := scheduler.GetStats()
	if running != 1 {
		t.Errorf("expected 1 running, got %d", running)
	}
	if waiting != 0 {
		t.Errorf("expected 0 waiting, got %d", waiting)
	}
	if preempted != 0 {
		t.Errorf("expected 0 preempted, got %d", preempted)
	}
	if completed != 0 {
		t.Errorf("expected 0 completed, got %d", completed)
	}
}

func TestContinuousScheduler_RemoveRequest(t *testing.T) {
	scheduler := NewContinuousScheduler(8, 1024*1024*1024)

	req := &Request{
		ID:        "req1",
		Tokens:    []int{1, 2, 3},
		MaxTokens: 100,
	}
	scheduler.AddRequest(req)

	scheduler.RemoveRequest("req1")

	running, _, _, _ := scheduler.GetStats()
	if running != 0 {
		t.Errorf("expected 0 running, got %d", running)
	}
}

func TestContinuousScheduler_Preempt(t *testing.T) {
	scheduler := NewContinuousScheduler(8, 1024*1024*1024)
	scheduler.preemptionEnabled = true

	req := &Request{
		ID:        "req1",
		Tokens:    []int{1, 2, 3},
		MaxTokens: 100,
		Priority:  10,
	}
	scheduler.AddRequest(req)

	preempted := scheduler.Preempt()
	if preempted != nil {
		t.Error("expected nil for high priority request (should not be preempted)")
	}
}

func TestContinuousScheduler_GetStats(t *testing.T) {
	scheduler := NewContinuousScheduler(8, 1024*1024*1024)

	_, _, preempted, completed := scheduler.GetStats()
	if preempted != 0 {
		t.Errorf("expected 0 preempted, got %d", preempted)
	}
	if completed != 0 {
		t.Errorf("expected 0 completed, got %d", completed)
	}
}

func TestSpeculativeDecoder_New(t *testing.T) {
	decoder := NewSpeculativeDecoder(nil, nil, 4)
	if decoder == nil {
		t.Fatal("expected non-nil SpeculativeDecoder")
	}
	if decoder.speculateLen != 4 {
		t.Errorf("expected 4, got %d", decoder.speculateLen)
	}
}

func TestSpeculativeDecoder_Generate(t *testing.T) {
	decoder := NewSpeculativeDecoder(nil, nil, 4)

	tokens, err := decoder.Generate(context.Background(), []int{1, 2, 3})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(tokens) != 4 {
		t.Errorf("expected 4 speculative tokens, got %d", len(tokens))
	}
}

func TestSpeculativeDecoder_Verify(t *testing.T) {
	decoder := NewSpeculativeDecoder(nil, nil, 4)

	verified, err := decoder.Verify([]int{1, 2, 3}, []int{1, 2, 3, 4})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(verified) != 4 {
		t.Errorf("expected 4 verified tokens, got %d", len(verified))
	}
}

func TestSpeculativeDecoder_SetSpeculateLen(t *testing.T) {
	decoder := NewSpeculativeDecoder(nil, nil, 4)
	decoder.SetSpeculateLen(8)

	if decoder.GetSpeculateLen() != 8 {
		t.Errorf("expected 8, got %d", decoder.GetSpeculateLen())
	}
}

func TestSpeculativeDecoder_GetSpeculateLen(t *testing.T) {
	decoder := NewSpeculativeDecoder(nil, nil, 4)
	if decoder.GetSpeculateLen() != 4 {
		t.Errorf("expected 4, got %d", decoder.GetSpeculateLen())
	}
}

func TestBatch(t *testing.T) {
	batch := &Batch{
		Requests: []*RunningRequest{
			{Request: &Request{ID: "req1"}},
			{Request: &Request{ID: "req2"}},
		},
		InputIDs: []int{1, 2, 3, 4},
		MaxLen:   2,
	}

	if len(batch.Requests) != 2 {
		t.Errorf("expected 2 requests, got %d", len(batch.Requests))
	}
	if len(batch.InputIDs) != 4 {
		t.Errorf("expected 4 input IDs, got %d", len(batch.InputIDs))
	}
	if batch.MaxLen != 2 {
		t.Errorf("expected MaxLen 2, got %d", batch.MaxLen)
	}
}
