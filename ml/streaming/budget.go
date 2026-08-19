package streaming

import (
	"fmt"
	"log/slog"
	"sync"
)

// LayerState tracks whether a layer's weights are resident (loaded into device memory).
type LayerState int

const (
	LayerEvicted  LayerState = iota // not in device memory
	LayerLoading                    // I/O in flight
	LayerResident                   // ready for compute
)

func (s LayerState) String() string {
	switch s {
	case LayerLoading:
		return "loading"
	case LayerResident:
		return "resident"
	default:
		return "evicted"
	}
}

// BudgetTracker manages a fixed byte budget for layer residency. It tracks which layers are
// loaded and decides which to evict when the budget is exceeded. Thread-safe.
type BudgetTracker struct {
	mu          sync.Mutex
	budgetBytes uint64
	usedBytes   uint64
	states      []layerEntry
}

type layerEntry struct {
	state     LayerState
	sizeBytes uint64
	seqNo     uint64 // incremented on each load; lowest seqNo evicted first (LRU)
}

// NewBudgetTracker creates a tracker for a model with the given layer sizes and a total byte budget.
func NewBudgetTracker(layerSizes []uint64, budgetBytes uint64) *BudgetTracker {
	entries := make([]layerEntry, len(layerSizes))
	for i, sz := range layerSizes {
		entries[i] = layerEntry{state: LayerEvicted, sizeBytes: sz}
	}
	return &BudgetTracker{
		budgetBytes: budgetBytes,
		states:      entries,
	}
}

// MarkLoading transitions a layer to loading state (I/O started) without consuming budget yet.
func (bt *BudgetTracker) MarkLoading(layerIdx int) error {
	bt.mu.Lock()
	defer bt.mu.Unlock()
	if layerIdx < 0 || layerIdx >= len(bt.states) {
		return fmt.Errorf("layer %d out of range", layerIdx)
	}
	bt.states[layerIdx].state = LayerLoading
	return nil
}

var budgetSeqNo uint64

// MarkResident transitions a layer to resident state, consuming budget. Returns an error if the
// layer's size alone exceeds the total budget (cannot fit even after evicting everything else).
func (bt *BudgetTracker) MarkResident(layerIdx int) error {
	bt.mu.Lock()
	defer bt.mu.Unlock()
	if layerIdx < 0 || layerIdx >= len(bt.states) {
		return fmt.Errorf("layer %d out of range", layerIdx)
	}
	e := &bt.states[layerIdx]
	if e.state == LayerResident {
		budgetSeqNo++
		e.seqNo = budgetSeqNo
		return nil
	}
	if e.sizeBytes > bt.budgetBytes {
		return fmt.Errorf("layer %d (%d bytes) exceeds total budget (%d bytes)", layerIdx, e.sizeBytes, bt.budgetBytes)
	}
	bt.usedBytes += e.sizeBytes
	e.state = LayerResident
	budgetSeqNo++
	e.seqNo = budgetSeqNo
	return nil
}

// MarkEvicted transitions a layer to evicted state, freeing budget.
func (bt *BudgetTracker) MarkEvicted(layerIdx int) {
	bt.mu.Lock()
	defer bt.mu.Unlock()
	if layerIdx < 0 || layerIdx >= len(bt.states) {
		return
	}
	e := &bt.states[layerIdx]
	if e.state == LayerResident {
		bt.usedBytes -= e.sizeBytes
	}
	e.state = LayerEvicted
}

// NeedEvict returns layer indices that should be evicted to make room for wantBytes additional
// bytes. Evicts in LRU order (lowest seqNo first). Skips layers listed in protect.
func (bt *BudgetTracker) NeedEvict(wantBytes uint64, protect map[int]bool) []int {
	bt.mu.Lock()
	defer bt.mu.Unlock()

	avail := bt.budgetBytes - bt.usedBytes
	if avail >= wantBytes {
		return nil
	}
	deficit := wantBytes - avail

	var cands []candidate
	for i, e := range bt.states {
		if e.state != LayerResident || protect[i] {
			continue
		}
		cands = append(cands, candidate{idx: i, seqNo: e.seqNo, size: e.sizeBytes})
	}
	// LRU: evict oldest first
	sortCandidates(cands)

	var toEvict []int
	var freed uint64
	for _, c := range cands {
		if freed >= deficit {
			break
		}
		toEvict = append(toEvict, c.idx)
		freed += c.size
	}
	if freed < deficit {
		slog.Warn("streaming budget: cannot free enough memory",
			"deficit", deficit, "freed", freed, "budget", bt.budgetBytes, "used", bt.usedBytes)
	}
	return toEvict
}

// State returns the current state of a layer.
func (bt *BudgetTracker) State(layerIdx int) LayerState {
	bt.mu.Lock()
	defer bt.mu.Unlock()
	if layerIdx < 0 || layerIdx >= len(bt.states) {
		return LayerEvicted
	}
	return bt.states[layerIdx].state
}

// UsedBytes returns bytes currently consumed by resident layers.
func (bt *BudgetTracker) UsedBytes() uint64 {
	bt.mu.Lock()
	defer bt.mu.Unlock()
	return bt.usedBytes
}

// AvailableBytes returns remaining budget.
func (bt *BudgetTracker) AvailableBytes() uint64 {
	bt.mu.Lock()
	defer bt.mu.Unlock()
	if bt.usedBytes >= bt.budgetBytes {
		return 0
	}
	return bt.budgetBytes - bt.usedBytes
}

func sortCandidates(cands []candidate) {
	for i := 1; i < len(cands); i++ {
		for j := i; j > 0 && cands[j].seqNo < cands[j-1].seqNo; j-- {
			cands[j], cands[j-1] = cands[j-1], cands[j]
		}
	}
}

type candidate struct {
	idx   int
	seqNo uint64
	size  uint64
}
