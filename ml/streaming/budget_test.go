package streaming

import (
	"testing"
)

func TestBudgetTracker_BasicResidency(t *testing.T) {
	sizes := []uint64{100, 200, 300}
	bt := NewBudgetTracker(sizes, 400)

	if bt.UsedBytes() != 0 {
		t.Fatalf("initial used: %d", bt.UsedBytes())
	}
	if bt.AvailableBytes() != 400 {
		t.Fatalf("initial available: %d", bt.AvailableBytes())
	}

	if err := bt.MarkResident(0); err != nil {
		t.Fatal(err)
	}
	if bt.UsedBytes() != 100 {
		t.Fatalf("after layer 0: used=%d", bt.UsedBytes())
	}
	if bt.State(0) != LayerResident {
		t.Fatalf("layer 0 state: %v", bt.State(0))
	}

	if err := bt.MarkResident(1); err != nil {
		t.Fatal(err)
	}
	if bt.UsedBytes() != 300 {
		t.Fatalf("after layer 0+1: used=%d", bt.UsedBytes())
	}

	bt.MarkEvicted(0)
	if bt.UsedBytes() != 200 {
		t.Fatalf("after evict 0: used=%d", bt.UsedBytes())
	}
	if bt.State(0) != LayerEvicted {
		t.Fatalf("layer 0 after evict: %v", bt.State(0))
	}
}

func TestBudgetTracker_ExceedsBudget(t *testing.T) {
	sizes := []uint64{500}
	bt := NewBudgetTracker(sizes, 100)

	if err := bt.MarkResident(0); err == nil {
		t.Fatal("expected error when layer exceeds budget")
	}
}

func TestBudgetTracker_NeedEvict_LRU(t *testing.T) {
	sizes := []uint64{100, 100, 100}
	bt := NewBudgetTracker(sizes, 250)

	bt.MarkResident(0)
	bt.MarkResident(1)

	toEvict := bt.NeedEvict(100, map[int]bool{1: true})
	if len(toEvict) != 1 || toEvict[0] != 0 {
		t.Fatalf("expected evict [0], got %v", toEvict)
	}
}

func TestBudgetTracker_NeedEvict_NoneNeeded(t *testing.T) {
	sizes := []uint64{100, 100}
	bt := NewBudgetTracker(sizes, 500)
	bt.MarkResident(0)

	toEvict := bt.NeedEvict(100, nil)
	if len(toEvict) != 0 {
		t.Fatalf("expected no evictions, got %v", toEvict)
	}
}

func TestBudgetTracker_DoubleResident(t *testing.T) {
	sizes := []uint64{100}
	bt := NewBudgetTracker(sizes, 200)
	bt.MarkResident(0)
	bt.MarkResident(0)
	if bt.UsedBytes() != 100 {
		t.Fatalf("double resident should not double-count: %d", bt.UsedBytes())
	}
}

func TestBudgetTracker_EvictNonResident(t *testing.T) {
	sizes := []uint64{100}
	bt := NewBudgetTracker(sizes, 200)
	bt.MarkEvicted(0) // no-op
	if bt.UsedBytes() != 0 {
		t.Fatalf("evict non-resident changed used: %d", bt.UsedBytes())
	}
}

func TestLayerState_String(t *testing.T) {
	cases := map[LayerState]string{
		LayerEvicted:  "evicted",
		LayerLoading:  "loading",
		LayerResident: "resident",
	}
	for s, want := range cases {
		if s.String() != want {
			t.Errorf("LayerState(%d).String() = %q, want %q", s, s.String(), want)
		}
	}
}
