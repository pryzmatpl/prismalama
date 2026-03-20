package backend

import (
	"context"
	"testing"
)

func TestNVMEBackend_New(t *testing.T) {
	backend := NewNVMEBackend("/nvme/models", 100*1024*1024)
	if backend == nil {
		t.Fatal("expected non-nil NVMEBackend")
	}
	if backend.nvmePath != "/nvme/models" {
		t.Errorf("expected nvmePath /nvme/models, got %s", backend.nvmePath)
	}
	if backend.maxCacheSize != 100*1024*1024 {
		t.Errorf("expected maxCacheSize 100MB, got %d", backend.maxCacheSize)
	}
}

func TestNVMEBackend_PrefetchLayer(t *testing.T) {
	backend := NewNVMEBackend("/nvme/models", 100*1024*1024)
	ctx := context.Background()

	err := backend.PrefetchLayer(ctx, 5)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestNVMEBackend_EvictLayer(t *testing.T) {
	backend := NewNVMEBackend("/nvme/models", 100*1024*1024)

	err := backend.EvictLayer(5)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestNVMEBackend_GetLayer_NotFound(t *testing.T) {
	backend := NewNVMEBackend("/nvme/models", 100*1024*1024)

	tensor, err := backend.GetLayer(5)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if tensor != nil {
		t.Error("expected nil tensor for non-existent layer")
	}
}

func TestNVMEBackend_PredictNextLayer_Sequential(t *testing.T) {
	backend := NewNVMEBackend("/nvme/models", 100*1024*1024)
	backend.SetStrategy(Sequential)

	next := backend.PredictNextLayer(10)
	if next != 11 {
		t.Errorf("expected next layer 11, got %d", next)
	}
}

func TestNVMEBackend_PredictNextLayer_AttentionAware(t *testing.T) {
	backend := NewNVMEBackend("/nvme/models", 100*1024*1024)
	backend.SetStrategy(AttentionAware)

	next := backend.PredictNextLayer(10)
	if next != 11 {
		t.Errorf("expected next layer 11, got %d", next)
	}
}

func TestNVMEBackend_PredictNextLayer_Speculative(t *testing.T) {
	backend := NewNVMEBackend("/nvme/models", 100*1024*1024)
	backend.SetStrategy(Speculative)

	next := backend.PredictNextLayer(10)
	if next != 12 {
		t.Errorf("expected next layer 12, got %d", next)
	}
}

func TestNVMEBackend_SetStrategy(t *testing.T) {
	backend := NewNVMEBackend("/nvme/models", 100*1024*1024)

	backend.SetStrategy(Sequential)
	if backend.strategy != Sequential {
		t.Errorf("expected Sequential strategy")
	}

	backend.SetStrategy(AttentionAware)
	if backend.strategy != AttentionAware {
		t.Errorf("expected AttentionAware strategy")
	}

	backend.SetStrategy(Speculative)
	if backend.strategy != Speculative {
		t.Errorf("expected Speculative strategy")
	}
}

func TestNVMEBackend_Close(t *testing.T) {
	backend := NewNVMEBackend("/nvme/models", 100*1024*1024)
	backend.Close()
}
