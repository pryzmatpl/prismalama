package ml

import (
	"testing"
)

func TestDeviceCapability(t *testing.T) {
	cap := DeviceCapability{
		DeviceID:          DeviceID{ID: "0", Library: "CUDA"},
		ComputeType:       "fp16",
		MemoryBandwidth:   500.0,
		ComputeThroughput: 10000.0,
		LatencyProfile:    "low-latency",
	}

	if cap.ComputeType != "fp16" {
		t.Errorf("expected fp16, got %s", cap.ComputeType)
	}
	if cap.MemoryBandwidth != 500.0 {
		t.Errorf("expected 500.0, got %f", cap.MemoryBandwidth)
	}
}

func TestOffloadPolicy(t *testing.T) {
	policy := OffloadPolicy{
		LayerDevices: []DeviceID{
			{ID: "0", Library: "CUDA"},
			{ID: "1", Library: "CUDA"},
		},
		KVCacheOnCPU: false,
		AttentionOn:  DeviceID{ID: "0", Library: "CUDA"},
		EmbeddingOn:  DeviceID{ID: "cpu", Library: "cpu"},
	}

	if len(policy.LayerDevices) != 2 {
		t.Errorf("expected 2 layer devices, got %d", len(policy.LayerDevices))
	}
	if policy.KVCacheOnCPU != false {
		t.Error("expected KVCacheOnCPU to be false")
	}
}

func TestHeterogeneousScheduler_New(t *testing.T) {
	scheduler := NewHeterogeneousScheduler()
	if scheduler == nil {
		t.Fatal("expected non-nil HeterogeneousScheduler")
	}
	if scheduler.deviceCapabilities == nil {
		t.Error("expected non-nil deviceCapabilities map")
	}
	if scheduler.layerCosts == nil {
		t.Error("expected non-nil layerCosts map")
	}
}

func TestHeterogeneousScheduler_RegisterDeviceCapability(t *testing.T) {
	scheduler := NewHeterogeneousScheduler()
	cap := DeviceCapability{
		DeviceID:          DeviceID{ID: "0", Library: "CUDA"},
		ComputeType:       "fp16",
		MemoryBandwidth:   500.0,
		ComputeThroughput: 10000.0,
		LatencyProfile:    "low-latency",
	}

	scheduler.RegisterDeviceCapability(cap)

	retrieved, ok := scheduler.GetDeviceCapability(DeviceID{ID: "0", Library: "CUDA"})
	if !ok {
		t.Error("expected to find registered device capability")
	}
	if retrieved.ComputeThroughput != 10000.0 {
		t.Errorf("expected 10000.0, got %f", retrieved.ComputeThroughput)
	}
}

func TestHeterogeneousScheduler_SetLayerCost(t *testing.T) {
	scheduler := NewHeterogeneousScheduler()
	cost := LayerCost{
		ComputeCost: 2.5,
		MemoryCost:  100 * 1024 * 1024,
	}

	scheduler.SetLayerCost(5, cost)
}

func TestSimpleModel_NumLayers(t *testing.T) {
	model := NewSimpleModel(32, make([]uint64, 32))
	if model.NumLayers() != 32 {
		t.Errorf("expected 32 layers, got %d", model.NumLayers())
	}
}

func TestSimpleModel_LayerSize(t *testing.T) {
	sizes := make([]uint64, 32)
	sizes[5] = 100 * 1024 * 1024
	model := NewSimpleModel(32, sizes)

	size := model.LayerSize(5)
	if size != 100*1024*1024 {
		t.Errorf("expected 100MB, got %d", size)
	}

	size = model.LayerSize(100)
	if size != 0 {
		t.Error("expected 0 for out-of-bounds layer")
	}
}
