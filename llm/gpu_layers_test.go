package llm

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/ml"
)

func TestFitLayersFromEnd(t *testing.T) {
	t.Parallel()
	const layer = uint64(897024768)
	const output = uint64(540352512)
	weights := make([]uint64, 41)
	for i := 0; i < 40; i++ {
		weights[i] = layer
	}
	weights[40] = output
	kv := make([]uint64, 40)
	for i := range kv {
		kv[i] = 16 << 20
	}

	budget := uint64(21 * format.GibiByte)
	got := fitLayersFromEnd(weights, kv, budget, -1)
	if len(got) == 0 {
		t.Fatal("expected a tail of layers to fit 21 GiB")
	}
	if got[0] == 0 {
		t.Fatalf("36.9 GB GGUF must not 100%%-offload onto 24 GiB, got %v", got)
	}
	if got[len(got)-1] != 40 {
		t.Fatalf("output layer should stay on GPU when it fits, got %v", got)
	}
	if len(got) < 15 || len(got) > 30 {
		t.Fatalf("unexpected offload count %d: %v", len(got), got)
	}
	for i := 1; i < len(got); i++ {
		if got[i] != got[i-1]+1 {
			t.Fatalf("offload must be contiguous from the end, got %v", got)
		}
	}

	got = fitLayersFromEnd(weights, kv, budget, 2)
	if !slices.Equal(got, []int{39, 40}) {
		t.Fatalf("NumGPU 2 should pack last two layers, got %v", got)
	}

	if fitLayersFromEnd(weights, kv, 1<<20, -1) != nil {
		t.Fatal("tiny budget should offload nothing")
	}
	if fitLayersFromEnd(weights, kv, budget, 0) != nil {
		t.Fatal("NumGPU 0 should offload nothing")
	}

	got = fitLayersFromEnd(weights, kv, 64*format.GibiByte, -1)
	if len(got) != 41 {
		t.Fatalf("64 GiB should take every layer, got %d", len(got))
	}
}

func TestGPUBudgetBytes(t *testing.T) {
	t.Parallel()
	gpu := ml.DeviceInfo{
		DeviceID:    ml.DeviceID{ID: "0000:0b:00.0", Library: "ROCm"},
		FreeMemory:  24 * format.GibiByte,
		TotalMemory: 24 * format.GibiByte,
	}
	budget := gpuBudgetBytes(gpu, 512*format.MebiByte)
	if budget == 0 || budget >= gpu.FreeMemory {
		t.Fatalf("budget %d should be less than free VRAM %d", budget, gpu.FreeMemory)
	}
	// 24 GiB - 457 MiB min - 1 GiB safety - 1 GiB graph floor
	if budget > 22*format.GibiByte {
		t.Fatalf("budget %d left too little headroom", budget)
	}
	if gpuBudgetBytes(ml.DeviceInfo{}, 0) != 0 {
		t.Fatal("zero free memory should yield zero budget")
	}
	fromTotal := gpuBudgetBytes(ml.DeviceInfo{TotalMemory: 24 * format.GibiByte}, 0)
	if fromTotal == 0 {
		t.Fatal("TotalMemory should be used when FreeMemory is unset")
	}
}

func TestGPULayersForEngine(t *testing.T) {
	t.Parallel()
	gpu := ml.DeviceInfo{DeviceID: ml.DeviceID{ID: "0000:0b:00.0", Library: "ROCm"}, FreeMemory: 24 * format.GibiByte}
	if gpuLayersForEngine(nil, -1, nil, api.Options{}, 1) != nil {
		t.Fatal("no GPU should return nil")
	}
	got := gpuLayersForEngine([]ml.DeviceInfo{gpu}, 0, nil, api.Options{}, 1)
	if got.Sum() != 0 {
		t.Fatalf("CPU-only should be empty, got %v", got)
	}
	got = gpuLayersForEngine([]ml.DeviceInfo{gpu}, -1, nil, api.Options{}, 1)
	if got.Sum() != 0 {
		t.Fatalf("auto without GGUF metadata must not 100%%-offload, got %v", got)
	}
}

func TestShrinkGPULayers(t *testing.T) {
	t.Parallel()
	gpu := ml.DeviceInfo{
		DeviceID:    ml.DeviceID{ID: "0", Library: "ROCm"},
		FreeMemory:  24 * format.GibiByte,
		TotalMemory: 24 * format.GibiByte,
	}
	cur := ml.GPULayersList{{DeviceID: gpu.DeviceID, Layers: make([]int, 41)}}
	for i := range cur[0].Layers {
		cur[0].Layers[i] = i
	}
	weights := make([]uint64, 41)
	for i := 0; i < 40; i++ {
		weights[i] = 897024768
	}
	weights[40] = 540352512
	mem := ml.BackendMemory{GPUs: []ml.DeviceMemory{{
		DeviceID: gpu.DeviceID,
		Weights:  weights,
	}}}

	got := shrinkGPULayers(cur, mem, gpu)
	if got.Sum() == 0 {
		t.Fatal("expected a reduced GPU assignment to remain")
	}
	if got.Sum() >= cur.Sum() {
		t.Fatalf("shrink must drop layers, before %d after %d", cur.Sum(), got.Sum())
	}
	layers := got[0].Layers
	if layers[len(layers)-1] != 40 {
		t.Fatalf("should keep the output tail, got %v", layers)
	}

	one := ml.GPULayersList{{DeviceID: gpu.DeviceID, Layers: []int{40}}}
	if shrinkGPULayers(one, ml.BackendMemory{}, gpu) != nil {
		t.Fatal("single layer with no measured sizes should drop to CPU")
	}
}

func TestDropLeadingGPULayer(t *testing.T) {
	t.Parallel()
	gpu := ml.DeviceID{ID: "0", Library: "ROCm"}
	got := dropLeadingGPULayer(ml.GPULayersList{{DeviceID: gpu, Layers: []int{38, 39, 40}}})
	if !slices.Equal(got[0].Layers, []int{39, 40}) {
		t.Fatalf("got %v", got)
	}
	if dropLeadingGPULayer(ml.GPULayersList{{DeviceID: gpu, Layers: []int{40}}}) != nil {
		t.Fatal("last layer should drop to CPU-only")
	}
}

func TestIsInsufficientMemory(t *testing.T) {
	t.Parallel()
	if !isInsufficientMemory("insufficient memory - required allocations: ...") {
		t.Fatal("expected match")
	}
	if isInsufficientMemory("model runner has unexpectedly stopped") {
		t.Fatal("did not expect match")
	}
}
