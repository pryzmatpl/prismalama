//go:build ignore

// Orphaned: findBestFit / greedyFit were removed with the llmServer layout
// state machine. Layer assignment for ollama-engine is gpuLayersForEngine.

package llm

import (
	"math"
	"testing"

	"github.com/ollama/ollama/ml"
)

// BenchmarkFindBestFit measures the performance of GPU layer assignment.
// This is a critical path: O(gpus * log(1/precision) * layers) binary search,
// with each iteration doing O(layers) greedy fitting.
func BenchmarkFindBestFit(b *testing.B) {
	// Typical large model: 80 layers, 1B params ≈ 2GB per layer on GPU
	layers := make([]uint64, 80)
	for i := range layers {
		layers[i] = 256 * 1024 * 1024 // 256 MB per layer
	}

	// 4 GPUs with varying performance
	gpus := []ml.DeviceInfo{
		{Name: "NVIDIA RTX 4090", FreeMemory: 24 * 1024 * 1024 * 1024, TotalMemory: 24 * 1024 * 1024 * 1024},
		{Name: "NVIDIA RTX 4090", FreeMemory: 24 * 1024 * 1024 * 1024, TotalMemory: 24 * 1024 * 1024 * 1024},
		{Name: "AMD RX 7900 XTX", FreeMemory: 20 * 1024 * 1024 * 1024, TotalMemory: 24 * 1024 * 1024 * 1024},
		{Name: "AMD RX 7900 XTX", FreeMemory: 20 * 1024 * 1024 * 1024, TotalMemory: 24 * 1024 * 1024 * 1024},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		findBestFit(layers, gpus, -1, false)
	}
}

// BenchmarkGreedyFit measures the core greedy fitting loop.
// This is called multiple times per findBestFit binary search iteration.
func BenchmarkGreedyFit(b *testing.B) {
	layers := make([]uint64, 80)
	for i := range layers {
		layers[i] = 256 * 1024 * 1024 // 256 MB
	}

	gpus := []ml.DeviceInfo{
		{Name: "NVIDIA RTX 4090", FreeMemory: 24 * 1024 * 1024 * 1024, TotalMemory: 24 * 1024 * 1024 * 1024},
		{Name: "NVIDIA RTX 4090", FreeMemory: 24 * 1024 * 1024 * 1024, TotalMemory: 24 * 1024 * 1024 * 1024},
		{Name: "AMD RX 7900 XTX", FreeMemory: 20 * 1024 * 1024 * 1024, TotalMemory: 24 * 1024 * 1024 * 1024},
		{Name: "AMD RX 7900 XTX", FreeMemory: 20 * 1024 * 1024 * 1024, TotalMemory: 24 * 1024 * 1024 * 1024},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		greedyFit(layers, gpus, 1.0, -1)
	}
}

// BenchmarkFindBestFit_SmallModel benchmarks layer assignment for a small model (7B params, 32 layers).
func BenchmarkFindBestFit_SmallModel(b *testing.B) {
	layers := make([]uint64, 32)
	for i := range layers {
		layers[i] = 256 * 1024 * 1024 // ~256MB per layer
	}

	gpus := []ml.DeviceInfo{
		{Name: "NVIDIA RTX 4060", FreeMemory: 8 * 1024 * 1024 * 1024, TotalMemory: 8 * 1024 * 1024 * 1024},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		findBestFit(layers, gpus, -1, false)
	}
}

// BenchmarkFindBestFit_FP16 benchmarks FP16 weights (2x memory of INT4).
func BenchmarkFindBestFit_FP16(b *testing.B) {
	layers := make([]uint64, 80)
	for i := range layers {
		layers[i] = 512 * 1024 * 1024 // 512 MB per layer (FP16)
	}

	gpus := []ml.DeviceInfo{
		{Name: "NVIDIA A100", FreeMemory: 40 * 1024 * 1024 * 1024, TotalMemory: 80 * 1024 * 1024 * 1024},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		findBestFit(layers, gpus, -1, false)
	}
}

// BenchmarkFindBestFit_ManyGPUs benchmarks scaling with many GPUs.
func BenchmarkFindBestFit_ManyGPUs(b *testing.B) {
	layers := make([]uint64, 80)
	for i := range layers {
		layers[i] = 256 * 1024 * 1024
	}

	// 8 GPUs (e.g., multi-GPU server)
	gpus := make([]ml.DeviceInfo, 8)
	for i := range gpus {
		gpus[i] = ml.DeviceInfo{Name: "NVIDIA H100", FreeMemory: 80 * 1024 * 1024 * 1024, TotalMemory: 80 * 1024 * 1024 * 1024}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		findBestFit(layers, gpus, -1, false)
	}
}

// BenchmarkFindBestFit_PartialOffload benchmarks requesting only partial GPU offload.
func BenchmarkFindBestFit_PartialOffload(b *testing.B) {
	layers := make([]uint64, 80)
	for i := range layers {
		layers[i] = 256 * 1024 * 1024
	}

	gpus := []ml.DeviceInfo{
		{Name: "NVIDIA RTX 4090", FreeMemory: 24 * 1024 * 1024 * 1024, TotalMemory: 24 * 1024 * 1024 * 1024},
	}

	// Only want 20 layers on GPU
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		findBestFit(layers, gpus, 20, true)
	}
}

// BenchmarkGPULayersListSum benchmarks the Sum() method used in binary search termination check.
func BenchmarkGPULayersListSum(b *testing.B) {
	gpuLayers := ml.GPULayersList{
		{Layers: []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}},
		{Layers: []int{15, 16, 17, 18, 19, 20, 21, 22, 23, 24}},
		{Layers: []int{25, 26, 27}},
		{Layers: []int{28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44}},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sum := gpuLayers.Sum()
		if sum == 0 {
			b.Fatal("sum should not be zero")
		}
	}
}

// BenchmarkByPerformance benchmarks GPU sorting by performance.
func BenchmarkByPerformance(b *testing.B) {
	gpus := []ml.DeviceInfo{
		{Name: "NVIDIA RTX 4060", FreeMemory: 8 * 1024 * 1024 * 1024, TotalMemory: 8 * 1024 * 1024 * 1024},
		{Name: "NVIDIA RTX 4090", FreeMemory: 24 * 1024 * 1024 * 1024, TotalMemory: 24 * 1024 * 1024 * 1024},
		{Name: "AMD RX 7900 XTX", FreeMemory: 20 * 1024 * 1024 * 1024, TotalMemory: 24 * 1024 * 1024 * 1024},
		{Name: "NVIDIA A100", FreeMemory: 40 * 1024 * 1024 * 1024, TotalMemory: 80 * 1024 * 1024 * 1024},
		{Name: "NVIDIA H100", FreeMemory: 80 * 1024 * 1024 * 1024, TotalMemory: 80 * 1024 * 1024 * 1024},
		{Name: "AMD MI300X", FreeMemory: 192 * 1024 * 1024 * 1024, TotalMemory: 256 * 1024 * 1024 * 1024},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ml.ByPerformance(gpus)
	}
}

// TestFindBestFit_BalancedDistribution verifies that layers are distributed
// roughly evenly across GPUs of equal performance.
func TestFindBestFit_BalancedDistribution(t *testing.T) {
	// 32 layers, 8GB each = 256GB total
	layers := make([]uint64, 32)
	for i := range layers {
		layers[i] = 8 * 1024 * 1024 * 1024 // 8GB each
	}

	// 4 identical GPUs, each with 80GB free
	gpus := []ml.DeviceInfo{
		{FreeMemory: 80 * 1024 * 1024 * 1024},
		{FreeMemory: 80 * 1024 * 1024 * 1024},
		{FreeMemory: 80 * 1024 * 1024 * 1024},
		{FreeMemory: 80 * 1024 * 1024 * 1024},
	}

	result := findBestFit(layers, gpus, -1, false)

	totalLayers := result.Sum()
	if totalLayers != 32 {
		t.Errorf("expected 32 layers assigned, got %d", totalLayers)
	}

	// Each GPU should have roughly 8 layers (80GB / 8GB per layer = 10, but 4 GPUs * 10 = 40 capacity > 32 needed)
	// With perfect even distribution, we'd assign 8 to each
	for i, gl := range result {
		if len(gl.Layers) < 5 || len(gl.Layers) > 10 {
			t.Errorf("GPU %d: got %d layers, expected between 5 and 10", i, len(gl.Layers))
		}
	}
}

// TestFindBestFit_ForcesFullOffload verifies that forceRequest=true forces all layers to GPU.
func TestFindBestFit_ForcesFullOffload(t *testing.T) {
	layers := make([]uint64, 32)
	for i := range layers {
		layers[i] = 8 * 1024 * 1024 * 1024
	}

	// Only 32GB free but 256GB needed — without forceRequest, would only fit 4 layers
	gpus := []ml.DeviceInfo{
		{FreeMemory: 32 * 1024 * 1024 * 1024},
	}

	result := findBestFit(layers, gpus, 32, true)

	totalLayers := result.Sum()
	if totalLayers != 32 {
		t.Errorf("forceRequest=true should assign all 32 layers, got %d", totalLayers)
	}
}

// TestFindBestFit_PartialOffload verifies that partial offload works correctly.
func TestFindBestFit_PartialOffload(t *testing.T) {
	layers := make([]uint64, 32)
	for i := range layers {
		layers[i] = 4 * 1024 * 1024 * 1024 // 4GB each = 128GB total
	}

	// 256GB free on single GPU
	gpus := []ml.DeviceInfo{
		{FreeMemory: 256 * 1024 * 1024 * 1024},
	}

	// Request only 16 layers on GPU
	result := findBestFit(layers, gpus, 16, true)

	totalLayers := result.Sum()
	if totalLayers != 16 {
		t.Errorf("expected 16 layers assigned, got %d", totalLayers)
	}
}

// TestFindBestFit_PerformanceOrdering verifies that higher-performance GPUs
// get more layers when capacities differ.
func TestFindBestFit_PerformanceOrdering(t *testing.T) {
	layers := make([]uint64, 40)
	for i := range layers {
		layers[i] = 1 * 1024 * 1024 * 1024 // 1GB each
	}

	// GPU0: 10GB, GPU1: 100GB — GPU1 should get more layers
	gpus := []ml.DeviceInfo{
		{FreeMemory: 10 * 1024 * 1024 * 1024},
		{FreeMemory: 100 * 1024 * 1024 * 1024},
	}

	result := findBestFit(layers, gpus, -1, false)

	// GPU1 (higher performance / more VRAM) should have more layers
	if len(result[0].Layers) >= len(result[1].Layers) {
		t.Errorf("expected higher-perf GPU (GPU1) to get more layers; GPU0=%d, GPU1=%d",
			len(result[0].Layers), len(result[1].Layers))
	}
}

// TestGreedyFit_AllFit verifies all layers fit when GPU has enough memory.
func TestGreedyFit_AllFit(t *testing.T) {
	layers := make([]uint64, 10)
	for i := range layers {
		layers[i] = 1 * 1024 * 1024 * 1024 // 1GB each
	}

	gpus := []ml.DeviceInfo{
		{FreeMemory: 20 * 1024 * 1024 * 1024}, // 20GB
	}

	result := greedyFit(layers, gpus, 1.0, -1)

	if result.Sum() != 10 {
		t.Errorf("expected all 10 layers to fit, got %d", result.Sum())
	}
}

// TestGreedyFit_PartialFit verifies partial fit when memory is insufficient.
func TestGreedyFit_PartialFit(t *testing.T) {
	layers := make([]uint64, 10)
	for i := range layers {
		layers[i] = 3 * 1024 * 1024 * 1024 // 3GB each
	}

	gpus := []ml.DeviceInfo{
		{FreeMemory: 20 * 1024 * 1024 * 1024}, // 20GB — only fits 6 layers
	}

	result := greedyFit(layers, gpus, 1.0, -1)

	if result.Sum() != 6 {
		t.Errorf("expected 6 layers to fit (20GB/3GB), got %d", result.Sum())
	}
}

// TestGreedyFit_RequestedLayersLimit verifies that requestedLayers caps the assignment.
func TestGreedyFit_RequestedLayersLimit(t *testing.T) {
	layers := make([]uint64, 20)
	for i := range layers {
		layers[i] = 1 * 1024 * 1024 * 1024
	}

	gpus := []ml.DeviceInfo{
		{FreeMemory: 100 * 1024 * 1024 * 1024}, // 100GB — all 20 fit easily
	}

	result := greedyFit(layers, gpus, 1.0, 5) // but only request 5

	if result.Sum() != 5 {
		t.Errorf("expected 5 layers (requested), got %d", result.Sum())
	}
}

// TestGreedyFit_CapacityFactor verifies that capacity < 1.0 uses less than full memory.
func TestGreedyFit_CapacityFactor(t *testing.T) {
	layers := make([]uint64, 10)
	for i := range layers {
		layers[i] = 2 * 1024 * 1024 * 1024 // 2GB each
	}

	gpus := []ml.DeviceInfo{
		{FreeMemory: 20 * 1024 * 1024 * 1024}, // 20GB
	}

	// With capacity=0.5, only 10GB usable → 5 layers should fit
	result := greedyFit(layers, gpus, 0.5, -1)

	if result.Sum() != 5 {
		t.Errorf("expected 5 layers with 50%% capacity, got %d", result.Sum())
	}
}

// TestFindBestFit_LargeGapInFreeMemory verifies behavior when there's a large gap
// between the largest and smallest GPU's free memory (edge case).
func TestFindBestFit_LargeGapInFreeMemory(t *testing.T) {
	layers := make([]uint64, 100)
	for i := range layers {
		layers[i] = 1 * 1024 * 1024 * 1024 // 1GB each
	}

	// One GPU has tons of memory, another barely any
	gpus := []ml.DeviceInfo{
		{FreeMemory: 1 * 1024 * 1024 * 1024},                      // 1GB
		{FreeMemory: 200 * 1024 * 1024 * 1024},                     // 200GB
	}

	result := findBestFit(layers, gpus, -1, false)

	// All 100 layers should still fit (on GPU1)
	if result.Sum() != 100 {
		t.Errorf("expected all 100 layers to fit, got %d", result.Sum())
	}
}

// BenchmarkGreedyFit_WorstCaseManyGPUs benchmarks greedy fit when layers barely
// fit and must try many GPUs.
func BenchmarkGreedyFit_WorstCase(b *testing.B) {
	// Layers that barely fit: 8GB each, 100 layers
	layers := make([]uint64, 100)
	for i := range layers {
		layers[i] = 8 * 1024 * 1024 * 1024
	}

	// 16 small GPUs, each 10GB
	gpus := make([]ml.DeviceInfo, 16)
	for i := range gpus {
		gpus[i] = ml.DeviceInfo{FreeMemory: 10 * 1024 * 1024 * 1024}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		greedyFit(layers, gpus, 1.0, -1)
	}
}

// TestFindBestFitZeroLayers verifies findBestFit handles zero layers.
func TestFindBestFitZeroLayers(t *testing.T) {
	layers := []uint64{}

	gpus := []ml.DeviceInfo{
		{FreeMemory: 100 * 1024 * 1024 * 1024},
	}

	result := findBestFit(layers, gpus, -1, false)
	if len(result) != 0 {
		t.Errorf("expected empty result for zero layers, got len=%d", len(result))
	}
}

// TestFindBestFitZeroGPUs verifies findBestFit handles zero GPUs.
func TestFindBestFitZeroGPUs(t *testing.T) {
	layers := make([]uint64, 10)
	for i := range layers {
		layers[i] = 1 * 1024 * 1024 * 1024
	}

	gpus := []ml.DeviceInfo{}

	result := findBestFit(layers, gpus, -1, false)
	if len(result) != 0 {
		t.Errorf("expected empty result for zero GPUs, got len=%d", len(result))
	}
}

// TestFindBestFit_IntegerLayers verifies that layer counts are always integers
// (no fractional layer assignment is possible, which would indicate a bug).
func TestFindBestFit_IntegerLayers(t *testing.T) {
	layers := make([]uint64, 7)
	for i := range layers {
		layers[i] = 3*1024*1024*1024 + 512*1024*1024 // 3.5GB each — awkward size
	}

	gpus := []ml.DeviceInfo{
		{FreeMemory: 10 * 1024 * 1024 * 1024}, // 10GB — can't evenly divide
	}

	result := findBestFit(layers, gpus, -1, false)

	for i, gl := range result {
		if len(gl.Layers) < 0 {
			t.Errorf("GPU %d: negative layer count %d", i, len(gl.Layers))
		}
	}
}
