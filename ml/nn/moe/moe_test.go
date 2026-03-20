package moe

import (
	"testing"
)

func TestMoEConfig(t *testing.T) {
	config := MoEConfig{
		NumExperts:      8,
		TopKExperts:     2,
		RoutingStrategy: "topk",
		ExpertsOnDisk:   true,
		ExpertCacheSize: 4,
	}

	if config.NumExperts != 8 {
		t.Errorf("expected 8 experts, got %d", config.NumExperts)
	}
	if config.TopKExperts != 2 {
		t.Errorf("expected TopK 2, got %d", config.TopKExperts)
	}
	if config.RoutingStrategy != "topk" {
		t.Errorf("expected topk strategy, got %s", config.RoutingStrategy)
	}
}

func TestLRUCache_New(t *testing.T) {
	cache := NewLRUCache(10)
	if cache == nil {
		t.Fatal("expected non-nil LRUCache")
	}
	if cache.capacity != 10 {
		t.Errorf("expected capacity 10, got %d", cache.capacity)
	}
}

func TestLRUCache_Put_Get(t *testing.T) {
	cache := NewLRUCache(10)

	cache.Put(1, "value1")
	val, ok := cache.Get(1)
	if !ok {
		t.Error("expected to find value")
	}
	if val != "value1" {
		t.Errorf("expected value1, got %v", val)
	}
}

func TestLRUCache_Get_NotFound(t *testing.T) {
	cache := NewLRUCache(10)

	_, ok := cache.Get(999)
	if ok {
		t.Error("expected not to find nonexistent key")
	}
}

func TestLRUCache_Update(t *testing.T) {
	cache := NewLRUCache(10)

	cache.Put(1, "value1")
	cache.Put(1, "value2")

	val, ok := cache.Get(1)
	if !ok {
		t.Error("expected to find value")
	}
	if val != "value2" {
		t.Errorf("expected value2, got %v", val)
	}
}

func TestLRUCache_Remove(t *testing.T) {
	cache := NewLRUCache(10)

	cache.Put(1, "value1")
	cache.Remove(1)

	_, ok := cache.Get(1)
	if ok {
		t.Error("expected not to find removed value")
	}
}

func TestLRUCache_Evict(t *testing.T) {
	cache := NewLRUCache(2)

	cache.Put(1, "value1")
	cache.Put(2, "value2")
	cache.Put(3, "value3")

	_, ok := cache.Get(1)
	if ok {
		t.Error("expected oldest value to be evicted")
	}
}

func TestStreamingMoE_New(t *testing.T) {
	config := MoEConfig{
		NumExperts:      8,
		TopKExperts:     2,
		RoutingStrategy: "topk",
		ExpertsOnDisk:   true,
		ExpertCacheSize: 4,
	}

	moe := NewStreamingMoE(config, "/nvme/experts")
	if moe == nil {
		t.Fatal("expected non-nil StreamingMoE")
	}
	if len(moe.experts) != 8 {
		t.Errorf("expected 8 experts, got %d", len(moe.experts))
	}
	if moe.expertCache.capacity != 4 {
		t.Errorf("expected cache capacity 4, got %d", moe.expertCache.capacity)
	}
}

func TestStreamingMoE_LoadExpert(t *testing.T) {
	config := MoEConfig{
		NumExperts:      8,
		TopKExperts:     2,
		ExpertCacheSize: 4,
	}

	moe := NewStreamingMoE(config, "/nvme/experts")

	err := moe.LoadExpert(0)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !moe.experts[0].Loaded {
		t.Error("expected expert 0 to be loaded")
	}
}

func TestStreamingMoE_EvictExpert(t *testing.T) {
	config := MoEConfig{
		NumExperts:      8,
		TopKExperts:     2,
		ExpertCacheSize: 4,
	}

	moe := NewStreamingMoE(config, "/nvme/experts")
	moe.LoadExpert(0)

	err := moe.EvictExpert(0)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if moe.experts[0].Loaded {
		t.Error("expected expert 0 to be evicted")
	}
}

func TestStreamingMoE_GetActiveExperts(t *testing.T) {
	config := MoEConfig{
		NumExperts:      8,
		TopKExperts:     2,
		ExpertCacheSize: 4,
	}

	moe := NewStreamingMoE(config, "/nvme/experts")
	moe.LoadExpert(0)
	moe.LoadExpert(2)

	active := moe.GetActiveExperts()
	if len(active) != 2 {
		t.Errorf("expected 2 active experts, got %d", len(active))
	}
}

func TestStreamingMoE_GetAuxLoss(t *testing.T) {
	config := MoEConfig{
		NumExperts:      8,
		TopKExperts:     2,
		ExpertCacheSize: 4,
	}

	moe := NewStreamingMoE(config, "/nvme/experts")

	loss := moe.GetAuxLoss()
	if loss != 0.0 {
		t.Errorf("expected initial aux loss 0, got %f", loss)
	}
}

func TestTopKRouter_New(t *testing.T) {
	config := MoEConfig{
		NumExperts:  8,
		TopKExperts: 2,
	}

	router := NewTopKRouter(config)
	if router == nil {
		t.Fatal("expected non-nil TopKRouter")
	}
}

func TestTopKRouter_Route(t *testing.T) {
	config := MoEConfig{
		NumExperts:  8,
		TopKExperts: 2,
	}

	router := NewTopKRouter(config)
	expertIDs, auxLosses := router.Route([]int{1, 2, 3})

	if len(expertIDs) != 2 {
		t.Errorf("expected 2 expert IDs, got %d", len(expertIDs))
	}
	if len(auxLosses) != 2 {
		t.Errorf("expected 2 aux losses, got %d", len(auxLosses))
	}
}

func TestTopKRouter_UpdateLoadBalancing(t *testing.T) {
	config := MoEConfig{
		NumExperts:  8,
		TopKExperts: 2,
	}

	router := NewTopKRouter(config)
	router.UpdateLoadBalancing(0.5)

	if router.loadBalancing != 0.5 {
		t.Errorf("expected 0.5, got %f", router.loadBalancing)
	}
}

func TestExpertCache_New(t *testing.T) {
	cache := NewExpertCache("/nvme/experts", 1024*1024*1024)
	if cache == nil {
		t.Fatal("expected non-nil ExpertCache")
	}
	if cache.maxCacheSize != 1024*1024*1024 {
		t.Errorf("expected maxCacheSize 1GB, got %d", cache.maxCacheSize)
	}
}

func TestExpertCache_LoadExpert_NotFound(t *testing.T) {
	cache := NewExpertCache("/nvme/experts", 1024*1024*1024)

	data, err := cache.LoadExpert(0)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if data != nil {
		t.Error("expected nil for non-cached expert")
	}
}

func TestExpertCache_CacheExpert(t *testing.T) {
	cache := NewExpertCache("/nvme/experts", 1024*1024*1024)

	data := []byte{1, 2, 3, 4}
	err := cache.CacheExpert(0, data)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	cached, err := cache.LoadExpert(0)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if cached == nil {
		t.Error("expected cached data")
	}
}

func TestExpertCache_Clear(t *testing.T) {
	cache := NewExpertCache("/nvme/experts", 1024*1024*1024)
	cache.CacheExpert(0, []byte{1, 2, 3})
	cache.Clear()

	data, _ := cache.LoadExpert(0)
	if data != nil {
		t.Error("expected nil after clear")
	}
}

func TestFP8MoEKernels_New(t *testing.T) {
	kernels := NewFP8MoEKernels()
	if kernels == nil {
		t.Fatal("expected non-nil FP8MoEKernels")
	}
	if !kernels.IsEnabled() {
		t.Error("expected FP8 kernels to be enabled")
	}
}

func TestFP8MoEKernels_SetEnabled(t *testing.T) {
	kernels := NewFP8MoEKernels()
	kernels.SetEnabled(false)

	if kernels.IsEnabled() {
		t.Error("expected FP8 kernels to be disabled")
	}
}
