package server

import (
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/ml"
)

var (
	rocmGPU   = []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "ROCm"}}}
	vulkanGPU = []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "Vulkan"}}}
)

func TestDefaultNumCtxFromVRAM(t *testing.T) {
	t.Setenv("OLLAMA_MEMORY_POLICY", "")
	if g := defaultNumCtxFromVRAM(24*format.GibiByte, rocmGPU); g != 32768 {
		t.Fatalf("unset policy uses performance 24GiB tier: got %d want 32768", g)
	}
	if g := defaultNumCtxFromVRAM(24*format.GibiByte, vulkanGPU); g != 8192 {
		t.Fatalf("vulkan-only 24GiB performance: got %d want 8192", g)
	}

	t.Setenv("OLLAMA_MEMORY_POLICY", "balanced")
	if g := defaultNumCtxFromVRAM(24*format.GibiByte, vulkanGPU); g != 8192 {
		t.Fatalf("balanced 24GiB tier: got %d want 8192", g)
	}
	if g := defaultNumCtxFromVRAM(48*format.GibiByte, vulkanGPU); g != 65536 {
		t.Fatalf("balanced 48GiB tier: got %d want 65536", g)
	}
	if g := defaultNumCtxFromVRAM(8*format.GibiByte, vulkanGPU); g != 4096 {
		t.Fatalf("balanced low VRAM: got %d want 4096", g)
	}

	t.Setenv("OLLAMA_MEMORY_POLICY", "performance")
	if g := defaultNumCtxFromVRAM(24*format.GibiByte, rocmGPU); g != 32768 {
		t.Fatalf("performance 24GiB tier ROCm: got %d want 32768", g)
	}
	if g := defaultNumCtxFromVRAM(48*format.GibiByte, vulkanGPU); g != 262144 {
		t.Fatalf("performance 48GiB tier: got %d want 262144", g)
	}
}

func TestApplyAdaptiveNumCtxForParallel(t *testing.T) {
	t.Setenv("OLLAMA_MEMORY_POLICY", "balanced")
	t.Setenv("OLLAMA_NUM_PARALLEL", "4")

	opts := api.Options{Runner: api.Runner{NumCtx: 32768}}
	applyAdaptiveNumCtxForParallel(&opts)
	if opts.NumCtx != 8192 {
		t.Fatalf("expected num_ctx 8192 (32768/4 slots), got %d", opts.NumCtx)
	}

	t.Setenv("OLLAMA_MEMORY_POLICY", "performance")
	opts = api.Options{Runner: api.Runner{NumCtx: 32768}}
	applyAdaptiveNumCtxForParallel(&opts)
	if opts.NumCtx != 32768 {
		t.Fatalf("performance should not clamp, got %d", opts.NumCtx)
	}
}
