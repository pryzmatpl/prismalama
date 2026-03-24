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

// highRAM simulates a workstation with 128 GB RAM (no RAM cap applied).
const highRAM = 128 * format.GibiByte

func TestDefaultNumCtxFromVRAM(t *testing.T) {
	t.Setenv("OLLAMA_MEMORY_POLICY", "")
	if g := defaultNumCtxFromVRAM(24*format.GibiByte, rocmGPU, highRAM); g != 32768 {
		t.Fatalf("unset policy uses performance 24GiB tier: got %d want 32768", g)
	}
	if g := defaultNumCtxFromVRAM(24*format.GibiByte, vulkanGPU, highRAM); g != 8192 {
		t.Fatalf("vulkan-only 24GiB performance: got %d want 8192", g)
	}

	t.Setenv("OLLAMA_MEMORY_POLICY", "balanced")
	if g := defaultNumCtxFromVRAM(24*format.GibiByte, vulkanGPU, highRAM); g != 8192 {
		t.Fatalf("balanced 24GiB tier: got %d want 8192", g)
	}
	if g := defaultNumCtxFromVRAM(48*format.GibiByte, vulkanGPU, highRAM); g != 65536 {
		t.Fatalf("balanced 48GiB tier: got %d want 65536", g)
	}
	if g := defaultNumCtxFromVRAM(8*format.GibiByte, vulkanGPU, highRAM); g != 4096 {
		t.Fatalf("balanced low VRAM: got %d want 4096", g)
	}

	t.Setenv("OLLAMA_MEMORY_POLICY", "performance")
	if g := defaultNumCtxFromVRAM(24*format.GibiByte, rocmGPU, highRAM); g != 32768 {
		t.Fatalf("performance 24GiB tier ROCm: got %d want 32768", g)
	}
	if g := defaultNumCtxFromVRAM(48*format.GibiByte, vulkanGPU, highRAM); g != 262144 {
		t.Fatalf("performance 48GiB tier: got %d want 262144", g)
	}
}

func TestDefaultNumCtxRAMCap(t *testing.T) {
	t.Setenv("OLLAMA_MEMORY_POLICY", "performance")

	// 48 GiB VRAM → 262144 by tier, but 64 GB RAM → cap at 32768
	if g := defaultNumCtxFromVRAM(48*format.GibiByte, rocmGPU, 64*format.GibiByte); g != 32768 {
		t.Fatalf("performance 48GiB VRAM + 64GB RAM: got %d want 32768", g)
	}

	// 48 GiB VRAM → 262144, 96 GB RAM → cap at 65536
	if g := defaultNumCtxFromVRAM(48*format.GibiByte, rocmGPU, 96*format.GibiByte); g != 65536 {
		t.Fatalf("performance 48GiB VRAM + 96GB RAM: got %d want 65536", g)
	}

	// 48 GiB VRAM → 262144, 128 GB RAM → no cap
	if g := defaultNumCtxFromVRAM(48*format.GibiByte, rocmGPU, 128*format.GibiByte); g != 262144 {
		t.Fatalf("performance 48GiB VRAM + 128GB RAM: got %d want 262144", g)
	}

	// 24 GiB VRAM → 32768, 32 GB RAM → cap at 8192
	if g := defaultNumCtxFromVRAM(24*format.GibiByte, rocmGPU, 32*format.GibiByte); g != 8192 {
		t.Fatalf("performance 24GiB VRAM + 32GB RAM: got %d want 8192", g)
	}

	// 24 GiB VRAM → 32768, 48 GB RAM → no cap (32768 <= 32768)
	if g := defaultNumCtxFromVRAM(24*format.GibiByte, rocmGPU, 48*format.GibiByte); g != 32768 {
		t.Fatalf("performance 24GiB VRAM + 48GB RAM: got %d want 32768", g)
	}

	// 24 GiB VRAM → 32768, 16 GB RAM → cap at 4096
	if g := defaultNumCtxFromVRAM(24*format.GibiByte, rocmGPU, 16*format.GibiByte); g != 4096 {
		t.Fatalf("performance 24GiB VRAM + 16GB RAM: got %d want 4096", g)
	}

	// Zero system RAM → skip RAM cap
	if g := defaultNumCtxFromVRAM(48*format.GibiByte, rocmGPU, 0); g != 262144 {
		t.Fatalf("zero system RAM should skip cap: got %d want 262144", g)
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
