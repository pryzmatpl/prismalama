package server

import (
	"log/slog"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/ml"
)

func onlyVulkanGPUs(gpus []ml.DeviceInfo) bool {
	if len(gpus) == 0 {
		return false
	}
	for _, g := range gpus {
		if g.Library != "Vulkan" {
			return false
		}
	}
	return true
}

// defaultNumCtxFromVRAM maps total detected VRAM to a default num_ctx.
// Policy is OLLAMA_MEMORY_POLICY: performance (default) vs balanced (see envconfig.MemoryPolicy).
func defaultNumCtxFromVRAM(totalVRAM uint64, gpus []ml.DeviceInfo) int {
	var n int
	switch envconfig.MemoryPolicy() {
	case "performance":
		switch {
		case totalVRAM >= 47*format.GibiByte:
			n = 262144
		case totalVRAM >= 23*format.GibiByte:
			n = 32768
		default:
			n = 4096
		}
	default: // balanced
		// Favor running larger-than-VRAM models and many parallel agent slots without
		// exploding KV memory; users tune up via Modelfile / request options.
		switch {
		case totalVRAM >= 47*format.GibiByte:
			n = 65536
		case totalVRAM >= 23*format.GibiByte:
			n = 8192
		default:
			n = 4096
		}
	}
	// Vulkan-only (e.g. Linux+AMD+RADV): performance tier would use 32k ctx on 24 GiB class
	// GPUs — KV + MoE weights exceed VRAM during load. Match balanced-tier default.
	if onlyVulkanGPUs(gpus) && envconfig.MemoryPolicy() == "performance" &&
		totalVRAM < 32*format.GibiByte && n > 8192 {
		slog.Info("vulkan-only GPU: capping default num_ctx for VRAM headroom",
			"default_num_ctx_before", n, "default_num_ctx", 8192, "total_vram", format.HumanBytes2(totalVRAM))
		return 8192
	}
	return n
}

// kvSlotBudget is the maximum num_ctx * OLLAMA_NUM_PARALLEL before balanced policy clamps.
const kvSlotBudget = 32768

// applyAdaptiveNumCtxForParallel reduces num_ctx when balanced policy would exceed KV slot budget.
func applyAdaptiveNumCtxForParallel(opts *api.Options) {
	if envconfig.MemoryPolicy() != "balanced" {
		return
	}
	np := int(envconfig.NumParallel())
	if np < 1 {
		np = 1
	}
	if opts.NumCtx*np <= kvSlotBudget {
		return
	}
	newCtx := max(4, kvSlotBudget/np)
	if newCtx < opts.NumCtx {
		slog.Info("OLLAMA_MEMORY_POLICY=balanced: clamping num_ctx for parallel KV budget",
			"num_ctx", opts.NumCtx, "num_ctx_adapted", newCtx, "OLLAMA_NUM_PARALLEL", np, "kv_slot_budget", kvSlotBudget)
		opts.NumCtx = newCtx
	}
}
