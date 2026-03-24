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

// defaultNumCtxFromVRAM maps total detected VRAM to a default num_ctx, capped by system RAM.
// Policy is OLLAMA_MEMORY_POLICY: performance (default) vs balanced (see envconfig.MemoryPolicy).
// systemRAM is the total physical system memory in bytes (from discover.GetSystemInfo).
// A large num_ctx pre-allocates a huge KV cache at load time; without a RAM ceiling a
// 256k-context load on a 64 GB machine will OOM before the first token is generated.
func defaultNumCtxFromVRAM(totalVRAM uint64, gpus []ml.DeviceInfo, systemRAM uint64) int {
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
		n = 8192
	}
	// Cap by system RAM: a 256k KV cache (f16, 80-layer model) ≈ 40 GB; 64k ≈ 10 GB;
	// 32k ≈ 5 GB. Without enough headroom the runner will OOM during load.
	if systemRAM > 0 {
		ramCap := 4096
		switch {
		case systemRAM >= 128*format.GibiByte:
			ramCap = 262144
		case systemRAM >= 96*format.GibiByte:
			ramCap = 65536
		case systemRAM >= 48*format.GibiByte:
			ramCap = 32768
		case systemRAM >= 32*format.GibiByte:
			ramCap = 8192
		}
		if n > ramCap {
			slog.Info("capping default num_ctx by system RAM",
				"default_num_ctx_before", n, "default_num_ctx", ramCap,
				"system_ram", format.HumanBytes2(systemRAM))
			n = ramCap
		}
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
