package server

import (
	"log/slog"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
)

// defaultNumCtxFromVRAM maps total detected VRAM to a default num_ctx.
// Policy is OLLAMA_MEMORY_POLICY: performance (default) vs balanced (see envconfig.MemoryPolicy).
func defaultNumCtxFromVRAM(totalVRAM uint64) int {
	switch envconfig.MemoryPolicy() {
	case "performance":
		switch {
		case totalVRAM >= 47*format.GibiByte:
			return 262144
		case totalVRAM >= 23*format.GibiByte:
			return 32768
		default:
			return 4096
		}
	default: // balanced
		// Favor running larger-than-VRAM models and many parallel agent slots without
		// exploding KV memory; users tune up via Modelfile / request options.
		switch {
		case totalVRAM >= 47*format.GibiByte:
			return 65536
		case totalVRAM >= 23*format.GibiByte:
			return 8192
		default:
			return 4096
		}
	}
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
