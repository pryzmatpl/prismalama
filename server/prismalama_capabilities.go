package server

import (
	"net/http"
	"os"
	"runtime"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/version"
)

// PrismalamaCapabilitiesHandler documents GGML vs AirLLM inference semantics for enterprise operators.
func PrismalamaCapabilitiesHandler(c *gin.Context) {
	c.JSON(http.StatusOK, buildPrismalamaCapabilities())
}

func buildPrismalamaCapabilities() api.PrismalamaCapabilitiesResponse {
	var resp api.PrismalamaCapabilitiesResponse
	resp.Version = version.Version

	resp.GGUF.Engine = "llama.cpp / GGML (native runner)"
	resp.GGUF.WeightSemantics = "mmap, partial GPU offload, KV in VRAM/RAM; not PyTorch layer-by-layer NVMe streaming"

	resp.AirLLM.Engine = "Python AirLLM + PyTorch (airllm_runner)"
	resp.AirLLM.WeightSemantics = "Hugging Face–style checkpoints: layer-wise execution and NVMe-oriented streaming where AirLLM supports the architecture"
	resp.AirLLM.OptInEnv = "OLLAMA_USE_AIRLLM"

	resp.LayerStreaming.Enabled = envconfig.LayerStreaming()
	resp.LayerStreaming.BudgetBytes = envconfig.StreamingBudgetBytes()
	resp.LayerStreaming.Semantics = "GGUF layer-by-layer: load block from NVMe, compute on GPU, evict, prefetch next — AirLLM-like behavior for native GGUF"
	resp.LayerStreaming.EnableEnv = "OLLAMA_LAYER_STREAMING"

	resp.Environment.OLLAMA_USE_AIRLLM = os.Getenv("OLLAMA_USE_AIRLLM")
	resp.Environment.OLLAMA_LAYER_STREAMING = os.Getenv("OLLAMA_LAYER_STREAMING")
	resp.Environment.OLLAMA_MEMORY_POLICY = os.Getenv("OLLAMA_MEMORY_POLICY")
	resp.Environment.OLLAMA_VULKAN = os.Getenv("OLLAMA_VULKAN")
	resp.Environment.OLLAMA_MMAP_ALLOW_LOW_RAM = os.Getenv("OLLAMA_MMAP_ALLOW_LOW_RAM")

	resp.OperatorHints = prismalamaOperatorHints()

	resp.Enterprise.CapabilitiesPath = "/api/prismalama/capabilities"
	resp.Enterprise.DispatchDocs = "docs/RUNTIME_DISPATCH.md"
	resp.Enterprise.Note = "Layer streaming (OLLAMA_LAYER_STREAMING=1) brings AirLLM-like semantics to GGUF via native prismallama compute; see docs/PRISMALAMA_PRINCIPLE.md"

	return resp
}

func prismalamaOperatorHints() []string {
	var h []string
	if !envconfig.LayerStreaming() {
		h = append(h, "OLLAMA_LAYER_STREAMING is off (unset/false). Set OLLAMA_LAYER_STREAMING=1 so GGUF can use the layer-streaming path when weights exceed VRAM (requires backend support; runner logs show whether streaming activated).")
	}
	if envconfig.MemoryPolicy() != "balanced" {
		h = append(h, "OLLAMA_MEMORY_POLICY defaults to performance. For models much larger than VRAM or many parallel requests, set OLLAMA_MEMORY_POLICY=balanced for conservative default context and KV budgeting (see server/memory_policy.go).")
	}
	if runtime.GOOS == "linux" && !envconfig.EnableVulkan() {
		h = append(h, "OLLAMA_VULKAN is off: Vulkan GGML backends are skipped during GPU discovery (discover/runner.go). Set OLLAMA_VULKAN=1 if you rely on Vulkan. CUDA/HIP libraries are still discovered separately when present.")
	}
	if runtime.GOOS == "linux" && !envconfig.MmapAllowLowRamLinux() {
		h = append(h, "OLLAMA_MMAP_ALLOW_LOW_RAM is off: when free RAM is below the GGUF size, Linux may disable mmap and force full resident weights. For fast NVMe + larger-than-RAM GGUF, set OLLAMA_MMAP_ALLOW_LOW_RAM=1 (see llm/server.go applyLoadMmapPolicy).")
	}
	return h
}
