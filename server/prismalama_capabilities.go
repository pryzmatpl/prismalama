package server

import (
	"net/http"
	"os"

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

	resp.Enterprise.CapabilitiesPath = "/api/prismalama/capabilities"
	resp.Enterprise.DispatchDocs = "docs/RUNTIME_DISPATCH.md"
	resp.Enterprise.Note = "Layer streaming (OLLAMA_LAYER_STREAMING=1) brings AirLLM-like semantics to GGUF via native prismallama compute; see docs/PRISMALAMA_PRINCIPLE.md"

	return resp
}
