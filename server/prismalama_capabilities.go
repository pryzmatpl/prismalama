package server

import (
	"net/http"
	"os"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
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

	resp.Environment.OLLAMA_USE_AIRLLM = os.Getenv("OLLAMA_USE_AIRLLM")

	resp.Enterprise.CapabilitiesPath = "/api/prismalama/capabilities"
	resp.Enterprise.DispatchDocs = "docs/RUNTIME_DISPATCH.md"
	resp.Enterprise.Note = "GGUF default path does not replicate AirLLM streaming inside GGML; use AirLLM path for HF layouts when opted in, or see prismallama.cpp for GGUF engine evolution"

	return resp
}
