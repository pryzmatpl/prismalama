// loadrequest.go holds the IPC types used by the ollamarunner and llamarunner
// subprocess protocols. These are prismalama-specific and are not part of the
// upstream llama-server binary interface.
package llm

import (
	"fmt"
	"log/slog"

	"github.com/ollama/ollama/ml"
)

// ImageData is a backwards-compat alias for MediaData used by the llamarunner
// and ollamarunner internal path. New code should use MediaData directly.
type ImageData = MediaData

// EmbeddingRequest is sent by the runner HTTP server to request an embedding.
type EmbeddingRequest struct {
	Content string `json:"content"`
}

// EmbeddingResponse carries the embedding vector from the runner subprocess.
type EmbeddingResponse struct {
	Embedding       []float32 `json:"embedding"`
	PromptEvalCount int       `json:"prompt_eval_count"`
}

// LoadOperation enumerates the stages of a model load handshake.
type LoadOperation int

// The order of these constants is significant: operations are executed in
// ascending order (fit → alloc → commit).
const (
	LoadOperationFit    LoadOperation = iota // Return memory requirements without allocating
	LoadOperationAlloc                       // Reserve memory but do not copy weights
	LoadOperationCommit                      // Copy weights – no further changes after this
	LoadOperationClose                       // Release model memory
)

func (o LoadOperation) String() string {
	switch o {
	case LoadOperationFit:
		return "fit"
	case LoadOperationAlloc:
		return "alloc"
	case LoadOperationCommit:
		return "commit"
	case LoadOperationClose:
		return "close"
	default:
		return "unknown"
	}
}

// LoadRequest is the JSON payload sent from the server to a runner subprocess
// to initiate or advance a model load.
type LoadRequest struct {
	Operation LoadOperation

	LoraPath       []string
	Parallel       int
	BatchSize      int
	FlashAttention ml.FlashAttentionType
	KvSize         int
	KvCacheType    string
	NumThreads     int
	GPULayers      ml.GPULayersList
	MultiUserCache bool

	// Legacy fields – not used with the Ollama engine
	ProjectorPath string
	MainGPU       int
	UseMmap       bool
}

// LoadResponse is the JSON payload returned by a runner subprocess after
// processing a LoadRequest.
type LoadResponse struct {
	Success bool             `json:"success"`
	Memory  ml.BackendMemory `json:"memory,omitempty"`
	Error   string           `json:"error,omitempty"`
}

// backendMemoryFromRunner is true when the runner JSON included a non-empty
// memory breakdown (weights/cache/graph). The llama.cpp runner typically
// returns success without a "memory" field; an all-zero BackendMemory must
// not replace the GGUF estimate in s.mem.
func backendMemoryFromRunner(m ml.BackendMemory) bool {
	if m.InputWeights != 0 {
		return true
	}
	if m.CPU.Size() != 0 {
		return true
	}
	for i := range m.GPUs {
		if m.GPUs[i].Size() != 0 {
			return true
		}
	}
	return false
}

// logGGMLGPUOffload emits a structured log line with GPU-offload statistics.
func logGGMLGPUOffload(totalLayers uint64, gpuLayers ml.GPULayersList, useMmap bool) {
	off := gpuLayers.Sum()
	if totalLayers == 0 {
		return
	}
	pct := 100.0 * float64(off) / float64(totalLayers)
	slog.Info("ggml GPU layer offload",
		"gpu_layers", off,
		"total_layers", totalLayers,
		"offload_percent", fmt.Sprintf("%.0f%%", pct),
		"use_mmap", useMmap,
	)
}
