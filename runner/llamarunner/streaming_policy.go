package llamarunner

import (
	"log/slog"

	"github.com/ollama/ollama/ml/streaming"
)

// useMmapWithLayerStreaming honors OLLAMA_LAYER_STREAMING on the llama.cpp
// runner. LoRA still disables mmap (same constraint as the load handler).
// Eval-callback keep/evict needs ml.Backend and runs in ollamarunner; this
// path at least sequentializes weight I/O via mmap.
func useMmapWithLayerStreaming(requested bool, loraCount int, layerStreaming bool) bool {
	if loraCount > 0 {
		return false
	}
	if layerStreaming {
		return true
	}
	return requested
}

func logLlamaRunnerStreaming(modelPath string, budgetBytes uint64) {
	lm, err := streaming.MapFromFile(modelPath)
	if err != nil {
		slog.Warn("llamarunner: layer streaming map failed", "error", err)
		return
	}
	fits := lm.FitsInBudget(budgetBytes)
	slog.Info("llamarunner: OLLAMA_LAYER_STREAMING mmap policy",
		"blocks", lm.BlockCount,
		"total_weight_bytes", lm.TotalWeightBytes,
		"budget_bytes", budgetBytes,
		"layers_fitting_budget", fits,
		"keep_resident", fits >= len(lm.Layers),
	)
}
