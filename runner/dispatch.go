package runner

import (
	"os"
	"path/filepath"
	"strings"
)

// EngineKind selects which runner subprocess handles a model directory.
// It does not implement AirLLM-style layer streaming inside GGML; it only
// records which higher-level engine (GGML vs Python AirLLM) is selected.
type EngineKind int

const (
	// EngineGGML is llama.cpp / GGML (mmap, partial GPU offload; not PyTorch layer streaming).
	EngineGGML EngineKind = iota
	// EngineAirLLM is the Python + PyTorch AirLLM runner (HF / NVMe-oriented streaming).
	EngineAirLLM
)

func (k EngineKind) String() string {
	switch k {
	case EngineAirLLM:
		return "airllm"
	default:
		return "ggml"
	}
}

// DecideEngine returns which engine should load modelPath and a non-empty reason when
// EngineAirLLM is selected. Reason is diagnostic only (logs, /api/prismalama/capabilities).
func DecideEngine(modelPath string) (EngineKind, string) {
	if modelPath == "" {
		return EngineGGML, ""
	}

	if v := os.Getenv("OLLAMA_USE_AIRLLM"); v == "0" || strings.EqualFold(v, "false") || v == "no" {
		return EngineGGML, ""
	}

	if os.Getenv("OLLAMA_MULTI_GGUF") == "1" {
		return EngineAirLLM, "OLLAMA_MULTI_GGUF=1"
	}

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return EngineGGML, ""
	}

	safetensorsFile := filepath.Join(modelPath, "model.safetensors.index.json")
	if _, err := os.Stat(safetensorsFile); err == nil {
		return EngineAirLLM, "model.safetensors.index.json"
	}

	safetensorsFiles, _ := filepath.Glob(filepath.Join(modelPath, "*.safetensors"))
	if len(safetensorsFiles) > 0 {
		return EngineAirLLM, "safetensors_shards"
	}

	configFile := filepath.Join(modelPath, "config.json")
	if data, err := os.ReadFile(configFile); err == nil {
		content := strings.ToLower(string(data))
		if strings.Contains(content, "safetensors") ||
			strings.Contains(content, "torch_dtype") ||
			strings.Contains(content, "transformers") {
			return EngineAirLLM, "config.json_hf_heuristic"
		}
	}

	ggufFiles, _ := filepath.Glob(filepath.Join(modelPath, "*-00001-of-*.gguf"))
	if len(ggufFiles) > 0 {
		return EngineAirLLM, "multipart_gguf"
	}

	envFlag := os.Getenv("OLLAMA_USE_AIRLLM")
	if envFlag == "1" || strings.ToLower(envFlag) == "true" {
		return EngineAirLLM, "OLLAMA_USE_AIRLLM"
	}

	return EngineGGML, ""
}

// airLLMModelAndReason reports whether the AirLLM runner should handle modelPath and why.
// reason is empty when ok is false.
func airLLMModelAndReason(modelPath string) (ok bool, reason string) {
	k, r := DecideEngine(modelPath)
	return k == EngineAirLLM, r
}

func isAirLLMModel(modelPath string) bool {
	k, _ := DecideEngine(modelPath)
	return k == EngineAirLLM
}
