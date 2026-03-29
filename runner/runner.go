package runner

import (
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/runner/airllmrunner"
	"github.com/ollama/ollama/runner/llamarunner"
	"github.com/ollama/ollama/runner/ollamarunner"
	"github.com/ollama/ollama/x/imagegen"
)

// airLLMModelAndReason reports whether the AirLLM runner should handle modelPath and why.
// reason is empty when ok is false.
func airLLMModelAndReason(modelPath string) (ok bool, reason string) {
	if modelPath == "" {
		return false, ""
	}

	// Explicit opt-out: always use GGML/llama.cpp (ROCm/Vulkan GPU). Without this, multi-part
	// GGUF and other heuristics still selected AirLLM even when OLLAMA_USE_AIRLLM was unset;
	// setting OLLAMA_USE_AIRLLM=0 now disables all AirLLM routing (see docs/RUNTIME_DISPATCH.md).
	if v := os.Getenv("OLLAMA_USE_AIRLLM"); v == "0" || strings.EqualFold(v, "false") || v == "no" {
		return false, ""
	}

	if os.Getenv("OLLAMA_MULTI_GGUF") == "1" {
		return true, "OLLAMA_MULTI_GGUF=1"
	}

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return false, ""
	}

	safetensorsFile := filepath.Join(modelPath, "model.safetensors.index.json")
	if _, err := os.Stat(safetensorsFile); err == nil {
		return true, "model.safetensors.index.json"
	}

	safetensorsFiles, _ := filepath.Glob(filepath.Join(modelPath, "*.safetensors"))
	if len(safetensorsFiles) > 0 {
		return true, "safetensors_shards"
	}

	configFile := filepath.Join(modelPath, "config.json")
	if data, err := os.ReadFile(configFile); err == nil {
		content := strings.ToLower(string(data))
		if strings.Contains(content, "safetensors") ||
			strings.Contains(content, "torch_dtype") ||
			strings.Contains(content, "transformers") {
			return true, "config.json_hf_heuristic"
		}
	}

	ggufFiles, _ := filepath.Glob(filepath.Join(modelPath, "*-00001-of-*.gguf"))
	if len(ggufFiles) > 0 {
		return true, "multipart_gguf"
	}

	envFlag := os.Getenv("OLLAMA_USE_AIRLLM")
	if envFlag == "1" || strings.ToLower(envFlag) == "true" {
		return true, "OLLAMA_USE_AIRLLM"
	}

	return false, ""
}

func isAirLLMModel(modelPath string) bool {
	ok, _ := airLLMModelAndReason(modelPath)
	return ok
}

func getModelPath(args []string) string {
	for i, arg := range args {
		if arg == "--model" && i+1 < len(args) {
			return args[i+1]
		}
		if strings.HasPrefix(arg, "--model=") {
			return strings.TrimPrefix(arg, "--model=")
		}
	}
	return ""
}

func Execute(args []string) error {
	if args[0] == "runner" {
		args = args[1:]
	}

	if len(args) > 0 {
		switch args[0] {
		case "--ollama-engine":
			slog.Info("runner dispatch", "engine", "ollama", "model", getModelPath(args))
			return ollamarunner.Execute(args[1:])
		case "--imagegen-engine":
			slog.Info("runner dispatch", "engine", "imagegen", "model", getModelPath(args))
			return imagegen.Execute(args[1:])
		case "--airllm-engine":
			modelPath := getModelPath(args)
			slog.Info("runner dispatch", "engine", "airllm", "model", modelPath, "reason", "explicit_flag")
			// Pass args unchanged; the airllmrunner parses --model and --port from it.
			return airllmrunner.Execute(args)
		}
	}

	modelPath := getModelPath(args)
	if useAir, why := airLLMModelAndReason(modelPath); useAir {
		slog.Info("runner dispatch", "engine", "airllm", "model", modelPath, "reason", why)
		return airllmrunner.Execute(args)
	}
	slog.Info("runner dispatch", "engine", "llama", "model", modelPath)
	return llamarunner.Execute(args)
}
