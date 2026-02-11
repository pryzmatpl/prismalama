package runner

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/runner/airllmrunner"
	"github.com/ollama/ollama/runner/llamarunner"
	"github.com/ollama/ollama/runner/ollamarunner"
	"github.com/ollama/ollama/x/imagegen"
)

func isAirLLMModel(modelPath string) bool {
	if modelPath == "" {
		return false
	}

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return false
	}

	safetensorsFile := filepath.Join(modelPath, "model.safetensors.index.json")
	if _, err := os.Stat(safetensorsFile); err == nil {
		return true
	}

	safetensorsFiles, _ := filepath.Glob(filepath.Join(modelPath, "*.safetensors"))
	if len(safetensorsFiles) > 0 {
		return true
	}

	configFile := filepath.Join(modelPath, "config.json")
	if data, err := os.ReadFile(configFile); err == nil {
		content := strings.ToLower(string(data))
		if strings.Contains(content, "safetensors") ||
			strings.Contains(content, "torch_dtype") ||
			strings.Contains(content, "transformers") {
			return true
		}
	}

	envFlag := os.Getenv("OLLAMA_USE_AIRLLM")
	if envFlag == "1" || strings.ToLower(envFlag) == "true" {
		return true
	}

	return false
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
			return ollamarunner.Execute(args[1:])
		case "--imagegen-engine":
			return imagegen.Execute(args[1:])
		case "--airllm-engine":
			return airllmrunner.Execute(args[1:])
		}
	}

	modelPath := getModelPath(args)
	if isAirLLMModel(modelPath) {
		return airllmrunner.Execute(args)
	}

	return llamarunner.Execute(args)
}
