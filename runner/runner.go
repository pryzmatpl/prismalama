package runner

import (
	"log/slog"
	"strings"

	"github.com/ollama/ollama/runner/airllmrunner"
	"github.com/ollama/ollama/runner/llamarunner"
	"github.com/ollama/ollama/runner/ollamarunner"
	"github.com/ollama/ollama/x/imagegen"
)

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
	if kind, why := DecideEngine(modelPath); kind == EngineAirLLM {
		slog.Info("runner dispatch", "engine", "airllm", "model", modelPath, "reason", why)
		return airllmrunner.Execute(args)
	}
	slog.Info("runner dispatch", "engine", "llama", "model", modelPath)
	return llamarunner.Execute(args)
}
