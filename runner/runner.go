package runner

import (
	"log/slog"
	"strings"

	"github.com/ollama/ollama/runner/airllmrunner"
	"github.com/ollama/ollama/runner/llamarunner"
	"github.com/ollama/ollama/runner/ollamarunner"
	"github.com/ollama/ollama/x/imagegen"
	"github.com/ollama/ollama/x/mlxrunner"
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
		case "--mlx-engine":
			return mlxrunner.Execute(args[1:])
		}
	}

	modelPath := getModelPath(args)
	d := DecideEngineDetailed(modelPath)
	if d.Kind == EngineAirLLM {
		slog.Info("runner dispatch",
			"engine", "airllm",
			"model", modelPath,
			"reason", d.Selected.String(),
			"reason_id", int(d.Selected),
			"reason_trace", reasonTraceString(d.Reasons),
		)
		return airllmrunner.Execute(args)
	}
	slog.Info("runner dispatch",
		"engine", "llama",
		"model", modelPath,
		"reason", d.Selected.String(),
		"reason_id", int(d.Selected),
		"reason_trace", reasonTraceString(d.Reasons),
	)
	return llamarunner.Execute(args)
}

// reasonTraceString joins the reason trace into a single string for slog.
// Uses Reason.String() so values are stable identifiers — same as
// POST /api/prismalama/dispatch will return.
func reasonTraceString(rs []Reason) string {
	if len(rs) == 0 {
		return ""
	}
	out := make([]string, len(rs))
	for i, r := range rs {
		out[i] = r.String()
	}
	return strings.Join(out, ",")
}
