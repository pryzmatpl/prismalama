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

// Reason is the typed cause of an EngineDecision. String() returns the
// machine-stable identifier used in logs and the capabilities endpoint;
// values are part of the Prismalama public contract — do not rename without
// updating docs/RUNTIME_DISPATCH.md and POST /api/prismalama/dispatch.
type Reason int

const (
	// ReasonUnknown is the zero value; indicates no decision was recorded.
	ReasonUnknown Reason = iota
	// ReasonExplicitOptOut — OLLAMA_USE_AIRLLM ∈ {0, "false", "no"} forces GGML.
	ReasonExplicitOptOut
	// ReasonExplicitOptIn — OLLAMA_USE_AIRLLM ∈ {1, "true"} forces AirLLM.
	ReasonExplicitOptIn
	// ReasonMultiGGUF — OLLAMA_MULTI_GGUF=1 forces AirLLM.
	ReasonMultiGGUF
	// ReasonSafetensorsIndex — model.safetensors.index.json present.
	ReasonSafetensorsIndex
	// ReasonSafetensorsShards — *.safetensors present.
	ReasonSafetensorsShards
	// ReasonConfigHF — config.json mentions safetensors / torch_dtype / transformers.
	ReasonConfigHF
	// ReasonMultipartGGUF — *-00001-of-*.gguf present.
	ReasonMultipartGGUF
	// ReasonEmptyPath — modelPath is empty (defensive default).
	ReasonEmptyPath
	// ReasonPathMissing — modelPath does not exist (defensive default).
	ReasonPathMissing
	// ReasonDefaultGGML — no rule fired; GGML is the shipping default.
	ReasonDefaultGGML
)

// String returns the stable identifier for the reason (logs, JSON, capabilities).
func (r Reason) String() string {
	switch r {
	case ReasonExplicitOptOut:
		return "OLLAMA_USE_AIRLLM=0"
	case ReasonExplicitOptIn:
		return "OLLAMA_USE_AIRLLM=1"
	case ReasonMultiGGUF:
		return "OLLAMA_MULTI_GGUF=1"
	case ReasonSafetensorsIndex:
		return "model.safetensors.index.json"
	case ReasonSafetensorsShards:
		return "safetensors_shards"
	case ReasonConfigHF:
		return "config.json_hf_heuristic"
	case ReasonMultipartGGUF:
		return "multipart_gguf"
	case ReasonEmptyPath:
		return "empty_path"
	case ReasonPathMissing:
		return "path_missing"
	case ReasonDefaultGGML:
		return "default_ggml"
	default:
		return "unknown"
	}
}

// EngineDecision is the structured record of a dispatch decision. Reasons
// holds every rule that fired (in order); Selected is the *winning* reason
// (the last rule that decided the engine).
//
// Kept as a separate struct so the capabilities endpoint and dry-run
// dispatch endpoint can surface the trace without changing the public
// DecideEngine(path) (EngineKind, string) signature.
type EngineDecision struct {
	Kind     EngineKind
	Selected Reason
	Reasons  []Reason // all rules evaluated, in order
}

// DecideEngine returns which engine should load modelPath and a non-empty reason when
// EngineAirLLM is selected. Reason is diagnostic only (logs, /api/prismalama/capabilities).
//
// Signature is stable across the Phase 0 work; for the full decision trace use
// DecideEngineDetailed. Behavior is unchanged from the pre-Phase-0 implementation.
func DecideEngine(modelPath string) (EngineKind, string) {
	d := DecideEngineDetailed(modelPath)
	return d.Kind, d.Selected.String()
}

// DecideEngineDetailed returns the full decision trace. Use this when you need
// to surface the rule sequence (e.g. POST /api/prismalama/dispatch,
// capabilities.last_decision, structured logs).
func DecideEngineDetailed(modelPath string) EngineDecision {
	d := EngineDecision{Kind: EngineGGML, Selected: ReasonUnknown}

	record := func(eng EngineKind, r Reason) EngineDecision {
		d.Reasons = append(d.Reasons, r)
		d.Kind = eng
		d.Selected = r
		return d
	}

	if modelPath == "" {
		return record(EngineGGML, ReasonEmptyPath)
	}

	if v := os.Getenv("OLLAMA_USE_AIRLLM"); v == "0" || strings.EqualFold(v, "false") || v == "no" {
		return record(EngineGGML, ReasonExplicitOptOut)
	}

	if os.Getenv("OLLAMA_MULTI_GGUF") == "1" {
		return record(EngineAirLLM, ReasonMultiGGUF)
	}

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return record(EngineGGML, ReasonPathMissing)
	}

	safetensorsFile := filepath.Join(modelPath, "model.safetensors.index.json")
	if _, err := os.Stat(safetensorsFile); err == nil {
		return record(EngineAirLLM, ReasonSafetensorsIndex)
	}

	safetensorsFiles, _ := filepath.Glob(filepath.Join(modelPath, "*.safetensors"))
	if len(safetensorsFiles) > 0 {
		return record(EngineAirLLM, ReasonSafetensorsShards)
	}

	configFile := filepath.Join(modelPath, "config.json")
	if data, err := os.ReadFile(configFile); err == nil {
		content := strings.ToLower(string(data))
		if strings.Contains(content, "safetensors") ||
			strings.Contains(content, "torch_dtype") ||
			strings.Contains(content, "transformers") {
			return record(EngineAirLLM, ReasonConfigHF)
		}
	}

	ggufFiles, _ := filepath.Glob(filepath.Join(modelPath, "*-00001-of-*.gguf"))
	if len(ggufFiles) > 0 {
		return record(EngineAirLLM, ReasonMultipartGGUF)
	}

	envFlag := os.Getenv("OLLAMA_USE_AIRLLM")
	if envFlag == "1" || strings.ToLower(envFlag) == "true" {
		return record(EngineAirLLM, ReasonExplicitOptIn)
	}

	return record(EngineGGML, ReasonDefaultGGML)
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