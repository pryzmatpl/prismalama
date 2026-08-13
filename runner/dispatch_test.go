package runner

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDecideEngine_EmptyPath(t *testing.T) {
	k, r := DecideEngine("")
	if k != EngineGGML || r != "" {
		t.Fatalf("empty path: got EngineKind=%v reason=%q", k, r)
	}
}

func TestDecideEngine_OptOutDisablesAirLLM(t *testing.T) {
	tmp := t.TempDir()
	if err := os.WriteFile(filepath.Join(tmp, "m-00001-of-00002.gguf"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("OLLAMA_USE_AIRLLM", "0")
	k, r := DecideEngine(tmp)
	if k != EngineGGML || r != "" {
		t.Fatalf("OLLAMA_USE_AIRLLM=0: want GGML, got kind=%v reason=%q", k, r)
	}
}

func TestDecideEngine_MultiPartGGUF(t *testing.T) {
	tmp := t.TempDir()
	if err := os.WriteFile(filepath.Join(tmp, "m-00001-of-00002.gguf"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("OLLAMA_USE_AIRLLM", "")
	k, r := DecideEngine(tmp)
	if k != EngineAirLLM || r != "multipart_gguf" {
		t.Fatalf("multipart gguf: got kind=%v reason=%q", k, r)
	}
}

func TestDecideEngine_SafetensorsIndex(t *testing.T) {
	tmp := t.TempDir()
	if err := os.WriteFile(filepath.Join(tmp, "model.safetensors.index.json"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	k, r := DecideEngine(tmp)
	if k != EngineAirLLM || r != "model.safetensors.index.json" {
		t.Fatalf("safetensors index: got kind=%v reason=%q", k, r)
	}
}

func TestDecideEngine_ConfigHFHeuristic(t *testing.T) {
	tmp := t.TempDir()
	if err := os.WriteFile(filepath.Join(tmp, "config.json"), []byte(`{"torch_dtype":"float16"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	k, r := DecideEngine(tmp)
	if k != EngineAirLLM || r != "config.json_hf_heuristic" {
		t.Fatalf("config heuristic: got kind=%v reason=%q", k, r)
	}
}

func TestDecideEngine_SingleGGUFUsesGGML(t *testing.T) {
	tmp := t.TempDir()
	if err := os.WriteFile(filepath.Join(tmp, "model.gguf"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("OLLAMA_USE_AIRLLM", "")
	k, r := DecideEngine(tmp)
	if k != EngineGGML || r != "" {
		t.Fatalf("single gguf: want GGML, got kind=%v reason=%q", k, r)
	}
}

func TestDecideEngine_ForceAirLLMEnv(t *testing.T) {
	tmp := t.TempDir()
	if err := os.WriteFile(filepath.Join(tmp, "model.gguf"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("OLLAMA_USE_AIRLLM", "true")
	k, r := DecideEngine(tmp)
	if k != EngineAirLLM || r != "OLLAMA_USE_AIRLLM" {
		t.Fatalf("OLLAMA_USE_AIRLLM=true: got kind=%v reason=%q", k, r)
	}
}

func TestDecideEngine_MissingPath(t *testing.T) {
	t.Setenv("OLLAMA_USE_AIRLLM", "")
	k, r := DecideEngine("/nonexistent/path/that/does/not/exist")
	if k != EngineGGML || r != "" {
		t.Fatalf("missing path: got kind=%v reason=%q", k, r)
	}
}

func TestAirLLMModelAndReasonCompat(t *testing.T) {
	tmp := t.TempDir()
	if err := os.WriteFile(filepath.Join(tmp, "m-00001-of-00002.gguf"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("OLLAMA_USE_AIRLLM", "")
	ok, why := airLLMModelAndReason(tmp)
	if !ok || why != "multipart_gguf" {
		t.Fatalf("compat: ok=%v why=%q", ok, why)
	}
	if !isAirLLMModel(tmp) {
		t.Fatal("isAirLLMModel should be true")
	}
}

// Phase 0 / JAISIU-2157: typed Reason coverage.

func TestReasonStrings_Stable(t *testing.T) {
	// Stable identifiers — part of the public contract (capabilities, dispatch
	// endpoint). Back-compat: pre-Phase-0 contract mapped OLLAMA_USE_AIRLLM=0/1
	// to the bare identifier "OLLAMA_USE_AIRLLM" (see runner/dispatch_test.go's
	// TestDecideEngine_ForceAirLLMEnv / _OptOutDisablesAirLLM).
	cases := map[Reason]string{
		ReasonUnknown:          "unknown",
		ReasonExplicitOptOut:   "OLLAMA_USE_AIRLLM",
		ReasonExplicitOptIn:    "OLLAMA_USE_AIRLLM",
		ReasonMultiGGUF:        "OLLAMA_MULTI_GGUF",
		ReasonSafetensorsIndex: "model.safetensors.index.json",
		ReasonSafetensorsShards:"safetensors_shards",
		ReasonConfigHF:         "config.json_hf_heuristic",
		ReasonMultipartGGUF:    "multipart_gguf",
		ReasonEmptyPath:        "empty_path",
		ReasonPathMissing:      "path_missing",
		ReasonDefaultGGML:      "default_ggml",
	}
	for r, want := range cases {
		if got := r.String(); got != want {
			t.Errorf("Reason(%d).String()=%q want %q", r, got, want)
		}
	}
	// Confirm both opt-out and opt-in collapse to the same identifier
	// (the bare "OLLAMA_USE_AIRLLM") — matches the legacy DecideEngine contract.
	if ReasonExplicitOptOut.String() != ReasonExplicitOptIn.String() {
		t.Fatal("opt-out and opt-in must collapse to the same back-compat identifier")
	}
}

func TestDecideEngineDetailed_AllReasons(t *testing.T) {
	// Each heuristic must produce the expected typed Reason via the detailed API.
	t.Run("explicit_opt_out", func(t *testing.T) {
		tmp := t.TempDir()
		os.WriteFile(filepath.Join(tmp, "m-00001-of-00002.gguf"), []byte("x"), 0o644)
		t.Setenv("OLLAMA_USE_AIRLLM", "0")
		d := DecideEngineDetailed(tmp)
		if d.Kind != EngineGGML || d.Selected != ReasonExplicitOptOut {
			t.Fatalf("got kind=%v selected=%v", d.Kind, d.Selected)
		}
	})
	t.Run("explicit_opt_out_wins_over_hf_layout", func(t *testing.T) {
		tmp := t.TempDir()
		os.WriteFile(filepath.Join(tmp, "config.json"), []byte(`{"torch_dtype":"float16"}`), 0o644)
		t.Setenv("OLLAMA_USE_AIRLLM", "false")
		d := DecideEngineDetailed(tmp)
		if d.Kind != EngineGGML || d.Selected != ReasonExplicitOptOut {
			t.Fatalf("opt-out must beat HF heuristic: kind=%v selected=%v", d.Kind, d.Selected)
		}
	})
	t.Run("explicit_opt_in_with_empty_path", func(t *testing.T) {
		t.Setenv("OLLAMA_USE_AIRLLM", "1")
		d := DecideEngineDetailed("")
		if d.Kind != EngineGGML || d.Selected != ReasonEmptyPath {
			t.Fatalf("empty path with opt-in: kind=%v selected=%v", d.Kind, d.Selected)
		}
	})
	t.Run("multi_gguf_env", func(t *testing.T) {
		t.Setenv("OLLAMA_USE_AIRLLM", "")
		t.Setenv("OLLAMA_MULTI_GGUF", "1")
		d := DecideEngineDetailed("/nonexistent")
		if d.Kind != EngineAirLLM || d.Selected != ReasonMultiGGUF {
			t.Fatalf("multi gguf env: kind=%v selected=%v", d.Kind, d.Selected)
		}
	})
	t.Run("missing_path", func(t *testing.T) {
		t.Setenv("OLLAMA_USE_AIRLLM", "")
		t.Setenv("OLLAMA_MULTI_GGUF", "")
		d := DecideEngineDetailed("/definitely/not/here")
		if d.Kind != EngineGGML || d.Selected != ReasonPathMissing {
			t.Fatalf("missing path: kind=%v selected=%v", d.Kind, d.Selected)
		}
	})
	t.Run("default_ggml", func(t *testing.T) {
		tmp := t.TempDir()
		os.WriteFile(filepath.Join(tmp, "model.gguf"), []byte("x"), 0o644)
		t.Setenv("OLLAMA_USE_AIRLLM", "")
		t.Setenv("OLLAMA_MULTI_GGUF", "")
		d := DecideEngineDetailed(tmp)
		if d.Kind != EngineGGML || d.Selected != ReasonDefaultGGML {
			t.Fatalf("default ggml: kind=%v selected=%v", d.Kind, d.Selected)
		}
		if len(d.Reasons) != 1 {
			t.Fatalf("default path: expected single reason, got %v", d.Reasons)
		}
	})
}

func TestDecideEngineDetailed_Precedence(t *testing.T) {
	// Order: opt-out → multi_gguf → path_missing → safetensors_index → safetensors_shards
	// → config.json → multipart_gguf → opt-in → default.
	// safetensors_index must beat safetensors_shards.
	tmp := t.TempDir()
	os.WriteFile(filepath.Join(tmp, "model.safetensors.index.json"), []byte("{}"), 0o644)
	os.WriteFile(filepath.Join(tmp, "weights.safetensors"), []byte("x"), 0o644)
	t.Setenv("OLLAMA_USE_AIRLLM", "")
	t.Setenv("OLLAMA_MULTI_GGUF", "")
	d := DecideEngineDetailed(tmp)
	if d.Selected != ReasonSafetensorsIndex {
		t.Fatalf("precedence: safetensors_index must win, got %v", d.Selected)
	}
}

func TestDecideEngineDetailed_ReasonsTrace(t *testing.T) {
	// The detailed API records the rules evaluated (single rule for default paths).
	tmp := t.TempDir()
	os.WriteFile(filepath.Join(tmp, "model-00001-of-00003.gguf"), []byte("x"), 0o644)
	t.Setenv("OLLAMA_USE_AIRLLM", "")
	d := DecideEngineDetailed(tmp)
	if d.Selected != ReasonMultipartGGUF {
		t.Fatalf("multipart: got selected=%v", d.Selected)
	}
	if len(d.Reasons) != 1 || d.Reasons[0] != ReasonMultipartGGUF {
		t.Fatalf("multipart: reasons trace = %v", d.Reasons)
	}
}

func TestDecideEngine_BackCompatString(t *testing.T) {
	// The legacy DecideEngine(path) (EngineKind, string) signature must keep
	// returning the same string identifiers it always has — back-compat for
	// existing log scrapers and capability consumers.
	tmp := t.TempDir()
	os.WriteFile(filepath.Join(tmp, "m-00001-of-00002.gguf"), []byte("x"), 0o644)
	t.Setenv("OLLAMA_USE_AIRLLM", "")
	k, r := DecideEngine(tmp)
	if k != EngineAirLLM || r != "multipart_gguf" {
		t.Fatalf("back-compat: got kind=%v reason=%q", k, r)
	}
}
