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
