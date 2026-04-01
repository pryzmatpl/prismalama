//go:build integration

package integration

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/runner"
)

// TestShipEngineDispatchOptOut locks the ship bar: OLLAMA_USE_AIRLLM=0 must route multipart GGUF to GGML.
func TestShipEngineDispatchOptOut(t *testing.T) {
	multi := filepath.Join(t.TempDir(), "multipart")
	if err := os.MkdirAll(multi, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(multi, "weights-00001-of-00004.gguf"), []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("OLLAMA_USE_AIRLLM", "0")
	k, r := runner.DecideEngine(multi)
	if k != runner.EngineGGML || r != "" {
		t.Fatalf("multipart gguf with opt-out: want EngineGGML, got kind=%v reason=%q", k, r)
	}
}

// TestShipEngineDispatchMultipartAirLLM ensures multipart GGUF without opt-out selects AirLLM.
func TestShipEngineDispatchMultipartAirLLM(t *testing.T) {
	multi := filepath.Join(t.TempDir(), "multipart")
	if err := os.MkdirAll(multi, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(multi, "weights-00001-of-00004.gguf"), []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("OLLAMA_USE_AIRLLM", "")
	k, r := runner.DecideEngine(multi)
	if k != runner.EngineAirLLM || r != "multipart_gguf" {
		t.Fatalf("multipart gguf: want AirLLM, got kind=%v reason=%q", k, r)
	}
}

func TestShipEngineKindString(t *testing.T) {
	if runner.EngineGGML.String() != "ggml" {
		t.Fatalf("EngineGGML.String: %q", runner.EngineGGML.String())
	}
	if runner.EngineAirLLM.String() != "airllm" {
		t.Fatalf("EngineAirLLM.String: %q", runner.EngineAirLLM.String())
	}
}
