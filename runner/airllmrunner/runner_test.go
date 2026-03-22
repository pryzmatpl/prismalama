package airllmrunner

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

func TestAirllmPythonPathDevCheckout(t *testing.T) {
	root := t.TempDir()
	runnerDir := filepath.Join(root, "runner", "airllmrunner")
	if err := os.MkdirAll(filepath.Join(root, "src", "airllm", "air_llm", "airllm"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(runnerDir, 0o755); err != nil {
		t.Fatal(err)
	}
	py := filepath.Join(runnerDir, "airllm_runner.py")
	if err := os.WriteFile(py, nil, 0o644); err != nil {
		t.Fatal(err)
	}

	got := airllmPythonPath(py)
	if !strings.Contains(got, filepath.Join("src", "airllm", "air_llm")) {
		t.Fatalf("expected dev PYTHONPATH segment, got %q", got)
	}
}

func TestAirllmPythonPathPackagedFallback(t *testing.T) {
	got := airllmPythonPath("/usr/share/ollama/airllm_runner.py")
	if !strings.Contains(got, "/usr/share/ollama/airllm") {
		t.Fatalf("expected packaged paths, got %q", got)
	}
}

func TestNewServerPythonUsesSeparatePort(t *testing.T) {
	s := NewServer("/tmp/model", 54321)
	if s.pythonPort != 54322 {
		t.Fatalf("pythonPort: want 54322, got %d", s.pythonPort)
	}
	if s.port != 54321 {
		t.Fatalf("port: want 54321, got %d", s.port)
	}
}

func TestPythonLoadBodyUsesStringOperation(t *testing.T) {
	req := llm.LoadRequest{
		Operation:      llm.LoadOperationCommit,
		BatchSize:      512,
		KvSize:         4096,
		FlashAttention: ml.FlashAttentionAuto,
	}
	body, err := pythonLoadBody(req, "/models/x")
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatal(err)
	}
	if m["operation"] != "commit" {
		t.Fatalf("operation: want commit string, got %v", m["operation"])
	}
	if m["flash_attention"] != "auto" {
		t.Fatalf("flash_attention: want auto, got %v", m["flash_attention"])
	}
}

func TestMainServerLoadJSONDecodesIntoLLMLoadRequest(t *testing.T) {
	req := llm.LoadRequest{
		Operation: llm.LoadOperationCommit,
		BatchSize: 512,
	}
	data, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	var got llm.LoadRequest
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("decode same as main server sends: %v", err)
	}
	if got.Operation != llm.LoadOperationCommit {
		t.Fatalf("Operation: got %v want commit", got.Operation)
	}
}
