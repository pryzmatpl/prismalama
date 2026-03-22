package airllmrunner

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
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
