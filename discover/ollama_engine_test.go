package discover

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"testing"
	"time"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

func TestOllamaEngineRunnerArgs(t *testing.T) {
	t.Parallel()
	got := ollamaEngineRunnerArgs("/usr/bin/ollama", 12345)
	want := []string{"/usr/bin/ollama", "runner", "--ollama-engine", "--port", "12345"}
	if len(got) != len(want) {
		t.Fatalf("args %v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("args[%d] = %q want %q", i, got[i], want[i])
		}
	}
}

func TestLlamaServerDiscoverFallsBackToOllamaEngine(t *testing.T) {
	origFind := findLlamaServer
	origEngine := ollamaEngineDiscover
	t.Cleanup(func() {
		findLlamaServer = origFind
		ollamaEngineDiscover = origEngine
	})

	findLlamaServer = func() (string, error) {
		return "", errors.New("llama-server binary not found")
	}
	called := false
	want := []ml.DeviceInfo{{
		DeviceID:    ml.DeviceID{ID: "0000:0b:00.0", Library: "ROCm"},
		Description: "AMD Radeon RX 7900 XTX",
		TotalMemory: 24 << 30,
	}}
	ollamaEngineDiscover = func(ctx context.Context, libDirs []string, extraEnvs map[string]string) ([]ml.DeviceInfo, *llm.StatusWriter, error) {
		called = true
		if extraEnvs["GGML_CUDA_INIT"] != "1" {
			t.Errorf("extraEnvs GGML_CUDA_INIT = %q", extraEnvs["GGML_CUDA_INIT"])
		}
		return want, llm.NewStatusWriter(io.Discard), nil
	}

	got, _, err := llamaServerDiscoverDevices(t.Context(), []string{"/usr/lib/ollama/rocm"}, map[string]string{"GGML_CUDA_INIT": "1"})
	if err != nil {
		t.Fatal(err)
	}
	if !called {
		t.Fatal("expected ollama-engine fallback when llama-server is missing")
	}
	if len(got) != 1 || got[0].Library != "ROCm" || got[0].ID != "0000:0b:00.0" {
		t.Fatalf("devices = %+v", got)
	}
}

type infoRunner struct {
	port   int
	exited bool
}

func (r infoRunner) GetPort() int    { return r.port }
func (r infoRunner) HasExited() bool { return r.exited }

func TestGetDevicesFromRunnerReadsOllamaEngineInfo(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := ln.Addr().(*net.TCPAddr).Port

	want := []ml.DeviceInfo{{
		DeviceID:    ml.DeviceID{ID: "0000:0b:00.0", Library: "ROCm"},
		Name:        "ROCm0",
		Description: "AMD Radeon RX 7900 XTX",
		TotalMemory: 25753026560,
		PCIID:       "0000:0b:00.0",
	}}
	mux := http.NewServeMux()
	mux.HandleFunc("/info", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(want)
	})
	srv := &http.Server{Handler: mux}
	go func() { _ = srv.Serve(ln) }()
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		_ = srv.Shutdown(ctx)
	})

	ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
	defer cancel()
	got, err := ml.GetDevicesFromRunner(ctx, infoRunner{port: port})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Library != "ROCm" || got[0].TotalMemory != want[0].TotalMemory {
		t.Fatalf("got %+v", got)
	}
}
