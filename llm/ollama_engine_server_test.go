package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"slices"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/ml"
	"golang.org/x/sync/semaphore"
)

func TestUseOllamaEngineRunner(t *testing.T) {
	orig := findLlamaServerForGenerate
	t.Cleanup(func() { findLlamaServerForGenerate = orig })

	findLlamaServerForGenerate = func() (string, error) {
		return "/usr/lib/ollama/llama-server", nil
	}
	if useOllamaEngineRunner() {
		t.Fatal("expected llama-server path when binary is found")
	}

	findLlamaServerForGenerate = func() (string, error) {
		return "", errors.New("llama-server binary not found")
	}
	if !useOllamaEngineRunner() {
		t.Fatal("expected ollama-engine fallback when llama-server is missing")
	}
}

func TestOllamaEngineGenerateArgs(t *testing.T) {
	t.Parallel()
	got := ollamaEngineGenerateArgs("/usr/bin/ollama", "/models/qwen.gguf", 12345)
	want := []string{"/usr/bin/ollama", "runner", "--ollama-engine", "--model", "/models/qwen.gguf", "--port", "12345"}
	if !slices.Equal(got, want) {
		t.Fatalf("args %v want %v", got, want)
	}
}

func TestGPULayersForEngine(t *testing.T) {
	t.Parallel()
	gpu := ml.DeviceInfo{DeviceID: ml.DeviceID{ID: "0000:0b:00.0", Library: "ROCm"}}
	got := gpuLayersForEngine([]ml.DeviceInfo{gpu}, -1, 3)
	if got.Sum() != 3 || got[0].ID != "0000:0b:00.0" {
		t.Fatalf("auto offload = %v", got)
	}
	got = gpuLayersForEngine([]ml.DeviceInfo{gpu}, 0, 3)
	if got.Sum() != 0 {
		t.Fatalf("CPU-only should be empty, got %v", got)
	}
	got = gpuLayersForEngine([]ml.DeviceInfo{gpu}, 2, 8)
	if got.Sum() != 2 || !slices.Equal(got[0].Layers, []int{0, 1}) {
		t.Fatalf("capped offload = %v", got)
	}
	if gpuLayersForEngine(nil, -1, 4) != nil {
		t.Fatal("no GPU should return nil")
	}
}

func TestChatMLPrompt(t *testing.T) {
	t.Parallel()
	got := chatMLPrompt([]api.Message{
		{Role: "user", Content: "hi"},
	})
	want := "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
	if got != want {
		t.Fatalf("prompt %q want %q", got, want)
	}
}

func TestApplyCompletionFormat(t *testing.T) {
	t.Parallel()
	req := CompletionRequest{Format: []byte(`"json"`)}
	if err := applyCompletionFormat(&req); err != nil {
		t.Fatal(err)
	}
	if req.Grammar == "" {
		t.Fatal("expected JSON grammar")
	}
	req = CompletionRequest{Format: []byte("X")}
	if err := applyCompletionFormat(&req); err == nil {
		t.Fatal("expected invalid format error")
	}
	req = CompletionRequest{Format: []byte(`{"type":"object"}`)}
	if err := applyCompletionFormat(&req); err != nil {
		t.Fatal(err)
	}
}

func TestStripEnvKeys(t *testing.T) {
	t.Parallel()
	env := []string{"PATH=/bin", "HIP_VISIBLE_DEVICES=1", "FOO=bar", "hip_visible_devices=2", "GPU_DEVICE_ORDINAL=3"}
	got := stripEnvKeys(env, []string{"HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"})
	want := []string{"PATH=/bin", "FOO=bar"}
	if !slices.Equal(got, want) {
		t.Fatalf("got %#v want %#v", got, want)
	}
}

func TestHipVisibleDeviceIndexForSingleROCR(t *testing.T) {
	t.Parallel()
	tests := []struct {
		goos, rocr string
		want       string
	}{
		{"linux", "0000:03:00.0", "0"},
		{"linux", "0", "0"},
		{"linux", "0,1", ""},
		{"linux", "", ""},
		{"windows", "0", ""},
		{"darwin", "0000:03:00.0", ""},
	}
	for _, tt := range tests {
		if got := hipVisibleDeviceIndexForSingleROCR(tt.goos, tt.rocr); got != tt.want {
			t.Errorf("goos=%q rocr=%q: got %q want %q", tt.goos, tt.rocr, got, tt.want)
		}
	}
}

func TestBackendMemoryFromRunner(t *testing.T) {
	t.Parallel()
	if backendMemoryFromRunner(ml.BackendMemory{}) {
		t.Fatal("empty memory should not count as runner-reported")
	}
	if !backendMemoryFromRunner(ml.BackendMemory{InputWeights: 1}) {
		t.Fatal("input weights should count")
	}
}

func TestOllamaEngineServerLoadAndCompletion(t *testing.T) {
	var committed atomic.Bool
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ln.Close() })

	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		st := ServerStatusLaunched
		if committed.Load() {
			st = ServerStatusReady
		}
		_ = json.NewEncoder(w).Encode(ServerStatusResponse{Status: st})
	})
	mux.HandleFunc("POST /load", func(w http.ResponseWriter, r *http.Request) {
		var req LoadRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if req.Operation != LoadOperationCommit {
			http.Error(w, fmt.Sprintf("unexpected op %s", req.Operation), http.StatusBadRequest)
			return
		}
		committed.Store(true)
		_ = json.NewEncoder(w).Encode(LoadResponse{
			Success: true,
			Memory:  ml.BackendMemory{InputWeights: 42, GPUs: []ml.DeviceMemory{{DeviceID: ml.DeviceID{ID: "0", Library: "ROCm"}, Weights: []uint64{100}}}},
		})
	})
	mux.HandleFunc("POST /completion", func(w http.ResponseWriter, r *http.Request) {
		var req CompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if req.Prompt == "" {
			http.Error(w, "missing prompt", http.StatusBadRequest)
			return
		}
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "no flush", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(CompletionResponse{Content: "hello"})
		flusher.Flush()
		_ = json.NewEncoder(w).Encode(CompletionResponse{Content: " world", Done: true, DoneReason: DoneReasonStop, EvalCount: 2})
		flusher.Flush()
	})

	srv := &http.Server{Handler: mux}
	go func() { _ = srv.Serve(ln) }()
	t.Cleanup(func() { _ = srv.Close() })

	port := ln.Addr().(*net.TCPAddr).Port
	gpu := ml.DeviceInfo{DeviceID: ml.DeviceID{ID: "0", Library: "ROCm"}}
	s := &ollamaEngineServer{
		port:    port,
		client:  http.DefaultClient,
		options: api.Options{Runner: api.Runner{NumGPU: -1, NumCtx: 128}},
		sem:     semaphore.NewWeighted(1),
		gpus:    []ml.DeviceInfo{gpu},
		ggml:    nil,
		loadRequest: LoadRequest{
			GPULayers: gpuLayersForEngine([]ml.DeviceInfo{gpu}, -1, 2),
		},
	}

	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	ids, err := s.Load(ctx, ml.SystemInfo{}, []ml.DeviceInfo{gpu}, false)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(ids) != 1 || ids[0].ID != "0" {
		t.Fatalf("gpu ids = %+v", ids)
	}
	if _, vram := s.MemorySize(); vram == 0 {
		t.Fatal("expected VRAM from load response")
	}

	var got string
	var done bool
	err = s.Completion(ctx, CompletionRequest{Prompt: "hi", Options: &api.Options{}}, func(r CompletionResponse) {
		got += r.Content
		done = r.Done
	})
	if err != nil {
		t.Fatalf("Completion: %v", err)
	}
	if got != "hello world" || !done {
		t.Fatalf("completion %q done=%v", got, done)
	}

	prompt, err := s.ApplyChatTemplate(ctx, ChatRequest{Messages: []api.Message{{Role: "user", Content: "hi"}}})
	if err != nil {
		t.Fatal(err)
	}
	if prompt == "" {
		t.Fatal("expected chat prompt")
	}
}

func TestOllamaEngineServerCompletionCanceledBeforeHTTP(t *testing.T) {
	s := &ollamaEngineServer{
		sem:     semaphore.NewWeighted(1),
		options: api.Options{Runner: api.Runner{NumCtx: 32}},
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	err := s.Completion(ctx, CompletionRequest{Options: new(api.Options), Format: []byte(`"json"`)}, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v; want context.Canceled", err)
	}
}
