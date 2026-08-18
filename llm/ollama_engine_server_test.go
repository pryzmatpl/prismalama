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
	"github.com/ollama/ollama/format"
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
			GPULayers: ml.GPULayersList{{DeviceID: gpu.DeviceID, Layers: []int{0, 1}}},
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

func TestOllamaEngineLoadShrinksAfterOOM(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ln.Close() })

	var commits atomic.Int32
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		st := ServerStatusLaunched
		if commits.Load() >= 2 {
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
		n := commits.Add(1)
		if n == 1 {
			_ = json.NewEncoder(w).Encode(LoadResponse{
				Success: false,
				Error:   "insufficient memory - required allocations: GPU ROCm0 Weights too large",
				Memory: ml.BackendMemory{GPUs: []ml.DeviceMemory{{
					DeviceID: ml.DeviceID{ID: "0", Library: "ROCm"},
					Weights:  []uint64{897024768, 897024768, 540352512},
				}}},
			})
			return
		}
		if req.GPULayers.Sum() >= 3 {
			http.Error(w, "retry must shrink GPU layers", http.StatusInternalServerError)
			return
		}
		_ = json.NewEncoder(w).Encode(LoadResponse{
			Success: true,
			Memory:  ml.BackendMemory{GPUs: []ml.DeviceMemory{{DeviceID: ml.DeviceID{ID: "0", Library: "ROCm"}, Weights: []uint64{1}}}},
		})
	})

	srv := &http.Server{Handler: mux}
	go func() { _ = srv.Serve(ln) }()
	t.Cleanup(func() { _ = srv.Close() })

	gpu := ml.DeviceInfo{
		DeviceID:    ml.DeviceID{ID: "0", Library: "ROCm"},
		FreeMemory:  24 * format.GibiByte,
		TotalMemory: 24 * format.GibiByte,
	}
	s := &ollamaEngineServer{
		port:    ln.Addr().(*net.TCPAddr).Port,
		client:  http.DefaultClient,
		options: api.Options{Runner: api.Runner{NumGPU: -1, NumCtx: 256}},
		gpus:    []ml.DeviceInfo{gpu},
		loadRequest: LoadRequest{
			GPULayers: ml.GPULayersList{{DeviceID: gpu.DeviceID, Layers: []int{0, 1, 2}}},
		},
	}

	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	ids, err := s.Load(ctx, ml.SystemInfo{}, []ml.DeviceInfo{gpu}, false)
	if err != nil {
		t.Fatalf("Load after shrink: %v", err)
	}
	if commits.Load() < 2 {
		t.Fatalf("expected a retry after OOM, commits=%d", commits.Load())
	}
	if s.loadRequest.GPULayers.Sum() >= 3 {
		t.Fatalf("expected fewer than 3 GPU layers after shrink, got %v", s.loadRequest.GPULayers)
	}
	if len(ids) != 1 {
		t.Fatalf("gpu ids = %+v", ids)
	}
}

func TestOllamaEngineTokenizeDetokenize(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ln.Close() })

	mux := http.NewServeMux()
	mux.HandleFunc("POST /tokenize", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Content string `json:"content"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if req.Content != "Say hi" {
			http.Error(w, "unexpected content", http.StatusBadRequest)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string][]int{"tokens": {151644, 8948, 198}})
	})
	mux.HandleFunc("POST /detokenize", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Tokens []int `json:"tokens"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if !slices.Equal(req.Tokens, []int{151644, 8948, 198}) {
			http.Error(w, "unexpected tokens", http.StatusBadRequest)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]string{"content": "Say hi"})
	})

	srv := &http.Server{Handler: mux}
	go func() { _ = srv.Serve(ln) }()
	t.Cleanup(func() { _ = srv.Close() })

	s := &ollamaEngineServer{
		port:   ln.Addr().(*net.TCPAddr).Port,
		client: http.DefaultClient,
	}
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	ids, err := s.Tokenize(ctx, "Say hi")
	if err != nil {
		t.Fatalf("Tokenize: %v", err)
	}
	if !slices.Equal(ids, []int{151644, 8948, 198}) {
		t.Fatalf("tokens = %v", ids)
	}
	got, err := s.Detokenize(ctx, ids)
	if err != nil {
		t.Fatalf("Detokenize: %v", err)
	}
	if got != "Say hi" {
		t.Fatalf("detokenize = %q", got)
	}
}
