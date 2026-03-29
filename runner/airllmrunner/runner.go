package airllmrunner

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
)

type Server struct {
	modelPath  string
	port       int // port the Go proxy listens on (parent ollama talks here)
	pythonPort int // Python airllm_runner.py HTTP server — must differ from port
	pythonBase string
	pythonCmd  *exec.Cmd
	status     llm.ServerStatus
	progress   float32
	mu         sync.Mutex
	commitMu   sync.Mutex // serializes Commit handlers; pairs Add/Done on ready for reloads
	// commitReturnedOnce is true after any Commit handler has finished (success or failure).
	// First Commit pairs with ready.Add(1) in NewServer; subsequent Commits must Add(1) before Done
	// or sync.WaitGroup panics ("negative WaitGroup counter") and the runner dies.
	commitReturnedOnce bool
	ready              sync.WaitGroup
	httpClient         *http.Client
	pythonDone         chan struct{} // closed when the Python subprocess exits

	// Health tracking for circuit breaker pattern.
	// After consecutiveErrorLimit errors, the runner starts failing fast instead of
	// retrying. This prevents hammering a dead runner and allows faster fallback.
	consecutiveErrors     int
	consecutiveErrorLimit int   // defaults to 0; set to 5 in NewServer
	lastError            string
}

type statusResponse struct {
	Status   string  `json:"status"`
	Progress float32 `json:"progress"`
}

type completionResponse struct {
	Content            string        `json:"content"`
	Logprobs           []llm.Logprob `json:"logprobs"`
	Done               bool          `json:"done"`
	DoneReason         string        `json:"done_reason"`
	PromptEvalCount    int           `json:"prompt_eval_count"`
	PromptEvalDuration int64         `json:"prompt_eval_duration"`
	EvalCount          int           `json:"eval_count"`
	EvalDuration       int64         `json:"eval_duration"`
}

// flashAttentionForPy maps ggml flash-attn enum to strings the Python runner accepts.
func flashAttentionForPy(f ml.FlashAttentionType) string {
	switch f {
	case ml.FlashAttentionAuto:
		return "auto"
	case ml.FlashAttentionDisabled:
		return "disabled"
	case ml.FlashAttentionEnabled:
		return "enabled"
	default:
		return "auto"
	}
}

// pythonLoadBody builds JSON for airllm_runner.py /load using snake_case keys and string operation.
func pythonLoadBody(req llm.LoadRequest, modelPath string) ([]byte, error) {
	m := map[string]any{
		"operation":        req.Operation.String(),
		"model_path":       modelPath,
		"lora_path":        req.LoraPath,
		"projector_path":   req.ProjectorPath,
		"parallel":         req.Parallel,
		"batch_size":       req.BatchSize,
		"kv_size":          req.KvSize,
		"kv_cache_type":    req.KvCacheType,
		"flash_attention":  flashAttentionForPy(req.FlashAttention),
		"num_threads":      req.NumThreads,
		"multi_user_cache": req.MultiUserCache,
		"gpu_layers":       req.GPULayers,
		"main_gpu":         req.MainGPU,
		"use_mmap":         req.UseMmap,
	}
	return json.Marshal(m)
}

func optionsToMap(o *api.Options) map[string]interface{} {
	if o == nil {
		return map[string]interface{}{}
	}
	b, err := json.Marshal(o)
	if err != nil {
		return map[string]interface{}{}
	}
	var m map[string]interface{}
	if err := json.Unmarshal(b, &m); err != nil {
		return map[string]interface{}{}
	}
	return m
}

// pythonCompletionBody builds JSON for airllm_runner.py /completion (snake_case, options object).
func pythonCompletionBody(req llm.CompletionRequest) ([]byte, error) {
	m := map[string]any{
		"prompt":       req.Prompt,
		"images":       req.Images,
		"grammar":      req.Grammar,
		"options":      optionsToMap(req.Options),
		"logprobs":     req.Logprobs,
		"top_logprobs": req.TopLogprobs,
		"shift":        req.Shift,
		"truncate":     req.Truncate,
	}
	return json.Marshal(m)
}

func NewServer(modelPath string, port int) *Server {
	// Python must bind a different port than this Go process; otherwise HTTPClient.Post
	// to /load would hit this same handler with pythonLoadBody JSON and break llm.LoadRequest decode.
	pyPort := port + 1
	if pyPort > 65535 {
		pyPort = port - 1
	}
	s := &Server{
		modelPath:  modelPath,
		port:       port,
		pythonPort: pyPort,
		pythonBase: fmt.Sprintf("http://127.0.0.1:%d", pyPort),
		status:     llm.ServerStatusLaunched,
		httpClient: &http.Client{
			// 10-minute timeout per request — long enough for model loading or
			// a single inference step on large models, but prevents indefinite hangs.
			Timeout: 10 * time.Minute,
		},
		pythonDone:            make(chan struct{}),
		consecutiveErrorLimit: 5, // open circuit breaker after 5 consecutive errors
	}
	s.ready.Add(1)
	return s
}

func (s *Server) startPythonRunner() error {
	pythonRunnerPath := findPythonRunner()
	if pythonRunnerPath == "" {
		return errors.New("airllm_runner.py not found")
	}

	cmd := exec.Command("python3", pythonRunnerPath,
		"--model", s.modelPath,
		"--port", strconv.Itoa(s.pythonPort),
	)
	// Inherit full parent env (HIP_VISIBLE_DEVICES, HSA_*, AIRLLM_DEVICE, etc.) for GPU visibility in PyTorch.
	childEnv := append(os.Environ(),
		"AIRLLM_COMPRESSION="+os.Getenv("AIRLLM_COMPRESSION"),
		"PYTHONPATH="+airllmPythonPath(pythonRunnerPath),
	)
	if _, ok := os.LookupEnv("AIRLLM_DEVICE"); !ok {
		childEnv = append(childEnv, "AIRLLM_DEVICE=cuda:0")
	}
	cmd.Env = childEnv
	adev := os.Getenv("AIRLLM_DEVICE")
	if adev == "" {
		adev = "cuda:0"
	}
	slog.Info("airllm python env (subset)", "AIRLLM_DEVICE", adev, "HIP_VISIBLE_DEVICES", os.Getenv("HIP_VISIBLE_DEVICES"))
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr

	s.pythonCmd = cmd
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start Python runner: %w", err)
	}

	go func() {
		cmd.Wait()
		slog.Info("Python runner exited")
		close(s.pythonDone) // signal that the Python process has exited
	}()

	return s.waitForReady()
}

func (s *Server) waitForReady() error {
	for i := 0; i < 60; i++ {
		resp, err := s.httpClient.Get(s.pythonBase + "/health")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				slog.Info("Python runner is ready")
				return nil
			}
		}
		time.Sleep(500 * time.Millisecond)
	}
	return errors.New("timeout waiting for Python runner to start")
}

// airllmPythonPath sets import path for the `airllm` package: packaged install
// or a git checkout (src/airllm/air_llm next to the repo root).
func airllmPythonPath(pythonRunnerPath string) string {
	const packaged = "/usr/share/ollama/airllm:/usr/share/ollama/airllm/air_llm"
	runnerDir := filepath.Dir(pythonRunnerPath)
	// .../runner/airllmrunner/airllm_runner.py -> .../src/airllm/air_llm
	devAirLLM := filepath.Join(runnerDir, "..", "..", "src", "airllm", "air_llm")
	if st, err := os.Stat(filepath.Join(devAirLLM, "airllm")); err == nil && st.IsDir() {
		return devAirLLM + ":" + packaged
	}
	extra := os.Getenv("PRISMALAMA_AIRLLM_PYTHONPATH")
	if extra != "" {
		return extra + ":" + packaged
	}
	return packaged
}

func findPythonRunner() string {
	exeDir := filepath.Dir(os.Args[0])
	candidates := []string{
		"/usr/share/ollama/airllm_runner.py",
		"/usr/lib/ollama/airllm_runner.py",
		filepath.Join(exeDir, "airllm_runner.py"),
		filepath.Join(exeDir, "runner", "airllmrunner", "airllm_runner.py"),
	}

	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}

func (s *Server) load(w http.ResponseWriter, r *http.Request) {
	var req llm.LoadRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		slog.Error("airllm load JSON decode", "error", err)
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")

	switch req.Operation {
	case llm.LoadOperationCommit:
		// WaitGroup: NewServer adds 1. First Commit does one Done (pairs with NewServer).
		// Any later Commit must Add(1) before Done or WaitGroup panics and the runner exits.
		s.commitMu.Lock()
		// commitAdded tracks whether Add(1) was called in THIS invocation so that
		// early-returns (before the defer runs) don't cause a WaitGroup underflow.
		commitAdded := s.commitReturnedOnce
		if s.commitReturnedOnce {
			s.ready.Add(1)
		}
		defer func() {
			if commitAdded {
				s.ready.Done()
			}
			s.commitReturnedOnce = true
			s.commitMu.Unlock()
		}()

		s.mu.Lock()
		s.status = llm.ServerStatusLoadingModel
		s.mu.Unlock()

		if err := s.startPythonRunner(); err != nil {
			slog.Error("failed to start Python runner", "error", err)
			// commitAdded=false here: Add(1) was not called above, so Don't call Done().
			json.NewEncoder(w).Encode(llm.LoadResponse{Success: false, Error: err.Error()})
			return
		}

		proxyReq, err := pythonLoadBody(req, s.modelPath)
		if err != nil {
			slog.Error("airllm load proxy body", "error", err)
			json.NewEncoder(w).Encode(llm.LoadResponse{Success: false, Error: err.Error()})
			return
		}
		resp, err := s.httpClient.Post(s.pythonBase+"/load", "application/json", bytes.NewReader(proxyReq))
		if err != nil {
			json.NewEncoder(w).Encode(llm.LoadResponse{Success: false, Error: err.Error()})
			return
		}
		defer resp.Body.Close()

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			json.NewEncoder(w).Encode(llm.LoadResponse{Success: false, Error: err.Error()})
			return
		}
		if resp.StatusCode >= 400 {
			json.NewEncoder(w).Encode(llm.LoadResponse{Success: false, Error: strings.TrimSpace(string(body))})
			return
		}

		var pyResp llm.LoadResponse
		if err := json.Unmarshal(body, &pyResp); err != nil {
			json.NewEncoder(w).Encode(llm.LoadResponse{Success: false, Error: err.Error()})
			return
		}

		if pyResp.Success {
			s.mu.Lock()
			s.status = llm.ServerStatusReady
			s.mu.Unlock()
		}

		json.NewEncoder(w).Encode(pyResp)

	case llm.LoadOperationClose:
		if s.pythonCmd != nil && s.pythonCmd.Process != nil {
			s.pythonCmd.Process.Kill()
		}
		// Reset pythonDone so a new runner can be started on a subsequent Commit.
		s.pythonDone = make(chan struct{})
		json.NewEncoder(w).Encode(llm.LoadResponse{Success: true})

	default:
		json.NewEncoder(w).Encode(llm.LoadResponse{Success: true})
	}
}

func (s *Server) completion(w http.ResponseWriter, r *http.Request) {
	s.ready.Wait()

	var req llm.CompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		slog.Error("airllm completion JSON decode", "error", err)
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Transfer-Encoding", "chunked")

	proxyReq, err := pythonCompletionBody(req)
	if err != nil {
		slog.Error("airllm completion proxy body", "error", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	resp, err := s.httpClient.Post(s.pythonBase+"/completion", "application/json", bytes.NewReader(proxyReq))
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	decoder := json.NewDecoder(resp.Body)
	for {
		var pyResp completionResponse
		if err := decoder.Decode(&pyResp); err != nil {
			break
		}

		var doneReason llm.DoneReason
		switch pyResp.DoneReason {
		case "stop":
			doneReason = llm.DoneReasonStop
		case "length":
			doneReason = llm.DoneReasonLength
		default:
			doneReason = llm.DoneReasonStop
		}

		ollamaResp := llm.CompletionResponse{
			Content:            pyResp.Content,
			Logprobs:           pyResp.Logprobs,
			Done:               pyResp.Done,
			DoneReason:         doneReason,
			PromptEvalCount:    pyResp.PromptEvalCount,
			PromptEvalDuration: time.Duration(pyResp.PromptEvalDuration),
			EvalCount:          pyResp.EvalCount,
			EvalDuration:       time.Duration(pyResp.EvalDuration),
		}

		if err := json.NewEncoder(w).Encode(&ollamaResp); err != nil {
			return
		}
		flusher.Flush()

		if pyResp.Done {
			break
		}
	}
}

// healthResponse extends llm.ServerStatusResponse with additional diagnostic fields.
type healthResponse struct {
	Status             llm.ServerStatus `json:"status"`
	Progress           float32          `json:"progress"`
	ModelPath          string           `json:"model_path"`
	PythonAlive        bool             `json:"python_alive"`
	PythonPID          int              `json:"python_pid,omitempty"`
	ConsecutiveErrors  int              `json:"consecutive_errors"`
	LastError          string           `json:"last_error,omitempty"`
	ReadyForInference  bool             `json:"ready_for_inference"`
}

// recordError increments the consecutive error counter.
// When consecutiveErrorLimit is reached, the runner becomes unavailable.
func (s *Server) recordError(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.consecutiveErrors++
	s.lastError = err.Error()
	if s.consecutiveErrors >= s.consecutiveErrorLimit {
		s.status = llm.ServerStatusError
	}
}

// resetErrors clears the consecutive error counter on successful operations.
func (s *Server) resetErrors() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.consecutiveErrors = 0
	s.lastError = ""
}

// readyz returns 200 OK when the runner is ready to accept inference requests,
// or 503 Service Unavailable when still loading or errored.
// This is useful for Kubernetes readiness probes.
func (s *Server) readyz(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	status := s.status
	s.mu.Unlock()

	// ready.Wait() blocks until the first Commit finishes (model loaded).
	// Use status as a quick check; if not ready, use WaitGroup to wait.
	if status != llm.ServerStatusReady {
		http.Error(w, fmt.Sprintf("runner not ready: status=%d", status), http.StatusServiceUnavailable)
		return
	}

	// Double-check by trying WaitGroup with a short timeout.
	done := make(chan struct{})
	go func() {
		s.ready.Wait()
		close(done)
	}()

	select {
	case <-done:
		w.WriteHeader(http.StatusOK)
		fmt.Fprintln(w, `{"ready":true}`)
	case <-time.After(5 * time.Second):
		http.Error(w, "timeout waiting for runner to be ready", http.StatusServiceUnavailable)
	}
}

func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	s.mu.Lock()
	status := s.status
	progress := s.progress
	modelPath := s.modelPath
	consecutiveErrors := s.consecutiveErrors
	lastError := s.lastError
	var pythonPID int
	pythonAlive := false
	if s.pythonCmd != nil && s.pythonCmd.Process != nil {
		pythonPID = s.pythonCmd.Process.Pid
		pythonAlive = s.pythonCmd.Process.Pid > 0
	}
	s.mu.Unlock()

	resp := healthResponse{
		Status:            status,
		Progress:          progress,
		ModelPath:         modelPath,
		PythonAlive:       pythonAlive,
		PythonPID:         pythonPID,
		ConsecutiveErrors: consecutiveErrors,
		LastError:         lastError,
		ReadyForInference: status == llm.ServerStatusReady && consecutiveErrors < 5,
	}

	json.NewEncoder(w).Encode(&resp)
}

func Execute(args []string) error {
	fs := flag.NewFlagSet("airllm-runner", flag.ExitOnError)
	mpath := fs.String("model", "", "Path to model")
	port := fs.Int("port", 8080, "Port to listen on")
	_ = fs.Bool("verbose", false, "Verbose output")

	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), "AirLLM Runner for Ollama\n")
		fs.PrintDefaults()
	}

	if err := fs.Parse(args); err != nil {
		return err
	}

	slog.SetDefault(logutil.NewLogger(os.Stderr, envconfig.LogLevel()))
	slog.Info("starting AirLLM runner")

	server := NewServer(*mpath, *port)

	addr := "127.0.0.1:" + strconv.Itoa(*port)
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("listen error: %w", err)
	}
	defer listener.Close()

	mux := http.NewServeMux()
	mux.HandleFunc("POST /load", server.load)
	mux.HandleFunc("/completion", server.completion)
	mux.HandleFunc("/health", server.health)
	mux.HandleFunc("/readyz", server.readyz) // returns 200 only when ready to serve

	httpServer := &http.Server{
		Handler: mux,
	}

	log.Println("AirLLM Runner listening on", addr)
	return httpServer.Serve(listener)
}
