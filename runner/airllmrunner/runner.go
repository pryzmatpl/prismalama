package airllmrunner

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
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

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
)

type Server struct {
	modelPath  string
	port       int
	pythonCmd  *exec.Cmd
	status     llm.ServerStatus
	progress   float32
	mu         sync.Mutex
	ready      sync.WaitGroup
	httpClient *http.Client
	baseURL    string
}

type loadRequest struct {
	Operation      string           `json:"operation"`
	ModelPath      string           `json:"model_path"`
	LoraPath       []string         `json:"lora_path"`
	ProjectorPath  string           `json:"projector_path"`
	Parallel       int              `json:"parallel"`
	BatchSize      int              `json:"batch_size"`
	KvSize         int              `json:"kv_size"`
	KvCacheType    string           `json:"kv_cache_type"`
	FlashAttention string           `json:"flash_attention"`
	NumThreads     int              `json:"num_threads"`
	MultiUserCache bool             `json:"multi_user_cache"`
	GPULayers      ml.GPULayersList `json:"gpu_layers"`
	MainGPU        int              `json:"main_gpu"`
	UseMmap        bool             `json:"use_mmap"`
}

type loadResponse struct {
	Success bool   `json:"success"`
	Error   string `json:"error,omitempty"`
}

type statusResponse struct {
	Status   string  `json:"status"`
	Progress float32 `json:"progress"`
}

type completionRequest struct {
	Prompt      string                  `json:"prompt"`
	Images      []llm.ImageData         `json:"images"`
	Grammar     string                  `json:"grammar"`
	Options     *map[string]interface{} `json:"options"`
	Logprobs    bool                    `json:"logprobs"`
	TopLogprobs int                     `json:"top_logprobs"`
	Shift       bool                    `json:"shift"`
	Truncate    bool                    `json:"truncate"`
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

func NewServer(modelPath string, port int) *Server {
	s := &Server{
		modelPath: modelPath,
		port:      port,
		status:    llm.ServerStatusLaunched,
		httpClient: &http.Client{
			Timeout: 0,
		},
		baseURL: fmt.Sprintf("http://127.0.0.1:%d", port),
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
		"--port", strconv.Itoa(s.port),
	)
	cmd.Env = append(os.Environ(),
		"AIRLLM_COMPRESSION="+os.Getenv("AIRLLM_COMPRESSION"),
		"PYTHONPATH=/usr/share/ollama/airllm:/usr/share/ollama/airllm/air_llm",
	)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr

	s.pythonCmd = cmd
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start Python runner: %w", err)
	}

	go func() {
		cmd.Wait()
		slog.Info("Python runner exited")
	}()

	return s.waitForReady()
}

func (s *Server) waitForReady() error {
	for i := 0; i < 60; i++ {
		resp, err := s.httpClient.Get(s.baseURL + "/health")
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

func findPythonRunner() string {
	candidates := []string{
		"/usr/share/ollama/airllm_runner.py",
		"/usr/lib/ollama/airllm_runner.py",
		filepath.Join(filepath.Dir(os.Args[0]), "airllm_runner.py"),
	}

	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}

func (s *Server) load(w http.ResponseWriter, r *http.Request) {
	var req loadRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")

	switch req.Operation {
	case "commit":
		s.mu.Lock()
		s.status = llm.ServerStatusLoadingModel
		s.mu.Unlock()

		if err := s.startPythonRunner(); err != nil {
			slog.Error("failed to start Python runner", "error", err)
			json.NewEncoder(w).Encode(loadResponse{Success: false, Error: err.Error()})
			return
		}

		proxyReq, _ := json.Marshal(req)
		resp, err := s.httpClient.Post(s.baseURL+"/load", "application/json", strings.NewReader(string(proxyReq)))
		if err != nil {
			json.NewEncoder(w).Encode(loadResponse{Success: false, Error: err.Error()})
			return
		}
		defer resp.Body.Close()

		var pyResp loadResponse
		json.NewDecoder(resp.Body).Decode(&pyResp)

		if pyResp.Success {
			s.mu.Lock()
			s.status = llm.ServerStatusReady
			s.mu.Unlock()
			s.ready.Done()
		}

		json.NewEncoder(w).Encode(pyResp)

	case "close":
		if s.pythonCmd != nil && s.pythonCmd.Process != nil {
			s.pythonCmd.Process.Kill()
		}
		json.NewEncoder(w).Encode(loadResponse{Success: true})

	default:
		json.NewEncoder(w).Encode(loadResponse{Success: true})
	}
}

func (s *Server) completion(w http.ResponseWriter, r *http.Request) {
	s.ready.Wait()

	var req completionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Transfer-Encoding", "chunked")

	proxyReq, _ := json.Marshal(req)
	resp, err := s.httpClient.Post(s.baseURL+"/completion", "application/json", strings.NewReader(string(proxyReq)))
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

func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	s.mu.Lock()
	status := s.status
	progress := s.progress
	s.mu.Unlock()

	json.NewEncoder(w).Encode(&llm.ServerStatusResponse{
		Status:   status,
		Progress: progress,
	})
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

	httpServer := &http.Server{
		Handler: mux,
	}

	log.Println("AirLLM Runner listening on", addr)
	return httpServer.Serve(listener)
}
