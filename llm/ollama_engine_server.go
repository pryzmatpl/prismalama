package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"golang.org/x/sync/semaphore"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/template"
)

const ollamaEngineWaitDelay = 5 * time.Second

// ollamaEngineServer hosts a model via `runner --ollama-engine` (POST /load,
// POST /completion). Used when llama-server is not packaged.
type ollamaEngineServer struct {
	port      int
	cmd       *exec.Cmd
	done      chan struct{}
	doneErr   error
	client    *http.Client
	status    *StatusWriter
	options   api.Options
	modelPath string
	gpus      []ml.DeviceInfo
	ggml      *ggml.GGML

	sem         *semaphore.Weighted
	loadRequest LoadRequest
	tmpl        *template.Template

	mu  sync.RWMutex
	mem *ml.BackendMemory
}

// NewOllamaEngineServer starts this binary as `runner --ollama-engine --model`.
func NewOllamaEngineServer(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, modelPath string, f *ggml.GGML, adapters []string, opts api.Options, numParallel int) (LlamaServer, error) {
	if numParallel < 1 {
		numParallel = 1
	}
	opts.NumBatch = min(opts.NumBatch, opts.NumCtx)
	if opts.NumBatch < 1 {
		opts.NumBatch = 1
	}

	totalLayers := int(f.KV().BlockCount() + 1)
	gpuLayers := gpuLayersForEngine(gpus, opts.NumGPU, totalLayers)
	logGGMLGPUOffload(uint64(totalLayers), gpuLayers, false)

	numThreads := opts.NumThread
	if numThreads <= 0 {
		numThreads = runtime.NumCPU()
	}

	loadRequest := LoadRequest{
		LoraPath:       adapters,
		Parallel:       numParallel,
		BatchSize:      opts.NumBatch,
		FlashAttention: flashAttentionFromEnv(),
		KvSize:         opts.NumCtx * numParallel,
		KvCacheType:    envconfig.KvCacheType(),
		NumThreads:     numThreads,
		GPULayers:      gpuLayers,
		MultiUserCache: envconfig.MultiUserCache(),
	}

	s := &ollamaEngineServer{
		client:      newLlamaServerHTTPClient(),
		status:      NewStatusWriter(os.Stderr),
		options:     opts,
		modelPath:   modelPath,
		gpus:        slicesClone(gpus),
		ggml:        f,
		sem:         semaphore.NewWeighted(int64(numParallel)),
		loadRequest: loadRequest,
		tmpl:        chatTemplateFromGGUF(f),
	}

	if err := s.startProcess(gpus); err != nil {
		return nil, err
	}
	return s, nil
}

func slicesClone[T any](in []T) []T {
	if in == nil {
		return nil
	}
	out := make([]T, len(in))
	copy(out, in)
	return out
}

func flashAttentionFromEnv() ml.FlashAttentionType {
	if os.Getenv("OLLAMA_FLASH_ATTENTION") == "" {
		return ml.FlashAttentionAuto
	}
	if envconfig.FlashAttention(false) {
		return ml.FlashAttentionEnabled
	}
	return ml.FlashAttentionDisabled
}

func chatTemplateFromGGUF(f *ggml.GGML) *template.Template {
	if f == nil {
		return nil
	}
	raw := f.KV().ChatTemplate()
	if raw == "" {
		return nil
	}
	named, err := template.Named(raw)
	if err != nil || named == nil {
		return nil
	}
	t, err := template.Parse(string(named.Bytes))
	if err != nil {
		return nil
	}
	return t
}

// gpuLayersForEngine assigns layers to the first GPU. NumGPU 0 is CPU-only;
// NumGPU < 0 (auto) offloads every layer and lets streaming handle VRAM.
func gpuLayersForEngine(gpus []ml.DeviceInfo, numGPU, totalLayers int) ml.GPULayersList {
	if numGPU == 0 || len(gpus) == 0 || totalLayers <= 0 {
		return nil
	}
	n := totalLayers
	if numGPU > 0 && numGPU < n {
		n = numGPU
	}
	layers := make([]int, n)
	for i := range layers {
		layers[i] = i
	}
	return ml.GPULayersList{{DeviceID: gpus[0].DeviceID, Layers: layers}}
}

func ollamaEngineGenerateArgs(exe, modelPath string, port int) []string {
	args := []string{exe, "runner", "--ollama-engine"}
	if modelPath != "" {
		args = append(args, "--model", modelPath)
	}
	return append(args, "--port", strconv.Itoa(port))
}

func pickLoopbackPort() (int, error) {
	if a, err := net.ResolveTCPAddr("tcp", "localhost:0"); err == nil {
		if l, err := net.ListenTCP("tcp", a); err == nil {
			port := l.Addr().(*net.TCPAddr).Port
			_ = l.Close()
			return port, nil
		}
	}
	return rand.Intn(65535-49152) + 49152, nil
}

func (s *ollamaEngineServer) startProcess(gpus []ml.DeviceInfo) error {
	exe, err := os.Executable()
	if err != nil {
		return fmt.Errorf("ollama-engine: executable: %w", err)
	}
	if eval, err := filepath.EvalSymlinks(exe); err == nil {
		exe = eval
	}

	port, err := pickLoopbackPort()
	if err != nil {
		return fmt.Errorf("ollama-engine: port: %w", err)
	}

	args := ollamaEngineGenerateArgs(exe, s.modelPath, port)
	cmd := exec.Command(args[0], args[1:]...)
	cmd.WaitDelay = ollamaEngineWaitDelay
	cmd.SysProcAttr = LlamaServerSysProcAttr
	cmd.Stdout = s.status
	cmd.Stderr = s.status

	extraEnvs := ml.GetDevicesEnv(gpus)
	SetupLlamaServerCommandEnv(cmd, exe, ml.LibraryPaths(gpus), extraEnvs)
	pinROCmHIP(cmd, extraEnvs)

	slog.Info("starting ollama-engine runner", "cmd", cmd)
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("ollama-engine: start: %w", err)
	}

	s.cmd = cmd
	s.port = port
	s.done = make(chan struct{})
	go func() {
		s.doneErr = cmd.Wait()
		close(s.done)
	}()
	return nil
}

func pinROCmHIP(cmd *exec.Cmd, extraEnvs map[string]string) {
	if cmd == nil {
		return
	}
	rocr, ok := extraEnvs["ROCR_VISIBLE_DEVICES"]
	if !ok || rocr == "" {
		return
	}
	cmd.Env = stripEnvKeys(cmd.Env, []string{"HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"})
	if hipIdx := hipVisibleDeviceIndexForSingleROCR(runtime.GOOS, rocr); hipIdx != "" {
		cmd.Env = append(cmd.Env, "HIP_VISIBLE_DEVICES="+hipIdx)
	}
}

func stripEnvKeys(env []string, keys []string) []string {
	var out []string
nextVar:
	for _, e := range env {
		k, _, _ := strings.Cut(e, "=")
		for _, key := range keys {
			if strings.EqualFold(k, key) {
				continue nextVar
			}
		}
		out = append(out, e)
	}
	return out
}

func hipVisibleDeviceIndexForSingleROCR(goos, rocr string) string {
	if goos != "linux" || rocr == "" || strings.Contains(rocr, ",") {
		return ""
	}
	return "0"
}

func (s *ollamaEngineServer) endpoint(path string) string {
	return fmt.Sprintf("http://127.0.0.1:%d%s", s.port, path)
}

func (s *ollamaEngineServer) httpClient() *http.Client {
	if s.client != nil {
		return s.client
	}
	return defaultLlamaServerHTTPClient
}

func (s *ollamaEngineServer) ModelPath() string { return s.modelPath }

func (s *ollamaEngineServer) Pid() int {
	if s.cmd != nil && s.cmd.Process != nil {
		return s.cmd.Process.Pid
	}
	return 0
}

func (s *ollamaEngineServer) GetPort() int { return s.port }

func (s *ollamaEngineServer) HasExited() bool {
	return s.cmd != nil && s.cmd.ProcessState != nil && s.cmd.ProcessState.ExitCode() >= 0
}

func (s *ollamaEngineServer) ContextLength() int { return s.options.NumCtx }

func (s *ollamaEngineServer) processErr() error {
	if s.status != nil {
		if msg := s.status.LastError(); msg != "" {
			return errors.New(msg)
		}
	}
	if s.doneErr != nil {
		return s.doneErr
	}
	return errors.New("ollama-engine runner exited")
}

func (s *ollamaEngineServer) getServerStatus(ctx context.Context) (ServerStatus, error) {
	if s.HasExited() {
		return ServerStatusError, s.processErr()
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.endpoint("/health"), nil)
	if err != nil {
		return ServerStatusNotResponding, err
	}
	res, err := s.httpClient().Do(req)
	if err != nil {
		return ServerStatusNotResponding, err
	}
	defer res.Body.Close()
	if res.StatusCode >= 400 {
		body, _ := io.ReadAll(res.Body)
		return ServerStatusError, fmt.Errorf("health %d: %s", res.StatusCode, bytes.TrimSpace(body))
	}
	var st ServerStatusResponse
	if err := json.NewDecoder(res.Body).Decode(&st); err != nil {
		return ServerStatusNotResponding, err
	}
	return st.Status, nil
}

func (s *ollamaEngineServer) waitForStatus(ctx context.Context, want ReadyWait) error {
	stall := envconfig.LoadTimeout()
	deadline := time.Now().Add(stall)
	var last ServerStatus = -1
	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("timed out waiting for ollama-engine: %w", ctx.Err())
		case <-s.doneChan():
			return fmt.Errorf("ollama-engine process has terminated: %w", s.processErr())
		default:
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("timed out waiting for ollama-engine (%s)", want)
		}
		pollCtx, cancel := context.WithTimeout(ctx, 200*time.Millisecond)
		status, err := s.getServerStatus(pollCtx)
		cancel()
		if err == nil && want.match(status) {
			return nil
		}
		if err == nil && last != status {
			slog.Info("waiting for ollama-engine", "status", status, "want", want)
			last = status
		}
		time.Sleep(50 * time.Millisecond)
	}
}

func (s *ollamaEngineServer) doneChan() <-chan struct{} {
	if s.done != nil {
		return s.done
	}
	return nil
}

type ReadyWait string

const (
	waitHTTP  ReadyWait = "http"
	waitReady ReadyWait = "ready"
)

func (w ReadyWait) match(status ServerStatus) bool {
	switch w {
	case waitHTTP:
		return status == ServerStatusLaunched || status == ServerStatusLoadingModel || status == ServerStatusReady
	case waitReady:
		return status == ServerStatusReady
	default:
		return false
	}
}

func (s *ollamaEngineServer) Ping(ctx context.Context) error {
	_, err := s.getServerStatus(ctx)
	return err
}

func (s *ollamaEngineServer) WaitUntilRunning(ctx context.Context) error {
	return s.waitForStatus(ctx, waitReady)
}

func (s *ollamaEngineServer) Load(ctx context.Context, _ ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) ([]ml.DeviceID, error) {
	if requireFull && (s.options.NumGPU == 0 || len(gpus) == 0 && len(s.loadRequest.GPULayers) == 0) {
		return nil, ErrLoadRequiredFull
	}
	if len(gpus) > 0 {
		s.gpus = slicesClone(gpus)
		if s.ggml != nil {
			totalLayers := int(s.ggml.KV().BlockCount() + 1)
			s.loadRequest.GPULayers = gpuLayersForEngine(gpus, s.options.NumGPU, totalLayers)
		}
	}

	slog.Info("loading model via ollama-engine", "model", s.modelPath, "gpu_layers", s.loadRequest.GPULayers)

	if err := s.waitForStatus(ctx, waitHTTP); err != nil {
		return nil, err
	}

	status, err := s.getServerStatus(ctx)
	if err == nil && status == ServerStatusReady {
		return gpuLayerDeviceIDs(s.loadRequest.GPULayers), nil
	}

	req := s.loadRequest
	req.Operation = LoadOperationCommit
	resp, err := s.initModel(ctx, req)
	if err != nil {
		return nil, err
	}
	if !resp.Success {
		errMsg := resp.Error
		if errMsg == "" {
			errMsg = "load commit failed"
		}
		return nil, errors.New(errMsg)
	}
	if backendMemoryFromRunner(resp.Memory) {
		s.mu.Lock()
		s.mem = &resp.Memory
		s.mu.Unlock()
		resp.Memory.Log(slog.LevelInfo)
	}

	if err := s.waitForStatus(ctx, waitReady); err != nil {
		return nil, err
	}
	return gpuLayerDeviceIDs(s.loadRequest.GPULayers), nil
}

func gpuLayerDeviceIDs(layers ml.GPULayersList) []ml.DeviceID {
	out := make([]ml.DeviceID, 0, len(layers))
	for _, g := range layers {
		if len(g.Layers) > 0 {
			out = append(out, g.DeviceID)
		}
	}
	return out
}

func (s *ollamaEngineServer) initModel(ctx context.Context, req LoadRequest) (LoadResponse, error) {
	var buf bytes.Buffer
	if err := json.NewEncoder(&buf).Encode(req); err != nil {
		return LoadResponse{}, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, s.endpoint("/load"), &buf)
	if err != nil {
		return LoadResponse{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	res, err := s.httpClient().Do(httpReq)
	if err != nil {
		return LoadResponse{}, fmt.Errorf("ollama-engine load: %w", err)
	}
	defer res.Body.Close()
	body, err := io.ReadAll(res.Body)
	if err != nil {
		return LoadResponse{}, err
	}
	if res.StatusCode >= 400 {
		return LoadResponse{}, api.StatusError{StatusCode: res.StatusCode, ErrorMessage: strings.TrimSpace(string(body))}
	}
	var resp LoadResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return LoadResponse{}, fmt.Errorf("ollama-engine load decode: %w", err)
	}
	return resp, nil
}

func applyCompletionFormat(req *CompletionRequest) error {
	if len(req.Format) == 0 {
		return nil
	}
	switch string(req.Format) {
	case `null`, `""`:
		return nil
	case `"json"`:
		req.Grammar = grammarJSON
		return nil
	default:
		if req.Format[0] != '{' {
			return fmt.Errorf("invalid format: %q; expected \"json\" or a valid JSON Schema object", req.Format)
		}
		return nil
	}
}

func (s *ollamaEngineServer) Completion(ctx context.Context, req CompletionRequest, fn func(CompletionResponse)) error {
	if err := applyCompletionFormat(&req); err != nil {
		return err
	}
	if req.Options == nil {
		opts := api.DefaultOptions()
		req.Options = &opts
	}
	if err := s.sem.Acquire(ctx, 1); err != nil {
		return err
	}
	defer s.sem.Release(1)

	req.Options.NumPredict = boundedNumPredict(req.Options.NumPredict, s.options.NumCtx)

	status, err := s.getServerStatus(ctx)
	if err != nil {
		return err
	}
	if status != ServerStatusReady {
		return fmt.Errorf("unexpected server status: %s", status)
	}

	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(req); err != nil {
		return fmt.Errorf("failed to marshal completion request: %v", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, s.endpoint("/completion"), &buf)
	if err != nil {
		return err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	res, err := s.httpClient().Do(httpReq)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return err
		}
		return errors.New("model runner has unexpectedly stopped, this may be due to resource limitations or an internal error, check ollama server logs for details")
	}
	defer res.Body.Close()
	if res.StatusCode >= 400 {
		body, _ := io.ReadAll(res.Body)
		return api.StatusError{StatusCode: res.StatusCode, ErrorMessage: strings.TrimSpace(string(body))}
	}

	scanner := bufio.NewScanner(res.Body)
	scanner.Buffer(make([]byte, 0, llamaServerStreamInitialBufferSize), llamaServerStreamMaxBufferSize)
	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		evt, ok := bytes.CutPrefix(line, []byte("data: "))
		if !ok {
			evt = line
		}
		var c CompletionResponse
		if err := json.Unmarshal(evt, &c); err != nil {
			return fmt.Errorf("error unmarshalling llm prediction response: %v", err)
		}
		if fn != nil {
			fn(c)
		}
		if c.Done {
			return nil
		}
	}
	if err := scanner.Err(); err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(ctx.Err(), context.Canceled) {
			return ctx.Err()
		}
		return err
	}
	return nil
}

func chatMLPrompt(msgs []api.Message) string {
	var b strings.Builder
	for _, m := range msgs {
		role := m.Role
		if role == "" {
			role = "user"
		}
		fmt.Fprintf(&b, "<|im_start|>%s\n%s<|im_end|>\n", role, m.Content)
	}
	b.WriteString("<|im_start|>assistant\n")
	return b.String()
}

func (s *ollamaEngineServer) ApplyChatTemplate(_ context.Context, req ChatRequest) (string, error) {
	if s.tmpl != nil {
		var buf bytes.Buffer
		vals := template.Values{Messages: req.Messages, Tools: req.Tools}
		if err := s.tmpl.Execute(&buf, vals); err != nil {
			return "", err
		}
		return buf.String(), nil
	}
	return chatMLPrompt(req.Messages), nil
}

func (s *ollamaEngineServer) Chat(ctx context.Context, req ChatRequest, fn func(ChatResponse)) error {
	prompt, err := s.ApplyChatTemplate(ctx, req)
	if err != nil {
		return err
	}
	comp := CompletionRequest{
		Prompt:      prompt,
		Format:      req.Format,
		Options:     req.Options,
		Shift:       req.Shift,
		Logprobs:    req.Logprobs,
		TopLogprobs: req.TopLogprobs,
	}
	return s.Completion(ctx, comp, func(cr CompletionResponse) {
		if fn == nil {
			return
		}
		fn(ChatResponse{
			Message:            api.Message{Role: "assistant", Content: cr.Content},
			DoneReason:         cr.DoneReason,
			Done:               cr.Done,
			PromptEvalCount:    cr.PromptEvalCount,
			PromptEvalDuration: cr.PromptEvalDuration,
			EvalCount:          cr.EvalCount,
			EvalDuration:       cr.EvalDuration,
			Logprobs:           cr.Logprobs,
		})
	})
}

func (s *ollamaEngineServer) Embedding(ctx context.Context, input string) ([]float32, int, error) {
	var buf bytes.Buffer
	if err := json.NewEncoder(&buf).Encode(EmbeddingRequest{Content: input}); err != nil {
		return nil, 0, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, s.endpoint("/embedding"), &buf)
	if err != nil {
		return nil, 0, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	res, err := s.httpClient().Do(httpReq)
	if err != nil {
		return nil, 0, err
	}
	defer res.Body.Close()
	body, err := io.ReadAll(res.Body)
	if err != nil {
		return nil, 0, err
	}
	if res.StatusCode >= 400 {
		return nil, 0, api.StatusError{StatusCode: res.StatusCode, ErrorMessage: strings.TrimSpace(string(body))}
	}
	var resp EmbeddingResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, 0, err
	}
	return resp.Embedding, resp.PromptEvalCount, nil
}

func (s *ollamaEngineServer) Tokenize(_ context.Context, content string) ([]int, error) {
	n := utf8.RuneCountInString(content)
	if n == 0 {
		return nil, nil
	}
	count := max(1, n/4)
	out := make([]int, count)
	for i := range out {
		out[i] = i
	}
	return out, nil
}

func (s *ollamaEngineServer) Detokenize(_ context.Context, _ []int) (string, error) {
	return "", errors.New("detokenize is not supported by the ollama-engine runner")
}

func (s *ollamaEngineServer) Close() error {
	if s.cmd == nil || s.cmd.Process == nil {
		return nil
	}
	if s.cmd.ProcessState != nil {
		return nil
	}
	slog.Debug("stopping ollama-engine runner", "pid", s.Pid())
	if err := s.cmd.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
		return err
	}
	if s.done != nil {
		<-s.done
	}
	return nil
}

func (s *ollamaEngineServer) GetDeviceInfos(ctx context.Context) []ml.DeviceInfo {
	infos, err := ml.GetDevicesFromRunner(ctx, s)
	if err != nil {
		return s.gpus
	}
	if len(infos) == 0 {
		return s.gpus
	}
	return infos
}

func (s *ollamaEngineServer) MemorySize() (total, vram uint64) {
	s.mu.RLock()
	mem := s.mem
	s.mu.RUnlock()
	if mem == nil {
		if info, err := os.Stat(s.modelPath); err == nil {
			total = uint64(info.Size())
			if len(s.loadRequest.GPULayers) > 0 {
				vram = total
			}
		}
		return total, vram
	}
	total = mem.InputWeights + mem.CPU.Size()
	for _, g := range mem.GPUs {
		sz := g.Size()
		total += sz
		vram += sz
	}
	return total, vram
}

func (s *ollamaEngineServer) VRAMByGPU(id ml.DeviceID) uint64 {
	s.mu.RLock()
	mem := s.mem
	s.mu.RUnlock()
	if mem == nil {
		return 0
	}
	var sum uint64
	for _, g := range mem.GPUs {
		if g.DeviceID == id || (g.ID == id.ID && (id.Library == "" || g.Library == id.Library)) {
			sum += g.Size()
		}
	}
	return sum
}
