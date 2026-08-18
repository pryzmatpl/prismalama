package discover

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"os"
	"os/exec"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

// findLlamaServer and ollamaEngineDiscover are vars so tests can stub discovery
// without spawning GPU subprocesses.
var findLlamaServer = llm.FindLlamaServer
var ollamaEngineDiscover = ollamaEngineDiscoverDevices

const ollamaEngineDiscoveryWaitDelay = 5 * time.Second

type engineCmdRunner struct {
	cmd    *exec.Cmd
	port   int
	exited atomic.Bool
	done   chan struct{}
}

func (r *engineCmdRunner) GetPort() int { return r.port }

func (r *engineCmdRunner) HasExited() bool { return r.exited.Load() }

func (r *engineCmdRunner) Stop() {
	if r.cmd != nil && r.cmd.Process != nil {
		_ = r.cmd.Process.Kill()
	}
	if r.done != nil {
		<-r.done
	}
}

func ollamaEngineRunnerArgs(exe string, port int) []string {
	return []string{exe, "runner", "--ollama-engine", "--port", strconv.Itoa(port)}
}

func pickLocalPort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	port := l.Addr().(*net.TCPAddr).Port
	_ = l.Close()
	return port, nil
}

// ollamaEngineDiscoverDevices starts this binary as `runner --ollama-engine`
// and reads GET /info. Used when llama-server is not packaged (JAISIU-2298).
// Generate uses the same runner via llm.NewOllamaEngineServer.
func ollamaEngineDiscoverDevices(ctx context.Context, libDirs []string, extraEnvs map[string]string) ([]ml.DeviceInfo, *llm.StatusWriter, error) {
	status := llm.NewStatusWriter(llamaServerDiscoveryOutput(ctx))
	exe, err := os.Executable()
	if err != nil {
		return nil, status, fmt.Errorf("ollama-engine discovery: executable: %w", err)
	}

	port, err := pickLocalPort()
	if err != nil {
		return nil, status, fmt.Errorf("ollama-engine discovery: port: %w", err)
	}

	args := ollamaEngineRunnerArgs(exe, port)
	cmd := exec.CommandContext(ctx, args[0], args[1:]...)
	cmd.WaitDelay = ollamaEngineDiscoveryWaitDelay
	cmd.Stdout = status
	cmd.Stderr = status
	llm.SetupLlamaServerCommandEnv(cmd, exe, libDirs, extraEnvs)

	start := time.Now()
	defer func() {
		slog.Debug("ollama-engine device discovery took", "duration", time.Since(start), "libDirs", libDirs)
	}()

	if err := cmd.Start(); err != nil {
		return nil, status, fmt.Errorf("ollama-engine discovery: start: %w", err)
	}

	runner := &engineCmdRunner{cmd: cmd, port: port, done: make(chan struct{})}
	go func() {
		_ = cmd.Wait()
		runner.exited.Store(true)
		close(runner.done)
	}()
	defer runner.Stop()

	devices, err := ml.GetDevicesFromRunner(ctx, runner)
	if err != nil {
		return nil, status, fmt.Errorf("ollama-engine discovery: /info: %w", err)
	}
	return devices, status, nil
}
