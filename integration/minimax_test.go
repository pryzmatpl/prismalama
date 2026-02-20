//go:build integration && minimax

package integration

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
)

var (
	minimaxModelPath = "/nvme3/AI Models/MiniMaxM2.5"
	minimaxGGUFFiles = []string{
		"MiniMax-M2.5-Q4_1-00001-of-00004.gguf",
		"MiniMax-M2.5-Q4_1-00002-of-00004.gguf",
		"MiniMax-M2.5-Q4_1-00003-of-00004.gguf",
		"MiniMax-M2.5-Q4_1-00004-of-00004.gguf",
	}
)

func skipIfNoMiniMax(t *testing.T) {
	if _, err := os.Stat(minimaxModelPath); os.IsNotExist(err) {
		t.Skip("MiniMax model not found at " + minimaxModelPath)
	}
}

func TestMiniMaxModelDetection(t *testing.T) {
	skipIfNoMiniMax(t)

	files, err := os.ReadDir(minimaxModelPath)
	if err != nil {
		t.Fatalf("Failed to read MiniMax directory: %v", err)
	}

	slog.Info("MiniMax model files", "count", len(files))
	for _, f := range files {
		info, err := f.Info()
		if err != nil {
			continue
		}
		fmt.Printf("MINIMAX_FILE: name=%s size=%s\n", f.Name(), format.HumanBytes2(info.Size()))
	}

	for _, expectedFile := range minimaxGGUFFiles {
		found := false
		for _, f := range files {
			if f.Name() == expectedFile {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Expected file not found: %s", expectedFile)
		}
	}
}

func TestMiniMaxModelSize(t *testing.T) {
	skipIfNoMiniMax(t)

	var totalSize int64
	for _, f := range minimaxGGUFFiles {
		path := filepath.Join(minimaxModelPath, f)
		if info, err := os.Stat(path); err == nil {
			totalSize += info.Size()
		}
	}

	fmt.Printf("MINIMAX_SIZE: total=%s\n", format.HumanBytes2(totalSize))

	if totalSize < 100*int64(format.GibiByte) {
		t.Logf("Warning: MiniMax model size seems smaller than expected: %s", format.HumanBytes2(totalSize))
	}

	var vramSize int64 = 24 * int64(format.GibiByte)
	if totalSize > vramSize*2 {
		fmt.Printf("MINIMAX_STRATEGY: Model (%s) is larger than 2x VRAM (%s), AirLLM recommended\n",
			format.HumanBytes2(totalSize), format.HumanBytes2(vramSize))
	}
}

func TestMiniMaxInference(t *testing.T) {
	skipIfNoMiniMax(t)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	t.Setenv("OLLAMA_USE_AIRLLM", "1")

	req := api.GenerateRequest{
		Model:  minimaxModelPath,
		Prompt: "Hello, how are you today?",
		Stream: &stream,
		Options: map[string]interface{}{
			"temperature": 0.7,
			"num_predict": 50,
		},
	}

	slog.Info("Starting MiniMax inference test")
	startTime := time.Now()

	var response string
	var evalCount int
	var evalDuration time.Duration
	var promptEvalCount int
	var promptEvalDuration time.Duration

	done := make(chan error, 1)
	go func() {
		err := client.Generate(ctx, &req, func(resp api.GenerateResponse) error {
			response += resp.Response
			if resp.Done {
				evalCount = resp.EvalCount
				evalDuration = resp.EvalDuration
				promptEvalCount = resp.PromptEvalCount
				promptEvalDuration = resp.PromptEvalDuration
			}
			return nil
		})
		done <- err
	}()

	select {
	case <-ctx.Done():
		t.Fatalf("Test timed out after %v", time.Since(startTime))
	case err := <-done:
		if err != nil {
			if strings.Contains(err.Error(), "memory") || strings.Contains(err.Error(), "OOM") {
				t.Skipf("Insufficient memory for MiniMax: %v", err)
			}
			t.Fatalf("Generation failed: %v", err)
		}
	}

	totalDuration := time.Since(startTime)

	var tps float64
	if evalDuration > 0 {
		tps = float64(evalCount) / evalDuration.Seconds()
	}

	var promptTps float64
	if promptEvalDuration > 0 {
		promptTps = float64(promptEvalCount) / promptEvalDuration.Seconds()
	}

	fmt.Printf("MINIMAX_INFERENCE:\n")
	fmt.Printf("  Total time: %v\n", totalDuration)
	fmt.Printf("  Response length: %d chars\n", len(response))
	fmt.Printf("  Eval tokens: %d\n", evalCount)
	fmt.Printf("  Prompt tokens: %d\n", promptEvalCount)
	fmt.Printf("  Eval TPS: %.2f\n", tps)
	fmt.Printf("  Prompt TPS: %.2f\n", promptTps)

	fmt.Printf("MINIMAX_PERF: total_time=%v eval_tps=%.2f prompt_tps=%.2f response_len=%d\n",
		totalDuration, tps, promptTps, len(response))

	if len(response) == 0 {
		t.Error("Empty response from MiniMax model")
	}
}

func TestMiniMaxChatCompletion(t *testing.T) {
	skipIfNoMiniMax(t)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	t.Setenv("OLLAMA_USE_AIRLLM", "1")

	req := api.ChatRequest{
		Model: minimaxModelPath,
		Messages: []api.Message{
			{
				Role:    "system",
				Content: "You are a helpful AI assistant.",
			},
			{
				Role:    "user",
				Content: "What is the capital of France?",
			},
		},
		Stream: &stream,
		Options: map[string]interface{}{
			"temperature": 0.7,
			"num_predict": 100,
		},
	}

	slog.Info("Starting MiniMax chat test")
	startTime := time.Now()

	var response string
	done := make(chan error, 1)

	go func() {
		err := client.Chat(ctx, &req, func(resp api.ChatResponse) error {
			response += resp.Message.Content
			return nil
		})
		done <- err
	}()

	select {
	case <-ctx.Done():
		t.Fatalf("Test timed out")
	case err := <-done:
		if err != nil {
			if strings.Contains(err.Error(), "memory") {
				t.Skipf("Insufficient memory: %v", err)
			}
			t.Fatalf("Chat failed: %v", err)
		}
	}

	fmt.Printf("MINIMAX_CHAT: duration=%v response_len=%d\n", time.Since(startTime), len(response))

	if !strings.Contains(strings.ToLower(response), "paris") {
		t.Logf("Warning: Response doesn't contain expected answer 'Paris': %s", response)
	}
}

func TestMiniMaxLongContext(t *testing.T) {
	skipIfNoMiniMax(t)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	t.Setenv("OLLAMA_USE_AIRLLM", "1")

	longPrompt := strings.Repeat("This is a test sentence. ", 200)
	longPrompt += "\n\nNow summarize the above text in one sentence."

	req := api.GenerateRequest{
		Model:  minimaxModelPath,
		Prompt: longPrompt,
		Stream: &stream,
		Options: map[string]interface{}{
			"temperature": 0.5,
			"num_predict": 100,
			"num_ctx":     8192,
		},
	}

	slog.Info("Starting MiniMax long context test")
	startTime := time.Now()

	var response string
	done := make(chan error, 1)

	go func() {
		err := client.Generate(ctx, &req, func(resp api.GenerateResponse) error {
			response += resp.Response
			return nil
		})
		done <- err
	}()

	select {
	case <-ctx.Done():
		t.Fatalf("Test timed out")
	case err := <-done:
		if err != nil {
			if strings.Contains(err.Error(), "context") {
				t.Logf("Context length issue: %v", err)
			} else if strings.Contains(err.Error(), "memory") {
				t.Skipf("Insufficient memory: %v", err)
			} else {
				t.Fatalf("Generation failed: %v", err)
			}
		}
	}

	fmt.Printf("MINIMAX_LONG_CTX: duration=%v prompt_len=%d response_len=%d\n",
		time.Since(startTime), len(longPrompt), len(response))
}

func TestMiniMaxCodeGeneration(t *testing.T) {
	skipIfNoMiniMax(t)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	t.Setenv("OLLAMA_USE_AIRLLM", "1")

	req := api.GenerateRequest{
		Model:  minimaxModelPath,
		Prompt: "Write a Python function that implements the quicksort algorithm.",
		Stream: &stream,
		Options: map[string]interface{}{
			"temperature": 0.3,
			"num_predict": 300,
		},
	}

	slog.Info("Starting MiniMax code generation test")

	var response string
	done := make(chan error, 1)

	go func() {
		err := client.Generate(ctx, &req, func(resp api.GenerateResponse) error {
			response += resp.Response
			return nil
		})
		done <- err
	}()

	select {
	case <-ctx.Done():
		t.Fatalf("Test timed out")
	case err := <-done:
		if err != nil {
			t.Fatalf("Generation failed: %v", err)
		}
	}

	codeIndicators := []string{"def", "return", "if", "for", "while", "sort"}
	foundCount := 0
	responseLower := strings.ToLower(response)
	for _, indicator := range codeIndicators {
		if strings.Contains(responseLower, indicator) {
			foundCount++
		}
	}

	fmt.Printf("MINIMAX_CODE: response_len=%d code_indicators=%d/%d\n",
		len(response), foundCount, len(codeIndicators))

	if foundCount < 3 {
		t.Logf("Warning: Response doesn't look like code (found %d/%d indicators)", foundCount, len(codeIndicators))
	}
}

func TestMiniMaxMemoryEfficiency(t *testing.T) {
	skipIfNoMiniMax(t)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	t.Setenv("OLLAMA_USE_AIRLLM", "1")
	t.Setenv("AIRLLM_COMPRESSION", "4bit")

	req := api.GenerateRequest{
		Model:  minimaxModelPath,
		Prompt: "Hello!",
		Stream: &stream,
		Options: map[string]interface{}{
			"num_predict": 20,
		},
	}

	slog.Info("Testing MiniMax with 4-bit compression")

	startTime := time.Now()
	var response string
	done := make(chan error, 1)

	go func() {
		err := client.Generate(ctx, &req, func(resp api.GenerateResponse) error {
			response += resp.Response
			return nil
		})
		done <- err
	}()

	select {
	case <-ctx.Done():
		t.Fatalf("Test timed out")
	case err := <-done:
		if err != nil {
			t.Fatalf("Generation failed: %v", err)
		}
	}

	fmt.Printf("MINIMAX_4BIT: duration=%v response_len=%d\n", time.Since(startTime), len(response))
}
