//go:build integration && airllm

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
)

var (
	airllmModels = []string{
		"Qwen2.5-Coder-32B-Instruct",
		"nouscoder-14b",
	}

	largeSafetensorsModels = []string{
		"MiniMax-M2.5-Q4_1",
	}

	airLLMTestPrompt     = "Write a brief function in Python that calculates the factorial of a number."
	airLLMExpectedTokens = []string{"def", "factorial", "return", "if", "else"}
)

func isAirLLMModel(modelPath string) bool {
	safetensorsFile := filepath.Join(modelPath, "model.safetensors.index.json")
	if _, err := os.Stat(safetensorsFile); err == nil {
		return true
	}

	safetensorsFiles, _ := filepath.Glob(filepath.Join(modelPath, "*.safetensors"))
	if len(safetensorsFiles) > 0 {
		return true
	}

	configFile := filepath.Join(modelPath, "config.json")
	if data, err := os.ReadFile(configFile); err == nil {
		content := strings.ToLower(string(data))
		if strings.Contains(content, "safetensors") ||
			strings.Contains(content, "torch_dtype") ||
			strings.Contains(content, "transformers") {
			return true
		}
	}

	return false
}

func skipIfNoAirLLMEnv(t *testing.T) {
	if os.Getenv("OLLAMA_TEST_AIRLLM") == "" && os.Getenv("OLLAMA_TEST_EXISTING") == "" {
		t.Skip("Skipping AirLLM tests. Set OLLAMA_TEST_AIRLLM=1 to enable.")
	}
}

func TestAirLLMBasicGeneration(t *testing.T) {
	skipIfNoAirLLMEnv(t)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	model := os.Getenv("OLLAMA_AIRLLM_MODEL")
	if model == "" {
		model = airllmModels[0]
	}

	slog.Info("Testing AirLLM basic generation", "model", model)

	req := api.GenerateRequest{
		Model:  model,
		Prompt: "Hello, how are you?",
		Stream: &stream,
		Options: map[string]interface{}{
			"temperature": 0.7,
			"num_predict": 50,
		},
	}

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

	elapsed := time.Since(startTime)
	slog.Info("AirLLM generation completed", "duration", elapsed, "response_len", len(response))

	if len(response) == 0 {
		t.Error("Empty response received")
	}
}

func TestAirLLMCodeGeneration(t *testing.T) {
	skipIfNoAirLLMEnv(t)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	model := os.Getenv("OLLAMA_AIRLLM_MODEL")
	if model == "" {
		model = "Qwen2.5-Coder-32B-Instruct"
	}

	req := api.GenerateRequest{
		Model:  model,
		Prompt: airLLMTestPrompt,
		Stream: &stream,
		Options: map[string]interface{}{
			"temperature": 0.3,
			"num_predict": 256,
		},
	}

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

	responseLower := strings.ToLower(response)
	atLeastOne := false
	for _, token := range airLLMExpectedTokens {
		if strings.Contains(responseLower, token) {
			atLeastOne = true
			break
		}
	}

	if !atLeastOne {
		t.Errorf("Response does not contain expected code tokens. Response: %s", response)
	}

	slog.Info("AirLLM code generation test passed", "response_len", len(response))
}

func TestAirLLMChatCompletion(t *testing.T) {
	skipIfNoAirLLMEnv(t)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	model := os.Getenv("OLLAMA_AIRLLM_MODEL")
	if model == "" {
		model = airllmModels[0]
	}

	req := api.ChatRequest{
		Model: model,
		Messages: []api.Message{
			{
				Role:    "system",
				Content: "You are a helpful coding assistant.",
			},
			{
				Role:    "user",
				Content: "Write a simple hello world in Python.",
			},
		},
		Stream: &stream,
		Options: map[string]interface{}{
			"temperature": 0.5,
			"num_predict": 100,
		},
	}

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
			t.Fatalf("Chat failed: %v", err)
		}
	}

	if !strings.Contains(strings.ToLower(response), "print") && !strings.Contains(strings.ToLower(response), "hello") {
		t.Errorf("Response doesn't look like a hello world program: %s", response)
	}

	slog.Info("AirLLM chat completion test passed")
}

func TestAirLLMLongContext(t *testing.T) {
	skipIfNoAirLLMEnv(t)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	model := os.Getenv("OLLAMA_AIRLLM_MODEL")
	if model == "" {
		model = airllmModels[0]
	}

	longPrompt := strings.Repeat("This is a test sentence for context. ", 100)
	longPrompt += "\n\nNow summarize the above text in one sentence."

	req := api.GenerateRequest{
		Model:  model,
		Prompt: longPrompt,
		Stream: &stream,
		Options: map[string]interface{}{
			"temperature": 0.3,
			"num_predict": 100,
			"num_ctx":     4096,
		},
	}

	var response string
	done := make(chan error, 1)
	startTime := time.Now()

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

	elapsed := time.Since(startTime)
	slog.Info("AirLLM long context test completed", "duration", elapsed, "response_len", len(response))

	if len(response) == 0 {
		t.Error("Empty response for long context")
	}
}

func TestAirLLMStreaming(t *testing.T) {
	skipIfNoAirLLMEnv(t)

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	model := os.Getenv("OLLAMA_AIRLLM_MODEL")
	if model == "" {
		model = airllmModels[0]
	}

	streaming := true
	req := api.GenerateRequest{
		Model:  model,
		Prompt: "Count from 1 to 10.",
		Stream: &streaming,
		Options: map[string]interface{}{
			"temperature": 0.1,
			"num_predict": 50,
		},
	}

	chunks := 0
	var fullResponse string
	done := make(chan error, 1)

	go func() {
		err := client.Generate(ctx, &req, func(resp api.GenerateResponse) error {
			chunks++
			fullResponse += resp.Response
			return nil
		})
		done <- err
	}()

	select {
	case <-ctx.Done():
		t.Fatalf("Test timed out")
	case err := <-done:
		if err != nil {
			t.Fatalf("Streaming generation failed: %v", err)
		}
	}

	if chunks < 2 {
		t.Errorf("Expected multiple streaming chunks, got %d", chunks)
	}

	slog.Info("AirLLM streaming test passed", "chunks", chunks, "response_len", len(fullResponse))
}

func TestAirLLMCompressionModes(t *testing.T) {
	skipIfNoAirLLMEnv(t)

	compressionModes := []string{"4bit", "8bit"}

	for _, compression := range compressionModes {
		t.Run(compression, func(t *testing.T) {
			t.Setenv("AIRLLM_COMPRESSION", compression)

			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
			defer cancel()

			client, _, cleanup := InitServerConnection(ctx, t)
			defer cleanup()

			model := os.Getenv("OLLAMA_AIRLLM_MODEL")
			if model == "" {
				model = airllmModels[0]
			}

			req := api.GenerateRequest{
				Model:  model,
				Prompt: "Say hello.",
				Stream: &stream,
				Options: map[string]interface{}{
					"temperature": 0.7,
					"num_predict": 20,
				},
			}

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
				t.Fatalf("Test timed out for compression %s", compression)
			case err := <-done:
				if err != nil {
					t.Fatalf("Generation failed with %s compression: %v", compression, err)
				}
			}

			if len(response) == 0 {
				t.Errorf("Empty response with %s compression", compression)
			}

			slog.Info("AirLLM compression test passed", "compression", compression, "response_len", len(response))
		})
	}
}

func TestAirLLMModelDetection(t *testing.T) {
	skipIfNoAirLLMEnv(t)

	// Allow model dirs to be specified via env var as a comma-separated list.
	envDirs := os.Getenv("OLLAMA_TEST_MODEL_DIRS")
	var modelDirs []string
	if envDirs != "" {
		for _, d := range strings.Split(envDirs, ",") {
			d = strings.TrimSpace(d)
			if d != "" {
				modelDirs = append(modelDirs, d)
			}
		}
	}
	// If no env var, use defaults (will be skipped gracefully by the dir check below).
	if modelDirs == nil {
		modelDirs = []string{
			"/nvme3/AI Models/MiniMaxM2.5",
			"/run/media/piotro/CACHE/airllm/Qwen2.5-Coder-32B-Instruct",
		}
	}

	for _, dir := range modelDirs {
		t.Run(filepath.Base(dir), func(t *testing.T) {
			if _, err := os.Stat(dir); os.IsNotExist(err) {
				t.Skipf("Model directory not found: %s", dir)
			}

			isAirLLM := isAirLLMModel(dir)
			slog.Info("Model detection result", "dir", dir, "is_airllm", isAirLLM)

			if strings.Contains(dir, "Qwen") || strings.Contains(dir, "MiniMax") {
				if !isAirLLM {
					t.Errorf("Expected %s to be detected as AirLLM model", dir)
				}
			}
		})
	}
}

func TestAirLLMPerformance(t *testing.T) {
	skipIfNoAirLLMEnv(t)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	model := os.Getenv("OLLAMA_AIRLLM_MODEL")
	if model == "" {
		model = airllmModels[0]
	}

	prompt := "Write a detailed explanation of machine learning."

	req := api.GenerateRequest{
		Model:  model,
		Prompt: prompt,
		Stream: &stream,
		Options: map[string]interface{}{
			"temperature": 0.7,
			"num_predict": 200,
		},
	}

	var evalCount, promptEvalCount int
	var evalDuration, promptEvalDuration time.Duration
	var response string

	startTime := time.Now()
	done := make(chan error, 1)

	go func() {
		err := client.Generate(ctx, &req, func(resp api.GenerateResponse) error {
			response += resp.Response
			if resp.Done {
				evalCount = resp.EvalCount
				promptEvalCount = resp.PromptEvalCount
				evalDuration = resp.EvalDuration
				promptEvalDuration = resp.PromptEvalDuration
			}
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

	totalTime := time.Since(startTime)

	var tokensPerSecond float64
	if evalDuration > 0 {
		tokensPerSecond = float64(evalCount) / evalDuration.Seconds()
	}

	var promptTokensPerSecond float64
	if promptEvalDuration > 0 {
		promptTokensPerSecond = float64(promptEvalCount) / promptEvalDuration.Seconds()
	}

	fmt.Printf("AIRLLM_PERF_HEADER:%s,%s,%s,%s,%s,%s,%s\n",
		"MODEL", "TOTAL_TIME", "EVAL_TOKENS", "EVAL_TPS", "PROMPT_TOKENS", "PROMPT_TPS", "RESPONSE_LEN")
	fmt.Printf("AIRLLM_PERF_DATA:%s,%v,%d,%.2f,%d,%.2f,%d\n",
		model, totalTime, evalCount, tokensPerSecond, promptEvalCount, promptTokensPerSecond, len(response))

	slog.Info("AirLLM performance metrics",
		"model", model,
		"total_time", totalTime,
		"eval_tokens", evalCount,
		"eval_tps", tokensPerSecond,
		"prompt_tokens", promptEvalCount,
		"prompt_tps", promptTokensPerSecond)

	if evalCount == 0 {
		t.Error("No tokens were evaluated")
	}
}
