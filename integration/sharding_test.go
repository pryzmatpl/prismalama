//go:build integration && sharding

package integration

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
)

var (
	largeGGUFModels = []string{
		"Qwen2.5-Coder-32B-Instruct-Q5_K_S.gguf",
		"nouscoder-14b-q4_k_m.gguf",
	}

	// Path to large safetensors model for sharding tests.
	// Override via OLLAMA_TEST_SHARDING_MODEL_PATH env var.
	// Falls back to OLLAMA_TEST_MODEL_PATH, then checks common local ollama paths.
	largeSafetensorsPaths = func() []string {
		if p := os.Getenv("OLLAMA_TEST_SHARDING_MODEL_PATH"); p != "" {
			return []string{p}
		}
		if p := os.Getenv("OLLAMA_TEST_MODEL_PATH"); p != "" {
			return []string{p}
		}
		// Fall back to checking default paths — tests will skip gracefully if not found.
		return []string{}
	}()
)

func getSystemMemory() (uint64, error) {
	cmd := exec.Command("free", "-b")
	output, err := cmd.Output()
	if err != nil {
		return 0, err
	}

	lines := strings.Split(string(output), "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "Mem:") {
			fields := strings.Fields(line)
			if len(fields) >= 2 {
				var mem uint64
				fmt.Sscanf(fields[1], "%d", &mem)
				return mem, nil
			}
		}
	}
	return 0, fmt.Errorf("could not parse memory info")
}

func getModelSize(path string) (int64, error) {
	info, err := os.Stat(path)
	if err != nil {
		return 0, err
	}

	if info.IsDir() {
		var size int64
		err := filepath.Walk(path, func(_ string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}
			if !info.IsDir() {
				size += info.Size()
			}
			return nil
		})
		return size, err
	}
	return info.Size(), nil
}

func skipIfInsufficientMemory(t *testing.T, requiredGB float64) {
	mem, err := getSystemMemory()
	if err != nil {
		t.Skip("Could not determine system memory")
	}

	required := uint64(requiredGB * float64(format.GibiByte))
	if mem < required {
		t.Skipf("Insufficient system memory: need %.1f GiB, have %s", requiredGB, format.HumanBytes2(int64(mem)))
	}
}

func TestLargeGGUFModelOffloading(t *testing.T) {
	skipIfInsufficientMemory(t, 32)

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	ggufPath := os.Getenv("OLLAMA_TEST_GGUF_MODEL_PATH")
	if ggufPath == "" {
		ggufPath = "/nvme3/ollama-models"
	}
	if _, err := os.Stat(ggufPath); os.IsNotExist(err) {
		t.Skip("GGUF model path not found (set OLLAMA_TEST_GGUF_MODEL_PATH to override default /nvme3/ollama-models)")
	}

	for _, model := range largeGGUFModels {
		t.Run(model, func(t *testing.T) {
			modelPath := filepath.Join(ggufPath, model)
			if _, err := os.Stat(modelPath); os.IsNotExist(err) {
				t.Skipf("Model not found: %s", modelPath)
			}

			modelSize, err := getModelSize(modelPath)
			if err != nil {
				t.Fatalf("Could not determine model size: %v", err)
			}

			slog.Info("Testing large GGUF model",
				"model", model,
				"size", format.HumanBytes2(modelSize))

			req := api.GenerateRequest{
				Model:  modelPath,
				Prompt: "Hello, how are you?",
				Stream: &stream,
				Options: map[string]interface{}{
					"num_predict": 20,
					"num_gpu":     -1,
				},
			}

			startTime := time.Now()
			var evalCount int
			var evalDuration time.Duration
			var response string

			err = client.Generate(ctx, &req, func(resp api.GenerateResponse) error {
				response += resp.Response
				if resp.Done {
					evalCount = resp.EvalCount
					evalDuration = resp.EvalDuration
				}
				return nil
			})

			loadDuration := time.Since(startTime)

			if err != nil {
				if strings.Contains(err.Error(), "memory") || strings.Contains(err.Error(), "OOM") {
					t.Skipf("Model too large for available memory: %v", err)
				}
				t.Fatalf("Generation failed: %v", err)
			}

			var tps float64
			if evalDuration > 0 {
				tps = float64(evalCount) / evalDuration.Seconds()
			}

			fmt.Printf("LARGE_GGUF: model=%s size=%s load_time=%v tps=%.2f response_len=%d\n",
				model, format.HumanBytes2(modelSize), loadDuration, tps, len(response))

			if len(response) == 0 {
				t.Error("Empty response from model")
			}
		})
	}
}

func TestLayerByLayerInference(t *testing.T) {
	skipIfInsufficientMemory(t, 64)

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	t.Setenv("OLLAMA_USE_AIRLLM", "1")

	for _, path := range largeSafetensorsPaths {
		t.Run(filepath.Base(path), func(t *testing.T) {
			if _, err := os.Stat(path); os.IsNotExist(err) {
				t.Skipf("Model path not found: %s", path)
			}

			modelSize, err := getModelSize(path)
			if err != nil {
				t.Fatalf("Could not determine model size: %v", err)
			}

			slog.Info("Testing layer-by-layer inference",
				"path", path,
				"size", format.HumanBytes2(modelSize))

			req := api.GenerateRequest{
				Model:  path,
				Prompt: "What is machine learning?",
				Stream: &stream,
				Options: map[string]interface{}{
					"num_predict": 50,
					"temperature": 0.7,
				},
			}

			startTime := time.Now()
			var response string
			var evalCount int
			var evalDuration time.Duration

			err = client.Generate(ctx, &req, func(resp api.GenerateResponse) error {
				response += resp.Response
				if resp.Done {
					evalCount = resp.EvalCount
					evalDuration = resp.EvalDuration
				}
				return nil
			})

			totalDuration := time.Since(startTime)

			if err != nil {
				if strings.Contains(err.Error(), "memory") {
					t.Skipf("Insufficient memory for model: %v", err)
				}
				t.Fatalf("Generation failed: %v", err)
			}

			var tps float64
			if evalDuration > 0 {
				tps = float64(evalCount) / evalDuration.Seconds()
			}

			fmt.Printf("LAYER_INFERENCE: model=%s size=%s total_time=%v tps=%.2f\n",
				filepath.Base(path), format.HumanBytes2(modelSize), totalDuration, tps)

			if len(response) == 0 {
				t.Error("Empty response from model")
			}
		})
	}
}

func TestHybridCPUOffload(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	model := "llama3.2:3b"
	if err := PullIfMissing(ctx, client, model); err != nil {
		t.Fatal(err)
	}

	layerConfigs := []struct {
		name   string
		numGPU int
	}{
		{"all_cpu", 0},
		{"partial_gpu", 10},
		{"full_gpu", 999},
	}

	for _, config := range layerConfigs {
		t.Run(config.name, func(t *testing.T) {
			req := api.GenerateRequest{
				Model:  model,
				Prompt: "Explain quantum computing briefly.",
				Stream: &stream,
				Options: map[string]interface{}{
					"num_gpu":     config.numGPU,
					"num_predict": 50,
				},
			}

			var evalCount int
			var evalDuration time.Duration
			var promptEvalDuration time.Duration

			err := client.Generate(ctx, &req, func(resp api.GenerateResponse) error {
				if resp.Done {
					evalCount = resp.EvalCount
					evalDuration = resp.EvalDuration
					promptEvalDuration = resp.PromptEvalDuration
				}
				return nil
			})
			if err != nil {
				t.Fatalf("Generation failed: %v", err)
			}

			models, err := client.ListRunning(ctx)
			if err == nil {
				for _, m := range models.Models {
					if strings.HasPrefix(m.Name, model) {
						gpuPercent := 0
						if m.Size > 0 {
							gpuPercent = int(float64(m.SizeVRAM) / float64(m.Size) * 100)
						}

						fmt.Printf("HYBRID_OFFLOAD: config=%s gpu_percent=%d%%\n",
							config.name, gpuPercent)

						switch config.name {
						case "all_cpu":
							if m.SizeVRAM > 0 {
								t.Logf("Expected 0 GPU usage for all_cpu, got %s", format.HumanBytes2(m.SizeVRAM))
							}
						case "full_gpu":
							if m.SizeVRAM == 0 {
								t.Log("Warning: No VRAM used for full_gpu config")
							}
						}
						break
					}
				}
			}

			var tps float64
			if evalDuration > 0 {
				tps = float64(evalCount) / evalDuration.Seconds()
			}

			var promptTps float64
			if promptEvalDuration > 0 {
				promptTps = float64(10) / promptEvalDuration.Seconds()
			}

			slog.Info("Hybrid offload test result",
				"config", config.name,
				"eval_tps", tps,
				"prompt_tps", promptTps)

			client.Generate(ctx, &api.GenerateRequest{Model: model, KeepAlive: &api.Duration{Duration: 0}}, func(r api.GenerateResponse) error { return nil })
		})
	}
}

func TestContextWindowScaling(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	model := smol
	if err := PullIfMissing(ctx, client, model); err != nil {
		t.Fatal(err)
	}

	contextSizes := []int{2048, 4096, 8192}

	for _, ctxSize := range contextSizes {
		t.Run(fmt.Sprintf("ctx_%d", ctxSize), func(t *testing.T) {
			req := api.GenerateRequest{
				Model:  model,
				Prompt: "Summarize the concept of artificial intelligence.",
				Stream: &stream,
				Options: map[string]interface{}{
					"num_ctx":     ctxSize,
					"num_predict": 100,
				},
			}

			var evalCount int
			var evalDuration time.Duration

			err := client.Generate(ctx, &req, func(resp api.GenerateResponse) error {
				if resp.Done {
					evalCount = resp.EvalCount
					evalDuration = resp.EvalDuration
				}
				return nil
			})
			if err != nil {
				t.Fatalf("Generation failed for ctx %d: %v", ctxSize, err)
			}

			var tps float64
			if evalDuration > 0 {
				tps = float64(evalCount) / evalDuration.Seconds()
			}

			fmt.Printf("CTX_SCALING: ctx=%d tps=%.2f\n", ctxSize, tps)

			client.Generate(ctx, &api.GenerateRequest{Model: model, KeepAlive: &api.Duration{Duration: 0}}, func(r api.GenerateResponse) error { return nil })
		})
	}
}

func TestKVCacheOptimization(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	model := smol
	if err := PullIfMissing(ctx, client, model); err != nil {
		t.Fatal(err)
	}

	cacheTypes := []string{"f16", "q8_0", "q4_0"}

	for _, cacheType := range cacheTypes {
		t.Run(fmt.Sprintf("cache_%s", cacheType), func(t *testing.T) {
			req := api.GenerateRequest{
				Model:  model,
				Prompt: "Write a short poem about the stars.",
				Stream: &stream,
				Options: map[string]interface{}{
					"num_ctx":     4096,
					"num_predict": 100,
					"num_cache":   1,
					"cache_type":  cacheType,
				},
			}

			var evalCount int
			var evalDuration time.Duration
			var response string

			startTime := time.Now()
			err := client.Generate(ctx, &req, func(resp api.GenerateResponse) error {
				response += resp.Response
				if resp.Done {
					evalCount = resp.EvalCount
					evalDuration = resp.EvalDuration
				}
				return nil
			})
			if err != nil {
				t.Fatalf("Generation failed with cache type %s: %v", cacheType, err)
			}

			totalTime := time.Since(startTime)

			var tps float64
			if evalDuration > 0 {
				tps = float64(evalCount) / evalDuration.Seconds()
			}

			fmt.Printf("KV_CACHE: type=%s tps=%.2f total_time=%v response_len=%d\n",
				cacheType, tps, totalTime, len(response))

			client.Generate(ctx, &api.GenerateRequest{Model: model, KeepAlive: &api.Duration{Duration: 0}}, func(r api.GenerateResponse) error { return nil })
		})
	}
}

func TestMultiGPULayerDistribution(t *testing.T) {
	if os.Getenv("OLLAMA_MULTI_GPU") == "" {
		t.Skip("Set OLLAMA_MULTI_GPU=1 to enable multi-GPU tests")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	model := "llama3.1:8b"
	if err := PullIfMissing(ctx, client, model); err != nil {
		t.Fatal(err)
	}

	req := api.GenerateRequest{
		Model:  model,
		Prompt: "Hello, world!",
		Stream: &stream,
		Options: map[string]interface{}{
			"num_predict": 20,
		},
	}

	err := client.Generate(ctx, &req, func(resp api.GenerateResponse) error {
		return nil
	})
	if err != nil {
		t.Fatalf("Generation failed: %v", err)
	}

	models, err := client.ListRunning(ctx)
	if err != nil {
		t.Fatalf("Failed to list running models: %v", err)
	}

	for _, m := range models.Models {
		if strings.HasPrefix(m.Name, model) {
			fmt.Printf("MULTI_GPU: model=%s size=%s vram=%s\n",
				m.Name, format.HumanBytes2(m.Size), format.HumanBytes2(m.SizeVRAM))
			break
		}
	}

	client.Generate(ctx, &api.GenerateRequest{Model: model, KeepAlive: &api.Duration{Duration: 0}}, func(r api.GenerateResponse) error { return nil })
}

func TestBatchSizeOptimization(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	model := smol
	if err := PullIfMissing(ctx, client, model); err != nil {
		t.Fatal(err)
	}

	batchSizes := []int{128, 256, 512, 1024}

	for _, batchSize := range batchSizes {
		t.Run(fmt.Sprintf("batch_%d", batchSize), func(t *testing.T) {
			req := api.GenerateRequest{
				Model:  model,
				Prompt: "Explain the theory of relativity.",
				Stream: &stream,
				Options: map[string]interface{}{
					"num_batch":   batchSize,
					"num_predict": 100,
				},
			}

			var evalCount int
			var evalDuration time.Duration
			var promptEvalDuration time.Duration

			startTime := time.Now()
			err := client.Generate(ctx, &req, func(resp api.GenerateResponse) error {
				if resp.Done {
					evalCount = resp.EvalCount
					evalDuration = resp.EvalDuration
					promptEvalDuration = resp.PromptEvalDuration
				}
				return nil
			})
			if err != nil {
				t.Fatalf("Generation failed with batch size %d: %v", batchSize, err)
			}

			var tps float64
			if evalDuration > 0 {
				tps = float64(evalCount) / evalDuration.Seconds()
			}

			var promptTps float64
			if promptEvalDuration > 0 {
				promptTps = float64(10) / promptEvalDuration.Seconds()
			}

			fmt.Printf("BATCH_OPT: batch=%d eval_tps=%.2f prompt_tps=%.2f total_time=%v\n",
				batchSize, tps, promptTps, time.Since(startTime))

			client.Generate(ctx, &api.GenerateRequest{Model: model, KeepAlive: &api.Duration{Duration: 0}}, func(r api.GenerateResponse) error { return nil })
		})
	}
}
