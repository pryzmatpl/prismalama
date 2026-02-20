//go:build integration && gpu

package integration

import (
	"context"
	"fmt"
	"log/slog"
	"math"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
)

func getROCmVRAM() (uint64, error) {
	cmd := exec.Command("rocm-smi", "--showmeminfo", "vram")
	output, err := cmd.Output()
	if err != nil {
		return 0, fmt.Errorf("rocm-smi not available: %w", err)
	}

	lines := strings.Split(string(output), "\n")
	for _, line := range lines {
		if strings.Contains(line, "Total") && strings.Contains(line, "MiB") {
			fields := strings.Fields(line)
			for i, f := range fields {
				if f == "Total" && i+1 < len(fields) {
					val, err := strconv.ParseFloat(fields[i+1], 64)
					if err == nil {
						return uint64(val * float64(format.MebiByte)), nil
					}
				}
			}
		}
	}
	return 0, fmt.Errorf("could not parse VRAM from rocm-smi output")
}

func getROCmFreeVRAM() (uint64, error) {
	cmd := exec.Command("rocm-smi", "--showmeminfo", "vram")
	output, err := cmd.Output()
	if err != nil {
		return 0, fmt.Errorf("rocm-smi not available: %w", err)
	}

	lines := strings.Split(string(output), "\n")
	for _, line := range lines {
		if strings.Contains(line, "Free") && strings.Contains(line, "MiB") {
			fields := strings.Fields(line)
			for i, f := range fields {
				if f == "Free" && i+1 < len(fields) {
					val, err := strconv.ParseFloat(fields[i+1], 64)
					if err == nil {
						return uint64(val * float64(format.MebiByte)), nil
					}
				}
			}
		}
	}
	return 0, fmt.Errorf("could not parse free VRAM from rocm-smi output")
}

func skipIfNoROCm(t *testing.T) {
	if _, err := exec.LookPath("rocm-smi"); err != nil {
		t.Skip("ROCm not available, skipping GPU tests")
	}
}

func TestGPUUtilization(t *testing.T) {
	skipIfNoROCm(t)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	model := smol
	if err := PullIfMissing(ctx, client, model); err != nil {
		t.Fatal(err)
	}

	req := api.GenerateRequest{
		Model:  model,
		Prompt: "Write a story about a GPU.",
		Stream: &stream,
		Options: map[string]interface{}{
			"temperature": 0.7,
			"num_predict": 100,
		},
	}

	beforeVRAM, _ := getROCmFreeVRAM()
	slog.Info("VRAM before generation", "free_vram", format.HumanBytes2(int64(beforeVRAM)))

	var evalCount int
	var evalDuration time.Duration
	done := make(chan error, 1)

	startTime := time.Now()
	go func() {
		err := client.Generate(ctx, &req, func(resp api.GenerateResponse) error {
			if resp.Done {
				evalCount = resp.EvalCount
				evalDuration = resp.EvalDuration
			}
			return nil
		})
		done <- err
	}()

	select {
	case <-ctx.Done():
		t.Fatal("Test timed out")
	case err := <-done:
		if err != nil {
			t.Fatalf("Generation failed: %v", err)
		}
	}

	afterVRAM, _ := getROCmFreeVRAM()
	slog.Info("VRAM after generation", "free_vram", format.HumanBytes2(int64(afterVRAM)))

	var tps float64
	if evalDuration > 0 {
		tps = float64(evalCount) / evalDuration.Seconds()
	}

	fmt.Printf("GPU_PERF: model=%s tps=%.2f duration=%v vram_used=%s\n",
		model, tps, time.Since(startTime), format.HumanBytes2(int64(beforeVRAM-afterVRAM)))

	if tps < 1.0 {
		t.Errorf("Very low tokens per second: %.2f", tps)
	}
}

func TestGPULayerOffloading(t *testing.T) {
	skipIfNoROCm(t)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	model := smol
	if err := PullIfMissing(ctx, client, model); err != nil {
		t.Fatal(err)
	}

	slog.Info("Loading model to check GPU offloading", "model", model)

	err := client.Generate(ctx, &api.GenerateRequest{Model: model}, func(resp api.GenerateResponse) error {
		return nil
	})
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	time.Sleep(500 * time.Millisecond)

	models, err := client.ListRunning(ctx)
	if err != nil {
		t.Fatalf("Failed to list running models: %v", err)
	}

	var found bool
	for _, m := range models.Models {
		if strings.HasPrefix(m.Name, model) {
			found = true
			gpuPercent := 0
			if m.Size > 0 {
				gpuPercent = int(math.Round(float64(m.SizeVRAM) / float64(m.Size) * 100))
			}
			slog.Info("Model offloading status",
				"model", m.Name,
				"size", format.HumanBytes2(m.Size),
				"vram", format.HumanBytes2(m.SizeVRAM),
				"gpu_percent", gpuPercent)

			if m.SizeVRAM == 0 {
				t.Log("Warning: Model is not using GPU VRAM")
			}

			fmt.Printf("GPU_OFFLOAD: model=%s size=%s vram=%s gpu_percent=%d%%\n",
				m.Name, format.HumanBytes2(m.Size), format.HumanBytes2(m.SizeVRAM), gpuPercent)
			break
		}
	}

	if !found {
		t.Errorf("Model %s not found in running models", model)
	}

	client.Generate(ctx, &api.GenerateRequest{Model: model, KeepAlive: &api.Duration{Duration: 0}}, func(r api.GenerateResponse) error { return nil })
}

func TestGPUModelFit(t *testing.T) {
	skipIfNoROCm(t)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	models := []string{
		"llama3.2:1b",
		"llama3.2:3b",
		"qwen3:0.6b",
	}

	for _, model := range models {
		t.Run(model, func(t *testing.T) {
			if err := PullIfMissing(ctx, client, model); err != nil {
				t.Fatalf("Failed to pull model %s: %v", model, err)
			}

			beforeVRAM, _ := getROCmFreeVRAM()

			err := client.Generate(ctx, &api.GenerateRequest{Model: model}, func(resp api.GenerateResponse) error {
				return nil
			})
			if err != nil {
				t.Fatalf("Failed to load model %s: %v", model, err)
			}

			time.Sleep(500 * time.Millisecond)

			afterVRAM, _ := getROCmFreeVRAM()
			vramUsed := beforeVRAM - afterVRAM

			models, err := client.ListRunning(ctx)
			if err != nil {
				t.Fatalf("Failed to list running models: %v", err)
			}

			for _, m := range models.Models {
				if strings.HasPrefix(m.Name, model) {
					fmt.Printf("GPU_FIT: model=%s size=%s vram_used=%s reported_vram=%s\n",
						model, format.HumanBytes2(m.Size), format.HumanBytes2(int64(vramUsed)), format.HumanBytes2(m.SizeVRAM))
					break
				}
			}

			client.Generate(ctx, &api.GenerateRequest{Model: model, KeepAlive: &api.Duration{Duration: 0}}, func(r api.GenerateResponse) error { return nil })
		})
	}
}

func TestGPUNumGPUOption(t *testing.T) {
	skipIfNoROCm(t)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	model := smol
	if err := PullIfMissing(ctx, client, model); err != nil {
		t.Fatal(err)
	}

	testCases := []int{-1, 0, 1, 10, 999}

	for _, numGPU := range testCases {
		t.Run(fmt.Sprintf("num_gpu_%d", numGPU), func(t *testing.T) {
			req := api.GenerateRequest{
				Model: model,
				Options: map[string]interface{}{
					"num_gpu": numGPU,
				},
			}

			err := client.Generate(ctx, &req, func(resp api.GenerateResponse) error {
				return nil
			})

			if numGPU >= 0 && numGPU < 999 {
				if err != nil {
					t.Logf("num_gpu=%d may not be fully supported: %v", numGPU, err)
				}
			}

			client.Generate(ctx, &api.GenerateRequest{Model: model, KeepAlive: &api.Duration{Duration: 0}}, func(r api.GenerateResponse) error { return nil })
		})
	}
}

func TestGPUMultiBatch(t *testing.T) {
	skipIfNoROCm(t)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	model := smol
	if err := PullIfMissing(ctx, client, model); err != nil {
		t.Fatal(err)
	}

	err := client.Generate(ctx, &api.GenerateRequest{Model: model}, func(resp api.GenerateResponse) error {
		return nil
	})
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	prompts := []string{
		"What is 2+2?",
		"What is the capital of France?",
		"Write a haiku.",
	}

	results := make(chan float64, len(prompts))

	for i, prompt := range prompts {
		go func(idx int, p string) {
			req := api.GenerateRequest{
				Model:  model,
				Prompt: p,
				Stream: &stream,
				Options: map[string]interface{}{
					"num_predict": 20,
				},
			}

			start := time.Now()
			var evalCount int
			var evalDuration time.Duration

			client.Generate(ctx, &req, func(resp api.GenerateResponse) error {
				if resp.Done {
					evalCount = resp.EvalCount
					evalDuration = resp.EvalDuration
				}
				return nil
			})

			var tps float64
			if evalDuration > 0 {
				tps = float64(evalCount) / evalDuration.Seconds()
			}
			slog.Info("Batch request completed", "idx", idx, "tps", tps, "duration", time.Since(start))
			results <- tps
		}(i, prompt)
	}

	var totalTps float64
	for range prompts {
		tps := <-results
		totalTps += tps
	}

	avgTps := totalTps / float64(len(prompts))
	slog.Info("Multi-batch test completed", "avg_tps", avgTps)

	client.Generate(ctx, &api.GenerateRequest{Model: model, KeepAlive: &api.Duration{Duration: 0}}, func(r api.GenerateResponse) error { return nil })
}

func TestGPUFlashAttention(t *testing.T) {
	skipIfNoROCm(t)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	model := smol
	if err := PullIfMissing(ctx, client, model); err != nil {
		t.Fatal(err)
	}

	flashAttentionModes := []string{"0", "1"}

	for _, mode := range flashAttentionModes {
		t.Run(fmt.Sprintf("flash_attn_%s", mode), func(t *testing.T) {
			t.Setenv("OLLAMA_FLASH_ATTENTION", mode)

			req := api.GenerateRequest{
				Model:  model,
				Prompt: "Hello, world!",
				Stream: &stream,
				Options: map[string]interface{}{
					"num_ctx":     4096,
					"num_predict": 10,
				},
			}

			var evalDuration time.Duration
			var evalCount int

			err := client.Generate(ctx, &req, func(resp api.GenerateResponse) error {
				if resp.Done {
					evalDuration = resp.EvalDuration
					evalCount = resp.EvalCount
				}
				return nil
			})
			if err != nil {
				t.Fatalf("Generation failed: %v", err)
			}

			var tps float64
			if evalDuration > 0 {
				tps = float64(evalCount) / evalDuration.Seconds()
			}

			slog.Info("Flash attention test", "mode", mode, "tps", tps)
		})

		client.Generate(ctx, &api.GenerateRequest{Model: model, KeepAlive: &api.Duration{Duration: 0}}, func(r api.GenerateResponse) error { return nil })
	}
}

func TestLargeModelSharding(t *testing.T) {
	skipIfNoROCm(t)

	vram, err := getROCmVRAM()
	if err != nil {
		t.Skip("Could not determine VRAM size")
	}

	minVRAM := uint64(16 * format.GibiByte)
	if vram < minVRAM {
		t.Skipf("This test requires at least %s VRAM, have %s", format.HumanBytes2(int64(minVRAM)), format.HumanBytes2(int64(vram)))
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	models := []struct {
		name        string
		requireVRAM uint64
	}{
		{"llama3.1:8b", 6 * format.GibiByte},
		{"qwen2.5-coder:7b", 6 * format.GibiByte},
	}

	for _, tc := range models {
		t.Run(tc.name, func(t *testing.T) {
			if vram < tc.requireVRAM {
				t.Skipf("Insufficient VRAM for %s (need %s, have %s)",
					tc.name, format.HumanBytes2(int64(tc.requireVRAM)), format.HumanBytes2(int64(vram)))
			}

			if err := PullIfMissing(ctx, client, tc.name); err != nil {
				t.Fatalf("Failed to pull model: %v", err)
			}

			beforeVRAM, _ := getROCmFreeVRAM()

			req := api.GenerateRequest{
				Model:  tc.name,
				Prompt: "Hello!",
				Stream: &stream,
				Options: map[string]interface{}{
					"num_predict": 20,
				},
			}

			var evalDuration time.Duration
			var evalCount int

			err := client.Generate(ctx, &req, func(resp api.GenerateResponse) error {
				if resp.Done {
					evalDuration = resp.EvalDuration
					evalCount = resp.EvalCount
				}
				return nil
			})
			if err != nil {
				t.Fatalf("Generation failed: %v", err)
			}

			afterVRAM, _ := getROCmFreeVRAM()
			vramUsed := beforeVRAM - afterVRAM

			var tps float64
			if evalDuration > 0 {
				tps = float64(evalCount) / evalDuration.Seconds()
			}

			fmt.Printf("LARGE_MODEL_SHARD: model=%s vram_used=%s tps=%.2f\n",
				tc.name, format.HumanBytes2(int64(vramUsed)), tps)

			client.Generate(ctx, &api.GenerateRequest{Model: tc.name, KeepAlive: &api.Duration{Duration: 0}}, func(r api.GenerateResponse) error { return nil })
		})
	}
}

func TestNVMeModelLoading(t *testing.T) {
	nvmePath := "/nvme3/ollama-models"
	if _, err := os.Stat(nvmePath); os.IsNotExist(err) {
		t.Skip("NVMe model path not found")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	models, err := client.List(ctx)
	if err != nil {
		t.Fatalf("Failed to list models: %v", err)
	}

	slog.Info("Models available", "count", len(models.Models))
	for _, m := range models.Models {
		fmt.Printf("MODEL_LIST: name=%s size=%s\n", m.Name, format.HumanBytes2(m.Size))
	}
}
