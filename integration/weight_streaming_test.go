//go:build integration && weight_streaming

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
	// Model path can be overridden via OLLAMA_TEST_MODEL_PATH env var.
	// Defaults are example paths; tests skip gracefully when not found.
	weightStreamingModelPath = os.Getenv("OLLAMA_TEST_MODEL_PATH")
	weightStreamingGGUFGlob = "*-00001-of-*.gguf"
)

func skipIfNoWeightStreamingModel(t *testing.T) {
	info, err := os.Stat(weightStreamingModelPath)
	if os.IsNotExist(err) {
		t.Skip("Weight streaming model not found at " + weightStreamingModelPath)
	}
	if !info.IsDir() {
		t.Skip("Weight streaming path is not a directory")
	}
}

func TestWeightStreamingGGUFDetection(t *testing.T) {
	skipIfNoWeightStreamingModel(t)

	entries, err := os.ReadDir(weightStreamingModelPath)
	if err != nil {
		t.Fatalf("Failed to read model directory: %v", err)
	}

	var ggufFiles []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if strings.HasSuffix(name, ".gguf") {
			info, _ := e.Info()
			ggufFiles = append(ggufFiles, fmt.Sprintf("%s (%s)", name, format.HumanBytes2(info.Size())))
		}
	}

	slog.Info("GGUF files detected", "files", ggufFiles, "count", len(ggufFiles))

	if len(ggufFiles) == 0 {
		t.Fatal("No GGUF files found in model directory")
	}
}

func TestWeightStreamingMultiPartDetection(t *testing.T) {
	skipIfNoWeightStreamingModel(t)

	entries, err := os.ReadDir(weightStreamingModelPath)
	if err != nil {
		t.Fatalf("Failed to read model directory: %v", err)
	}

	var splitFiles []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if strings.Contains(name, "-00001-of-") {
			splitFiles = append(splitFiles, name)
		}
	}

	slog.Info("Multi-part GGUF files", "files", splitFiles)

	if len(splitFiles) == 0 {
		t.Log("No multi-part GGUF files found - this is expected for single-file models")
	}
}

func TestWeightStreamingModelSize(t *testing.T) {
	skipIfNoWeightStreamingModel(t)

	entries, err := os.ReadDir(weightStreamingModelPath)
	if err != nil {
		t.Fatalf("Failed to read model directory: %v", err)
	}

	var totalSize int64
	var fileCount int
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if strings.HasSuffix(name, ".gguf") {
			info, err := e.Info()
			if err != nil {
				continue
			}
			totalSize += info.Size()
			fileCount++
		}
	}

	fmt.Printf("MODEL_SIZE: total=%s files=%d\n", format.HumanBytes2(totalSize), fileCount)

	vramSize := int64(24 * format.GibiByte)
	if totalSize > vramSize {
		requiredOffload := float64(totalSize-vramSize) / float64(totalSize) * 100
		fmt.Printf("MODEL_MEMORY: VRAM=%s Model=%s OffloadNeeded=%.1f%%\n",
			format.HumanBytes2(vramSize),
			format.HumanBytes2(totalSize),
			requiredOffload)
	}
}

func TestWeightStreamingInference(t *testing.T) {
	skipIfNoWeightStreamingModel(t)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	req := api.GenerateRequest{
		Model:  weightStreamingModelPath,
		Prompt: "Say 'test' if you can understand this.",
		Options: map[string]interface{}{
			"temperature": 0.5,
			"num_predict": 10,
		},
	}

	slog.Info("Starting weight streaming inference test")
	startTime := time.Now()

	var response string
	var evalCount int
	var evalDuration time.Duration
	stream := true

	err := client.Generate(ctx, &req, func(resp api.GenerateResponse) error {
		response += resp.Response
		if resp.Done {
			evalCount = resp.EvalCount
			evalDuration = resp.EvalDuration
		}
		return nil
	})

	if err != nil {
		if strings.Contains(err.Error(), "memory") || strings.Contains(err.Error(), "OOM") {
			t.Skipf("Insufficient memory: %v", err)
		}
		t.Fatalf("Generation failed: %v", err)
	}

	duration := time.Since(startTime)
	fmt.Printf("WEIGHT_STREAMING_RESULT: response=%q duration=%v eval_count=%d eval_tps=%.2f\n",
		response, duration, evalCount,
		float64(evalCount)/evalDuration.Seconds())
}

func TestMoEExpertCount(t *testing.T) {
	skipIfNoWeightStreamingModel(t)

	entries, err := os.ReadDir(weightStreamingModelPath)
	if err != nil {
		t.Fatalf("Failed to read model directory: %v", err)
	}

	var hasExperts bool
	var expertCount int
	for _, e := range entries {
		name := e.Name()
		if strings.Contains(name, "expert") || strings.Contains(name, "moe") {
			hasExperts = true
		}
		if strings.HasPrefix(name, "blk.0.expert") {
			expertCount++
		}
	}

	if hasExperts || expertCount > 0 {
		fmt.Printf("MOE_DETECTED: experts=%d\n", expertCount)
		t.Logf("Model appears to be MoE (Mixture of Experts)")
	}
}

func TestWeightStreamingPartialOffload(t *testing.T) {
	skipIfNoWeightStreamingModel(t)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	testCases := []struct {
		name   string
		numGPU int
		numCtx int
	}{
		{"full_cpu", 0, 4096},
		{"partial_gpu_4", 4, 4096},
		{"partial_gpu_8", 8, 4096},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			req := api.GenerateRequest{
				Model:  weightStreamingModelPath,
				Prompt: "Count from 1 to 3.",
				Options: map[string]interface{}{
					"num_gpu":     tc.numGPU,
					"num_ctx":     tc.numCtx,
					"num_predict": 10,
				},
			}

			startTime := time.Now()
			var response string

			err := client.Generate(ctx, &req, func(resp api.GenerateResponse) error {
				response += resp.Response
				return nil
			})

			duration := time.Since(startTime)

			if err != nil {
				slog.Wlog("Generation failed", "test", tc.name, "error", err)
				return
			}

			fmt.Printf("PARTIAL_OFFLOAD_%s: duration=%v response=%q\n",
				strings.ToUpper(tc.name), duration, response)
		})
	}
}
