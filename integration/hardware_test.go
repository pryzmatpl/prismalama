//go:build integration && hardware

package integration

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/ml"
)

func TestHardwareROCmDetection(t *testing.T) {
	if _, err := exec.LookPath("rocm-smi"); err != nil {
		t.Skip("ROCm not available")
	}

	cmd := exec.Command("rocm-smi")
	output, err := cmd.Output()
	if err != nil {
		t.Fatalf("rocm-smi failed: %v", err)
	}

	slog.Info("ROCm System Info", "output", string(output))

	cmd = exec.Command("rocm-smi", "--showproductname")
	output, err = cmd.Output()
	if err != nil {
		t.Logf("Could not get product name: %v", err)
	} else {
		slog.Info("GPU Product Name", "output", string(output))
	}

	cmd = exec.Command("rocminfo")
	output, err = cmd.Output()
	if err != nil {
		t.Logf("rocminfo not available: %v", err)
	} else {
		lines := strings.Split(string(output), "\n")
		for _, line := range lines {
			if strings.Contains(line, "Name:") ||
				strings.Contains(line, "Market Name:") ||
				strings.Contains(line, "Compute Unit:") ||
				strings.Contains(line, "SIMD per CU:") ||
				strings.Contains(line, "Max Clock") ||
				strings.Contains(line, "Memory Size:") {
				slog.Info("ROCm Info", "line", strings.TrimSpace(line))
			}
		}
	}
}

func TestHardwareVRAMDetection(t *testing.T) {
	if _, err := exec.LookPath("rocm-smi"); err != nil {
		t.Skip("ROCm not available")
	}

	cmd := exec.Command("rocm-smi", "--showmeminfo", "vram")
	output, err := cmd.Output()
	if err != nil {
		t.Fatalf("rocm-smi --showmeminfo vram failed: %v", err)
	}

	slog.Info("VRAM Info", "output", string(output))

	lines := strings.Split(string(output), "\n")
	var totalVRAM, usedVRAM, freeVRAM uint64

	for _, line := range lines {
		fields := strings.Fields(line)
		for i, f := range fields {
			if f == "Total" && i+1 < len(fields) {
				val, err := strconv.ParseFloat(fields[i+1], 64)
				if err == nil {
					totalVRAM = uint64(val * float64(format.MebiByte))
				}
			}
			if f == "Used" && i+1 < len(fields) {
				val, err := strconv.ParseFloat(fields[i+1], 64)
				if err == nil {
					usedVRAM = uint64(val * float64(format.MebiByte))
				}
			}
			if f == "Free" && i+1 < len(fields) {
				val, err := strconv.ParseFloat(fields[i+1], 64)
				if err == nil {
					freeVRAM = uint64(val * float64(format.MebiByte))
				}
			}
		}
	}

	fmt.Printf("VRAM_DETECTION: total=%s used=%s free=%s\n",
		format.HumanBytes2(int64(totalVRAM)),
		format.HumanBytes2(int64(usedVRAM)),
		format.HumanBytes2(int64(freeVRAM)))

	if totalVRAM == 0 {
		t.Error("Could not detect total VRAM")
	}
}

func TestHardwareSystemMemory(t *testing.T) {
	cmd := exec.Command("free", "-b")
	output, err := cmd.Output()
	if err != nil {
		t.Fatalf("free command failed: %v", err)
	}

	lines := strings.Split(string(output), "\n")
	var totalMem, freeMem, availableMem uint64

	for _, line := range lines {
		if strings.HasPrefix(line, "Mem:") {
			fields := strings.Fields(line)
			if len(fields) >= 4 {
				fmt.Sscanf(fields[1], "%d", &totalMem)
				fmt.Sscanf(fields[3], "%d", &freeMem)
				if len(fields) >= 7 {
					fmt.Sscanf(fields[6], "%d", &availableMem)
				}
			}
		}
		if strings.HasPrefix(line, "Swap:") {
			fields := strings.Fields(line)
			if len(fields) >= 3 {
				var totalSwap, freeSwap uint64
				fmt.Sscanf(fields[1], "%d", &totalSwap)
				fmt.Sscanf(fields[3], "%d", &freeSwap)
				fmt.Printf("SWAP_DETECTION: total=%s free=%s\n",
					format.HumanBytes2(int64(totalSwap)),
					format.HumanBytes2(int64(freeSwap)))
			}
		}
	}

	fmt.Printf("MEMORY_DETECTION: total=%s free=%s available=%s\n",
		format.HumanBytes2(int64(totalMem)),
		format.HumanBytes2(int64(freeMem)),
		format.HumanBytes2(int64(availableMem)))

	if totalMem == 0 {
		t.Error("Could not detect total system memory")
	}
}

func TestHardwareCPUInfo(t *testing.T) {
	cmd := exec.Command("nproc")
	output, err := cmd.Output()
	if err != nil {
		t.Fatalf("nproc failed: %v", err)
	}

	cores := strings.TrimSpace(string(output))
	slog.Info("CPU Cores", "count", cores)

	cmd = exec.Command("lscpu")
	output, err = cmd.Output()
	if err != nil {
		t.Logf("lscpu not available: %v", err)
	} else {
		lines := strings.Split(string(output), "\n")
		for _, line := range lines {
			if strings.HasPrefix(line, "Model name:") ||
				strings.HasPrefix(line, "CPU(s):") ||
				strings.HasPrefix(line, "Thread(s) per core:") ||
				strings.HasPrefix(line, "Core(s) per socket:") ||
				strings.HasPrefix(line, "CPU MHz:") ||
				strings.HasPrefix(line, "CPU max MHz:") {
				slog.Info("CPU Info", "detail", strings.TrimSpace(line))
			}
		}
	}
}

func TestHardwareNVMePerformance(t *testing.T) {
	// Collect paths from env var (colon-separated), then fall back to common mount points.
	nvmePaths := []string{}
	if envPaths := os.Getenv("OLLAMA_TEST_NVME_PATHS"); envPaths != "" {
		for _, p := range strings.Split(envPaths, ":") {
			if p != "" {
				nvmePaths = append(nvmePaths, p)
			}
		}
	}
	if len(nvmePaths) == 0 {
		nvmePaths = []string{
			"/nvme3",
			"/run/media/piotro/CACHE",
			"/run/media/piotro/CACHE1",
		}
	}

	for _, path := range nvmePaths {
		if _, err := os.Stat(path); os.IsNotExist(err) {
			continue
		}

		t.Run(path, func(t *testing.T) {
			testFile := path + "/.write_test"
			defer os.Remove(testFile)

			data := make([]byte, 100*1024*1024)
			for i := range data {
				data[i] = byte(i % 256)
			}

			start := time.Now()
			err := os.WriteFile(testFile, data, 0644)
			writeDuration := time.Since(start)

			if err != nil {
				t.Fatalf("Write test failed: %v", err)
			}

			start = time.Now()
			_, err = os.ReadFile(testFile)
			readDuration := time.Since(start)

			if err != nil {
				t.Fatalf("Read test failed: %v", err)
			}

			writeSpeed := float64(len(data)) / writeDuration.Seconds() / 1024 / 1024
			readSpeed := float64(len(data)) / readDuration.Seconds() / 1024 / 1024

			fmt.Printf("NVME_PERF: path=%s write_mbps=%.2f read_mbps=%.2f\n",
				path, writeSpeed, readSpeed)

			if writeSpeed < 100 {
				t.Logf("Warning: Slow write speed: %.2f MB/s", writeSpeed)
			}
			if readSpeed < 100 {
				t.Logf("Warning: Slow read speed: %.2f MB/s", readSpeed)
			}
		})
	}
}

func TestHardwareOptimalSettings(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	var totalVRAM uint64
	if cmd := exec.Command("rocm-smi", "--showmeminfo", "vram"); cmd != nil {
		if output, err := cmd.Output(); err == nil {
			lines := strings.Split(string(output), "\n")
			for _, line := range lines {
				if strings.Contains(line, "Total") {
					fields := strings.Fields(line)
					for i, f := range fields {
						if f == "Total" && i+1 < len(fields) {
							val, _ := strconv.ParseFloat(fields[i+1], 64)
							totalVRAM = uint64(val * float64(format.MebiByte))
							break
						}
					}
				}
			}
		}
	}

	var systemMem uint64
	if cmd := exec.Command("free", "-b"); cmd != nil {
		if output, err := cmd.Output(); err == nil {
			lines := strings.Split(string(output), "\n")
			for _, line := range lines {
				if strings.HasPrefix(line, "Mem:") {
					fields := strings.Fields(line)
					if len(fields) >= 2 {
						fmt.Sscanf(fields[1], "%d", &systemMem)
					}
				}
			}
		}
	}

	cpuCores := 4
	if cmd := exec.Command("nproc"); cmd != nil {
		if output, err := cmd.Output(); err == nil {
			if val, err := strconv.Atoi(strings.TrimSpace(string(output))); err == nil {
				cpuCores = val
			}
		}
	}

	fmt.Printf("HARDWARE_SUMMARY:\n")
	fmt.Printf("  GPU VRAM: %s\n", format.HumanBytes2(int64(totalVRAM)))
	fmt.Printf("  System RAM: %s\n", format.HumanBytes2(int64(systemMem)))
	fmt.Printf("  CPU Cores: %d\n", cpuCores)

	recommendedNumGPU := -1
	if totalVRAM < 8*uint64(format.GibiByte) {
		recommendedNumGPU = 20
		fmt.Printf("  Recommended: Partial GPU offloading (num_gpu=20)\n")
	} else if totalVRAM >= 24*uint64(format.GibiByte) {
		recommendedNumGPU = 999
		fmt.Printf("  Recommended: Full GPU offloading (num_gpu=999)\n")
	} else {
		fmt.Printf("  Recommended: Auto GPU offloading (num_gpu=-1)\n")
	}

	recommendedCtx := 4096
	if systemMem >= 64*uint64(format.GibiByte) {
		recommendedCtx = 8192
	}
	if systemMem >= 128*uint64(format.GibiByte) {
		recommendedCtx = 16384
	}
	fmt.Printf("  Recommended context: %d\n", recommendedCtx)

	recommendedThreads := cpuCores / 2
	if recommendedThreads < 2 {
		recommendedThreads = 2
	}
	fmt.Printf("  Recommended threads: %d\n", recommendedThreads)

	fmt.Printf("RECOMMENDED_CONFIG: num_gpu=%d num_ctx=%d num_thread=%d\n",
		recommendedNumGPU, recommendedCtx, recommendedThreads)
}

func TestHardwareGPULayerAllocation(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	model := smol
	if err := PullIfMissing(ctx, client, model); err != nil {
		t.Fatal(err)
	}

	testCases := []struct {
		name   string
		numGPU int
	}{
		{"auto", -1},
		{"cpu_only", 0},
		{"minimal_gpu", 5},
		{"partial_gpu", 15},
		{"full_gpu", 999},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			req := api.GenerateRequest{
				Model: model,
				Options: map[string]interface{}{
					"num_gpu": tc.numGPU,
				},
			}

			err := client.Generate(ctx, &req, func(resp api.GenerateResponse) error {
				return nil
			})
			if err != nil {
				t.Fatalf("Failed to load model: %v", err)
			}

			time.Sleep(200 * time.Millisecond)

			models, err := client.ListRunning(ctx)
			if err != nil {
				t.Fatalf("Failed to list running models: %v", err)
			}

			for _, m := range models.Models {
				if strings.HasPrefix(m.Name, model) {
					var gpuPercent float64
					if m.Size > 0 {
						gpuPercent = float64(m.SizeVRAM) / float64(m.Size) * 100
					}

					fmt.Printf("LAYER_ALLOC: config=%s size=%s vram=%s gpu_percent=%.1f%%\n",
						tc.name,
						format.HumanBytes2(m.Size),
						format.HumanBytes2(m.SizeVRAM),
						gpuPercent)

					switch tc.name {
					case "cpu_only":
						if m.SizeVRAM > 0 {
							t.Logf("Note: CPU-only config still using %s VRAM", format.HumanBytes2(m.SizeVRAM))
						}
					case "full_gpu":
						if m.SizeVRAM < m.Size/2 {
							t.Logf("Warning: Full GPU config only using %s of %s",
								format.HumanBytes2(m.SizeVRAM), format.HumanBytes2(m.Size))
						}
					}
					break
				}
			}

			client.Generate(ctx, &api.GenerateRequest{Model: model, KeepAlive: &api.Duration{Duration: 0}}, func(r api.GenerateResponse) error { return nil })
		})
	}
}

func TestHardwareFlashAttentionSupport(t *testing.T) {
	devices := []ml.DeviceInfo{
		{DeviceID: ml.DeviceID{ID: "gpu0", Library: "ROCm"}},
		{DeviceID: ml.DeviceID{ID: "gpu1", Library: "CUDA", DriverMajor: 8}},
		{DeviceID: ml.DeviceID{ID: "gpu2", Library: "Metal"}},
		{DeviceID: ml.DeviceID{ID: "cpu0", Library: "cpu"}},
	}

	supported := ml.FlashAttentionSupported(devices)
	if !supported {
		t.Error("Expected flash attention to be supported on ROCm/CUDA/Metal/CPU")
	}

	unsupportedDevices := []ml.DeviceInfo{
		{DeviceID: ml.DeviceID{ID: "gpu0", Library: "CUDA", DriverMajor: 6}},
	}

	supported = ml.FlashAttentionSupported(unsupportedDevices)
	if supported {
		t.Error("Expected flash attention to NOT be supported on old CUDA")
	}

	fmt.Printf("FLASH_ATTENTION: supported=%v\n", supported)
}
