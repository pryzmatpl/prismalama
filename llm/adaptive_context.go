package llm

import (
	"fmt"
	"log/slog"
	"runtime"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
)

const adaptiveMinNumCtx = 512

// adaptNumCtxForAvailableMemory lowers opts.NumCtx (and syncs loadRequest) when the
// combined estimate of static weights + KV cache + compute graph for the requested
// context would exceed available system RAM. Keeps full requested context when the
// estimate fits. Disable with OLLAMA_ADAPTIVE_MEMORY=false.
func adaptNumCtxForAvailableMemory(
	opts *api.Options,
	loadRequest *LoadRequest,
	g *ggml.GGML,
	systemInfo ml.SystemInfo,
	gpus []ml.DeviceInfo,
	numParallel int,
	totalLayers int,
) {
	if !envconfig.AdaptiveMemory(true) {
		return
	}
	if opts == nil || loadRequest == nil || g == nil {
		return
	}
	if numParallel < 1 {
		numParallel = 1
	}

	weightBytes := estimateGGUFWeightBytes(g)
	if weightBytes == 0 {
		return
	}

	usable := adaptiveUsableBytes(systemInfo)
	if usable == 0 {
		return
	}

	mmapWeights := predictUseMmap(opts, systemInfo, gpus, weightBytes, totalLayers)

	requested := opts.NumCtx
	if requested < adaptiveMinNumCtx {
		return
	}

	trainCtx := int(g.KV().ContextLength())
	maxN := requested
	if trainCtx > 0 && maxN > trainCtx {
		maxN = trainCtx
	}

	kvct := loadRequest.KvCacheType
	if kvct == "" {
		kvct = "f16"
	}
	fa := loadRequest.FlashAttention

	// runtimeOverhead reserves memory for the Go runtime, runner process, and allocator
	// fragmentation that our GGUF-based estimate does not capture.
	runtimeOverhead := uint64(1024 * 1024 * 1024) // 1 GiB

	tryFit := func(n int) bool {
		if n < adaptiveMinNumCtx {
			return false
		}
		batch := min(opts.NumBatch, n)
		kv, part, full := g.GraphSize(uint64(n), uint64(batch), numParallel, kvct, fa)
		var kvSum uint64
		for _, x := range kv {
			kvSum += x
		}
		weightPart := weightBytes
		if mmapWeights {
			// Most weight bytes are file-backed mmap, not anonymous RAM.
			weightPart = weightBytes / 16
			if weightPart < 512*1024*1024 {
				weightPart = 512 * 1024 * 1024
			}
		}
		need := weightPart + kvSum + max(part, full) + runtimeOverhead
		return need <= usable
	}

	if tryFit(maxN) {
		return
	}

	lo, hi := adaptiveMinNumCtx, maxN
	var best int
	for lo <= hi {
		mid := (lo + hi) / 2
		if tryFit(mid) {
			best = mid
			lo = mid + 1
		} else {
			hi = mid - 1
		}
	}
	if best == 0 || best >= requested {
		return
	}

	slog.Info("adaptive memory: reduced num_ctx to fit available system memory",
		"num_ctx_requested", requested, "num_ctx_adapted", best,
		"usable_estimate", fmt.Sprintf("%.1f GiB", float64(usable)/(1024*1024*1024)),
		"model_weights_estimate", fmt.Sprintf("%.1f GiB", float64(weightBytes)/(1024*1024*1024)),
		"mmap_weights", mmapWeights)

	opts.NumCtx = best
	opts.NumBatch = min(opts.NumBatch, opts.NumCtx)
	loadRequest.KvSize = opts.NumCtx * numParallel
	loadRequest.BatchSize = min(loadRequest.BatchSize, opts.NumCtx)
}

func predictUseMmap(opts *api.Options, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, weightBytes uint64, totalLayers int) bool {
	if opts.UseMMap != nil {
		return *opts.UseMMap
	}
	// Mirrors llamaServer.Load mmap decision (server.go) using GGUF weight sum as size.
	for _, gpu := range gpus {
		if gpu.Library == "Metal" && opts.NumGPU > 0 && opts.NumGPU < totalLayers {
			return false
		}
	}
	linuxDisable := runtime.GOOS == "linux" && systemInfo.FreeMemory < weightBytes && opts.UseMMap == nil
	if envconfig.MmapAllowLowRamLinux() {
		linuxDisable = false
	}
	vulkanNoMmap := len(gpus) > 0 && gpus[0].Library == "Vulkan" && !envconfig.VulkanMmap(true)
	if (runtime.GOOS == "windows" && len(gpus) > 0 && gpus[0].Library == "CUDA") ||
		linuxDisable ||
		len(gpus) == 0 ||
		vulkanNoMmap {
		return false
	}
	return true
}

func adaptiveUsableBytes(info ml.SystemInfo) uint64 {
	b := info.FreeMemory
	b += info.FreeSwap / 4
	if b == 0 {
		return 0
	}
	reserve := b / 6
	if reserve < 256*1024*1024 {
		reserve = 256 * 1024 * 1024
	}
	if b > reserve {
		return b - reserve
	}
	return b / 2
}

func estimateGGUFWeightBytes(g *ggml.GGML) uint64 {
	layers := g.Tensors().GroupLayers()
	var sum uint64
	for i := range g.KV().BlockCount() {
		if blk, ok := layers[fmt.Sprintf("blk.%d", i)]; ok {
			sum += blk.Size()
		}
	}
	if layer, ok := layers["output_norm"]; ok {
		sum += layer.Size()
	}
	if layer, ok := layers["output"]; ok {
		sum += layer.Size()
	} else if layer, ok := layers["token_embd"]; ok {
		sum += layer.Size()
	}
	return sum
}
