package llm

import (
	"fmt"
	"log/slog"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
)

// HIP/ROCm keeps a chunk of VRAM for the runtime, command queues, and
// fragmentation on top of GGML's graph estimate. Measured idle+tiny-model
// overhead on the RX 7900 XTX is ~2.5–3 GiB; keep a floor so a 36 GB GGUF
// cannot 100%-offload into 24 GiB and then OOM at hipMalloc.
const (
	gpuAllocSafety = 1 * format.GibiByte
	gpuGraphFloor  = 1 * format.GibiByte
	maxLoadShrink  = 12
)

func layerWeightBytes(f *ggml.GGML) []uint64 {
	if f == nil {
		return nil
	}
	blocks := int(f.KV().BlockCount())
	sizes := make([]uint64, blocks+1)
	grouped := f.Tensors().GroupLayers()
	for i := 0; i < blocks; i++ {
		if layer, ok := grouped[fmt.Sprintf("blk.%d", i)]; ok {
			sizes[i] = layer.Size()
		}
	}
	if out, ok := grouped["output"]; ok {
		sizes[blocks] += out.Size()
	}
	if on, ok := grouped["output_norm"]; ok {
		sizes[blocks] += on.Size()
	}
	if sizes[blocks] == 0 {
		if te, ok := grouped["token_embd"]; ok {
			sizes[blocks] = te.Size()
		}
	}
	return sizes
}

func layerWeightsKnown(weights []uint64) bool {
	for _, w := range weights {
		if w > 0 {
			return true
		}
	}
	return false
}

func gpuBudgetBytes(gpu ml.DeviceInfo, graphBytes uint64) uint64 {
	free := gpu.FreeMemory
	if free == 0 {
		free = gpu.TotalMemory
	}
	if gpu.TotalMemory > 0 && free > gpu.TotalMemory {
		free = gpu.TotalMemory
	}
	graph := graphBytes
	if graph < gpuGraphFloor {
		graph = gpuGraphFloor
	}
	overhead := gpu.MinimumMemory() + gpuAllocSafety + graph
	if free <= overhead {
		return 0
	}
	return free - overhead
}

// fitLayersFromEnd assigns a contiguous tail of layers to GPU (llama.cpp -ngl
// convention). layerWeights is repeating blocks followed by the output layer.
// kv is per repeating layer; the output index has no KV. maxLayers < 0 means
// no count cap (auto); 0 means CPU-only.
func fitLayersFromEnd(layerWeights, kv []uint64, budget uint64, maxLayers int) []int {
	n := len(layerWeights)
	if n == 0 || budget == 0 || maxLayers == 0 {
		return nil
	}
	capN := n
	if maxLayers > 0 && maxLayers < n {
		capN = maxLayers
	}
	var picked []int
	var used uint64
	for i := n - 1; i >= 0; i-- {
		if len(picked) >= capN {
			break
		}
		extra := layerWeights[i]
		if i < len(kv) {
			extra += kv[i]
		}
		if extra > budget || used+extra > budget {
			break
		}
		used += extra
		picked = append(picked, i)
	}
	slices.Reverse(picked)
	return picked
}

func engineGraphSize(f *ggml.GGML, opts api.Options, numParallel int) (kv []uint64, graph uint64) {
	if f == nil {
		return nil, 0
	}
	ctx := opts.NumCtx
	if ctx < 1 {
		ctx = 2048
	}
	batch := opts.NumBatch
	if batch < 1 {
		batch = min(512, ctx)
	} else {
		batch = min(batch, ctx)
	}
	if numParallel < 1 {
		numParallel = 1
	}
	kvct := envconfig.KvCacheType()
	if kvct == "" {
		kvct = "f16"
	}
	var partial, full uint64
	kv, partial, full = f.GraphSize(uint64(ctx), uint64(batch), numParallel, kvct, flashAttentionFromEnv())
	return kv, max(partial, full)
}

// gpuLayersForEngine packs a VRAM-fitting tail of layers onto the first GPU.
// NumGPU 0 is CPU-only. NumGPU < 0 (auto) fills the budget. NumGPU > 0 caps
// the count (still packed from the end). Without GGUF sizes, auto does not
// 100%-offload — that is what OOM'd qwen35-uncensored on 24 GiB.
func gpuLayersForEngine(gpus []ml.DeviceInfo, numGPU int, f *ggml.GGML, opts api.Options, numParallel int) ml.GPULayersList {
	if numGPU == 0 || len(gpus) == 0 {
		return nil
	}
	if f == nil {
		slog.Warn("ollama-engine: no GGUF metadata for GPU layout; leaving weights on CPU")
		return nil
	}
	weights := layerWeightBytes(f)
	if numGPU < 0 && !layerWeightsKnown(weights) {
		slog.Warn("ollama-engine: GGUF layer sizes unavailable; leaving weights on CPU")
		return nil
	}
	kv, graph := engineGraphSize(f, opts, numParallel)
	budget := gpuBudgetBytes(gpus[0], graph)
	layers := fitLayersFromEnd(weights, kv, budget, numGPU)
	slog.Info("ollama-engine GPU layout",
		"gpu_layers", len(layers),
		"total_layers", len(weights),
		"budget_bytes", budget,
		"graph_bytes", graph,
		"free_vram", gpus[0].FreeMemory,
		"num_gpu", numGPU,
	)
	if len(layers) == 0 {
		return nil
	}
	return ml.GPULayersList{{DeviceID: gpus[0].DeviceID, Layers: layers}}
}

func isInsufficientMemory(msg string) bool {
	return strings.Contains(strings.ToLower(msg), "insufficient memory")
}

func dropLeadingGPULayer(list ml.GPULayersList) ml.GPULayersList {
	if len(list) == 0 || len(list[0].Layers) == 0 {
		return nil
	}
	layers := list[0].Layers
	if len(layers) == 1 {
		return nil
	}
	out := append([]int(nil), layers[1:]...)
	return ml.GPULayersList{{DeviceID: list[0].DeviceID, Layers: out}}
}

// shrinkGPULayers refits the current GPU assignment using actual padded
// sizes from an ErrNoMem response. Always drops at least one layer so a
// retry cannot repeat the failing layout.
func shrinkGPULayers(cur ml.GPULayersList, mem ml.BackendMemory, gpu ml.DeviceInfo) ml.GPULayersList {
	if len(cur) == 0 || len(cur[0].Layers) == 0 {
		return nil
	}
	layers := cur[0].Layers
	var weights, cache []uint64
	graph := uint64(0)
	if len(mem.GPUs) > 0 {
		weights = mem.GPUs[0].Weights
		cache = mem.GPUs[0].Cache
		graph = mem.GPUs[0].Graph
	}
	budget := gpuBudgetBytes(gpu, graph)
	var keep []int
	var used uint64
	for i := len(layers) - 1; i >= 0; i-- {
		idx := layers[i]
		extra := uint64(0)
		if idx >= 0 && idx < len(weights) {
			extra += weights[idx]
		}
		if idx >= 0 && idx < len(cache) {
			extra += cache[idx]
		}
		if extra == 0 {
			break
		}
		if extra > budget || used+extra > budget {
			break
		}
		used += extra
		keep = append(keep, idx)
	}
	slices.Reverse(keep)
	if len(keep) == 0 || len(keep) >= len(layers) {
		return dropLeadingGPULayer(cur)
	}
	return ml.GPULayersList{{DeviceID: cur[0].DeviceID, Layers: keep}}
}
