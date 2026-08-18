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

// isMoEExpertTensor reports GGUF routed-expert weights (ffn_{gate,up,down}_exps).
// Shared-expert tensors (*_shexp) are not experts and stay with attn/GDN.
func isMoEExpertTensor(name string) bool {
	return strings.Contains(name, "_exps")
}

func sumU64(xs []uint64) uint64 {
	var n uint64
	for _, x := range xs {
		n += x
	}
	return n
}

// moeExpertBytes splits repeating-layer weights into routed-expert tensors vs
// everything else (attn, GDN, router, shared experts). Output stays in experts[last]
// so the tail pack still prefers keeping the head on GPU.
func moeExpertBytes(f *ggml.GGML, totals []uint64) (experts []uint64, pinned uint64, hasMoE bool) {
	if f == nil || len(totals) == 0 {
		return totals, 0, false
	}
	experts = append([]uint64(nil), totals...)
	blocks := len(totals) - 1
	grouped := f.Tensors().GroupLayers()
	for i := 0; i < blocks; i++ {
		layer, ok := grouped[fmt.Sprintf("blk.%d", i)]
		if !ok {
			continue
		}
		var exp uint64
		for name, t := range layer {
			if t == nil {
				continue
			}
			if isMoEExpertTensor(name) || isMoEExpertTensor(t.Name) {
				exp += t.Size()
			}
		}
		if exp == 0 {
			continue
		}
		hasMoE = true
		experts[i] = exp
		if totals[i] > exp {
			pinned += totals[i] - exp
		}
	}
	return experts, pinned, hasMoE
}

// packGPULayers assigns GPU layers. Dense models keep a contiguous tail of
// full layers. MoE models pin attn/GDN of every layer on GPU and only pack
// routed-expert tensors into the remaining budget (plus KV for all layers,
// because those layers compute on GPU). When some experts stay on CPU, one
// extra expert-layer is reserved so a prefill MUL_MAT_ID op-offload cannot
// hipMalloc the full expert tensor on top of a packed VRAM working set.
func packGPULayers(totals, experts, kv []uint64, pinned uint64, hasMoE bool, budget uint64, maxLayers int) []int {
	if !hasMoE || pinned == 0 {
		return fitLayersFromEnd(totals, kv, budget, maxLayers)
	}
	allKV := sumU64(kv)
	if pinned >= budget || budget-pinned <= allKV {
		return fitLayersFromEnd(totals, kv, budget, maxLayers)
	}
	rest := budget - pinned - allKV
	full := fitLayersFromEnd(experts, nil, rest, maxLayers)
	if len(full) == len(experts) {
		return full
	}
	var maxExp uint64
	for i := 0; i < len(experts)-1; i++ {
		if experts[i] > maxExp {
			maxExp = experts[i]
		}
	}
	if maxExp == 0 || rest <= maxExp {
		return fitLayersFromEnd(totals, kv, budget, maxLayers)
	}
	return fitLayersFromEnd(experts, nil, rest-maxExp, maxLayers)
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
// MoE GGUFs pin attn/GDN of every layer and only count routed-expert
// tensors toward the tail pack.
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
	experts, pinned, moe := moeExpertBytes(f, weights)
	layers := packGPULayers(weights, experts, kv, pinned, moe, budget, numGPU)
	slog.Info("ollama-engine GPU layout",
		"gpu_layers", len(layers),
		"total_layers", len(weights),
		"budget_bytes", budget,
		"graph_bytes", graph,
		"free_vram", gpus[0].FreeMemory,
		"num_gpu", numGPU,
		"moe_pin_non_expert_bytes", pinned,
		"moe_split", moe,
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
