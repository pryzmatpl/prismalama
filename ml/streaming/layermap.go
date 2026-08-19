package streaming

import (
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

// LayerInfo describes one transformer block (or the output layer) in a GGUF model,
// with enough metadata to read its tensors from disk independently.
type LayerInfo struct {
	Index       int    // 0..N-1 for blk.0..blk.N-1; N for the output layer
	Name        string // "blk.0", "blk.1", ..., "output"
	ByteSize    uint64 // sum of all tensor sizes in this layer
	TensorCount int

	// Tensors in this layer: name → offset from tensor data base, size.
	Tensors []TensorRef
}

// TensorRef is a single tensor's location in the GGUF file.
type TensorRef struct {
	Name   string
	Offset uint64 // byte offset from the tensor data base in the GGUF file
	Size   uint64 // byte size of this tensor's data
}

// LayerMap is the parsed view of a GGUF model's transformer blocks, ordered by layer index.
type LayerMap struct {
	Layers           []LayerInfo
	TensorDataBase   uint64 // absolute file offset where tensor payloads start
	TotalWeightBytes uint64
	BlockCount       int // number of transformer blocks (excludes output layer)
}

// BuildLayerMap parses a decoded GGML model into a LayerMap. The GGML model must have been
// decoded with sufficient metadata (KV + tensor info). The returned map covers all blk.N
// layers and a synthetic "output" layer for output_norm, output, token_embd, etc.
func BuildLayerMap(g *ggml.GGML) (*LayerMap, error) {
	if g == nil {
		return nil, fmt.Errorf("nil GGML model")
	}

	tensors := g.Tensors()
	if tensors.Items() == nil || len(tensors.Items()) == 0 {
		return nil, fmt.Errorf("model has no tensors")
	}

	grouped := tensors.GroupLayers()
	blockCount := int(g.KV().BlockCount())

	lm := &LayerMap{
		TensorDataBase: tensors.Offset,
		BlockCount:     blockCount,
	}

	for i := 0; i < blockCount; i++ {
		key := fmt.Sprintf("blk.%d", i)
		layer, ok := grouped[key]
		if !ok {
			return nil, fmt.Errorf("missing layer %s in GGUF tensors", key)
		}
		li := buildLayerInfo(i, key, layer)
		lm.Layers = append(lm.Layers, li)
		lm.TotalWeightBytes += li.ByteSize
	}

	outputLI := buildOutputLayer(blockCount, grouped)
	if outputLI.TensorCount > 0 {
		lm.Layers = append(lm.Layers, outputLI)
		lm.TotalWeightBytes += outputLI.ByteSize
	}

	return lm, nil
}

func buildLayerInfo(index int, name string, layer ggml.Layer) LayerInfo {
	li := LayerInfo{Index: index, Name: name}
	for _, t := range layer {
		sz := t.Size()
		ref := TensorRef{
			Name:   t.Name,
			Offset: t.Offset,
			Size:   sz,
		}
		li.Tensors = append(li.Tensors, ref)
		li.ByteSize += sz
		li.TensorCount++
	}
	sort.Slice(li.Tensors, func(i, j int) bool { return li.Tensors[i].Offset < li.Tensors[j].Offset })
	return li
}

var outputKeys = []string{"output_norm", "output", "token_embd", "rope_freqs"}

func buildOutputLayer(blockCount int, grouped map[string]ggml.Layer) LayerInfo {
	li := LayerInfo{Index: blockCount, Name: "output"}
	for _, key := range outputKeys {
		layer, ok := grouped[key]
		if !ok {
			continue
		}
		for _, t := range layer {
			sz := t.Size()
			ref := TensorRef{
				Name:   t.Name,
				Offset: t.Offset,
				Size:   sz,
			}
			li.Tensors = append(li.Tensors, ref)
			li.ByteSize += sz
			li.TensorCount++
		}
	}
	sort.Slice(li.Tensors, func(i, j int) bool { return li.Tensors[i].Offset < li.Tensors[j].Offset })
	return li
}

// LayerByteRange returns the [start, end) absolute file offsets spanning all tensors in a layer.
// Useful for madvise or direct I/O alignment. Returns (0, 0) for empty layers.
func (lm *LayerMap) LayerByteRange(layerIndex int) (start, end uint64) {
	if layerIndex < 0 || layerIndex >= len(lm.Layers) {
		return 0, 0
	}
	li := lm.Layers[layerIndex]
	if li.TensorCount == 0 {
		return 0, 0
	}
	first := li.Tensors[0]
	last := li.Tensors[len(li.Tensors)-1]
	return lm.TensorDataBase + first.Offset, lm.TensorDataBase + last.Offset + last.Size
}

// FitsInBudget returns how many layers (starting from layer 0) fit within budgetBytes.
func (lm *LayerMap) FitsInBudget(budgetBytes uint64) int {
	var used uint64
	for i, li := range lm.Layers {
		if used+li.ByteSize > budgetBytes {
			return i
		}
		used += li.ByteSize
	}
	return len(lm.Layers)
}

// String returns a human-readable summary.
func (lm *LayerMap) String() string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "LayerMap: %d blocks + output, total %.2f GiB, tensor data @ 0x%x\n",
		lm.BlockCount, float64(lm.TotalWeightBytes)/(1024*1024*1024), lm.TensorDataBase)
	for _, li := range lm.Layers {
		fmt.Fprintf(&sb, "  %s: %d tensors, %.2f MiB\n", li.Name, li.TensorCount, float64(li.ByteSize)/(1024*1024))
	}
	return sb.String()
}

// ReadLayerTensors reads all tensor data for a layer from r (typically an *os.File opened on the GGUF).
// Returns a map of tensor name → raw bytes. The reader must support ReadAt (e.g. *os.File).
func (lm *LayerMap) ReadLayerTensors(r io.ReaderAt, layerIndex int) (map[string][]byte, error) {
	if layerIndex < 0 || layerIndex >= len(lm.Layers) {
		return nil, fmt.Errorf("layer index %d out of range [0, %d)", layerIndex, len(lm.Layers))
	}
	li := lm.Layers[layerIndex]
	result := make(map[string][]byte, li.TensorCount)
	for _, ref := range li.Tensors {
		buf := make([]byte, ref.Size)
		absOff := int64(lm.TensorDataBase + ref.Offset)
		if _, err := r.ReadAt(buf, absOff); err != nil {
			return nil, fmt.Errorf("read tensor %s at offset %d: %w", ref.Name, absOff, err)
		}
		result[ref.Name] = buf
	}
	return result, nil
}
