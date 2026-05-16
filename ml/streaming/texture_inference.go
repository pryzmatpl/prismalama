package streaming

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/weightimage"
)

type CompressedInferenceStreamer struct {
	*InferenceStreamer
	texBackend  *weightimage.VulkanWeightTextureBackend
	compression weightimage.CompressionFormat
}

func NewCompressedInferenceStreamer(
	backend ml.Backend,
	modelPath string,
	lm *LayerMap,
	compression weightimage.CompressionFormat,
) (*CompressedInferenceStreamer, error) {
	texBackend := weightimage.NewVulkanWeightTextureBackend("stream", "compressed")

	return &CompressedInferenceStreamer{
		InferenceStreamer: NewInferenceStreamer(backend, modelPath, lm),
		texBackend:        texBackend,
		compression:       compression,
	}, nil
}

func (cis *CompressedInferenceStreamer) PrepareForInference(ctx context.Context) error {
	return cis.InferenceStreamer.PrepareForInference(ctx)
}

func (cis *CompressedInferenceStreamer) OnBlockDone(blockIdx int) bool {
	is := cis.InferenceStreamer
	is.mu.Lock()
	defer is.mu.Unlock()

	nextBlock := blockIdx + 1
	if nextBlock < is.totalBlocks {
		start := time.Now()
		if err := cis.loadBlockCompressed(nextBlock); err != nil {
			slog.Error("compressed streaming inference: failed to load block",
				"block", nextBlock, "error", err)
			return false
		}
		slog.Debug("compressed streaming inference: loaded block",
			"block", nextBlock, "duration", time.Since(start), "format", cis.compression)
	}

	if blockIdx >= 0 && blockIdx < is.totalBlocks {
		is.evictBlock(blockIdx)
		slog.Debug("compressed streaming inference: evicted block", "block", blockIdx)
	}

	is.currentBlock = nextBlock
	return true
}

func (cis *CompressedInferenceStreamer) loadBlockCompressed(blockIdx int) error {
	is := cis.InferenceStreamer
	if blockIdx < 0 || blockIdx >= len(is.lm.Layers) {
		return fmt.Errorf("block %d out of range", blockIdx)
	}
	if is.loadedWeights[blockIdx] {
		return nil
	}
	if is.backend == nil {
		is.loadedWeights[blockIdx] = true
		return nil
	}

	li := is.lm.Layers[blockIdx]

	bigTensorName := ""
	var bigTensorSize uint64
	var bigTensorRows, bigTensorCols int
	var bigTensorData []byte

	for _, ref := range li.Tensors {
		t := is.backend.Get(ref.Name)
		if t == nil {
			continue
		}

		shape := t.Shape()
		if len(shape) >= 2 {
			rows := shape[len(shape)-2]
			cols := shape[len(shape)-1]
			size := uint64(rows) * uint64(cols)
			if size > bigTensorSize {
				bigTensorSize = size
				bigTensorName = ref.Name
				bigTensorRows = rows
				bigTensorCols = cols

				buf := make([]byte, ref.Size)
				absOff := int64(is.lm.TensorDataBase + ref.Offset)
				if _, err := is.file.ReadAt(buf, absOff); err == nil {
					bigTensorData = buf
				}
			}
		}

		buf := make([]byte, ref.Size)
		absOff := int64(is.lm.TensorDataBase + ref.Offset)
		if _, err := is.file.ReadAt(buf, absOff); err != nil {
			return fmt.Errorf("read tensor %s: %w", ref.Name, err)
		}

		if err := safeFromBytes(t, buf); err != nil {
			slog.Warn("compressed streaming inference: tensor size mismatch, skipping",
				"name", ref.Name, "gguf_size", ref.Size, "error", err)
		}
	}

	if bigTensorData != nil && bigTensorRows > 0 && bigTensorCols > 0 {
		floatData := make([]float32, bigTensorRows*bigTensorCols)
		for i := 0; i < len(floatData) && i*4 < len(bigTensorData); i++ {
			floatData[i] = float32(bigTensorData[i*4])/255.0*2.0 - 1.0
		}

		switch cis.compression {
		case weightimage.CompressionBC4:
			cw, err := weightimage.CompressBC4(floatData, bigTensorCols, bigTensorRows)
			if err != nil {
				slog.Warn("compressed streaming: BC4 compression failed", "error", err)
			} else {
				cis.texBackend.LoadCompressedBlock(cw)
				slog.Debug("compressed streaming: BC4 compressed",
					"name", bigTensorName,
					"original", bigTensorRows*bigTensorCols*4,
					"compressed", len(cw.Data),
					"ratio", cw.CompressionRatio())
			}
		case weightimage.CompressionDCT:
			dct, err := weightimage.CompressDCT(floatData, bigTensorCols, bigTensorRows, 0.8)
			if err != nil {
				slog.Warn("compressed streaming: DCT compression failed", "error", err)
			} else {
				slog.Debug("compressed streaming: DCT compressed",
					"name", bigTensorName,
					"original", bigTensorRows*bigTensorCols*4,
					"dct_coeffs", len(dct.DCTData),
					"frequency_energy", dct.Frequency)
			}
		}
	}

	is.loadedWeights[blockIdx] = true
	return nil
}

func (cis *CompressedInferenceStreamer) Close() {
	cis.texBackend.Close()
	cis.InferenceStreamer.Close()
}

type TextureBackedLayer struct {
	LayerIndex int
	Name       string
	Texture    *weightimage.WeightTexture
	Compressed *weightimage.CompressedWeights
	Layout     *weightimage.TextureLayout
}

type TextureStreamingCache struct {
	mu           sync.Mutex
	layers       map[int]*TextureBackedLayer
	backend      *weightimage.VulkanWeightTextureBackend
	maxResident  int
	currentIndex int
}

func NewTextureStreamingCache(maxResident int) *TextureStreamingCache {
	return &TextureStreamingCache{
		layers:      make(map[int]*TextureBackedLayer),
		backend:     weightimage.NewVulkanWeightTextureBackend("cache", "texture"),
		maxResident: maxResident,
	}
}

func (tsc *TextureStreamingCache) GetLayer(idx int) (*TextureBackedLayer, bool) {
	tsc.mu.Lock()
	defer tsc.mu.Unlock()
	layer, ok := tsc.layers[idx]
	return layer, ok
}

func (tsc *TextureStreamingCache) LoadLayer(idx int, weights []float32, rows, cols int) error {
	tsc.mu.Lock()
	defer tsc.mu.Unlock()

	if tsc.currentIndex >= tsc.maxResident {
		evictIdx := tsc.currentIndex - tsc.maxResident
		delete(tsc.layers, evictIdx)
	}

	img := weightimage.WeightsToImage(weights, rows, cols)
	layout := weightimage.ReshapeToTextureLayout(rows, cols)

	wt := &weightimage.WeightTexture{
		Name:   fmt.Sprintf("layer_%d", idx),
		Width:  img.Width,
		Height: img.Height,
		Format: weightimage.FormatRGB,
	}

	cw, err := weightimage.CompressBC4(weights, cols, rows)
	if err == nil {
		tsc.backend.LoadCompressedBlock(cw)
	}

	tsc.layers[idx] = &TextureBackedLayer{
		LayerIndex: idx,
		Name:       wt.Name,
		Texture:    wt,
		Compressed: cw,
		Layout:     layout,
	}

	tsc.currentIndex++
	return nil
}

func (tsc *TextureStreamingCache) Close() error {
	tsc.mu.Lock()
	defer tsc.mu.Unlock()
	tsc.layers = nil
	return tsc.backend.Close()
}

func (tsc *TextureStreamingCache) Stats() (loaded int, max int) {
	tsc.mu.Lock()
	defer tsc.mu.Unlock()
	return len(tsc.layers), tsc.maxResident
}

func (tsc *TextureStreamingCache) CompressionRatio() float64 {
	tsc.mu.Lock()
	defer tsc.mu.Unlock()

	if len(tsc.layers) == 0 {
		return 0
	}

	var totalRatio float64
	count := 0
	for _, layer := range tsc.layers {
		if layer.Compressed != nil {
			totalRatio += layer.Compressed.CompressionRatio()
			count++
		}
	}

	if count == 0 {
		return 0
	}
	return totalRatio / float64(count)
}