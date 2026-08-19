package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml/weightimage"
)

type WeightImageHandler struct{}

func NewWeightImageHandler() *WeightImageHandler {
	return &WeightImageHandler{}
}

func (h *WeightImageHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	modelPath := r.URL.Query().Get("model")
	if modelPath == "" {
		http.Error(w, "model path required", http.StatusBadRequest)
		return
	}

	modelFile := filepath.Join(modelPath, "model.gguf")
	if _, err := os.Stat(modelFile); os.IsNotExist(err) {
		http.Error(w, "model.gguf not found", http.StatusNotFound)
		return
	}

	f, err := os.Open(modelFile)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer f.Close()

	meta, err := ggml.Decode(f, 1024)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to decode GGUF: %v", err), http.StatusInternalServerError)
		return
	}

	tensors := meta.Tensors().Items()
	layers := meta.Tensors().GroupLayers()

	response := gin.H{
		"model":        modelPath,
		"architecture": meta.KV().Architecture(),
		"block_count":  meta.KV().BlockCount(),
		"tensor_count": len(tensors),
		"layers":       h.formatLayers(layers),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *WeightImageHandler) formatLayers(layers map[string]ggml.Layer) []gin.H {
	result := make([]gin.H, 0, len(layers))
	for name, layer := range layers {
		entry := gin.H{"name": name}

		var totalSize uint64
		var tensorNames []string
		for tName, t := range layer {
			totalSize += t.Size()
			tensorNames = append(tensorNames, tName)
		}
		entry["tensor_count"] = len(layer)
		entry["total_bytes"] = totalSize
		entry["tensors"] = tensorNames

		result = append(result, entry)
	}
	return result
}

type WeightLayerImageHandler struct{}

func NewWeightLayerImageHandler() *WeightLayerImageHandler {
	return &WeightLayerImageHandler{}
}

func (h *WeightLayerImageHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	modelPath := r.URL.Query().Get("model")
	if modelPath == "" {
		http.Error(w, "model path required", http.StatusBadRequest)
		return
	}

	layerStr := r.URL.Query().Get("layer")
	layerIdx, err := strconv.Atoi(layerStr)
	if err != nil {
		http.Error(w, "layer must be a number", http.StatusBadRequest)
		return
	}

	modelFile := filepath.Join(modelPath, "model.gguf")
	f, err := os.Open(modelFile)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer f.Close()

	meta, err := ggml.Decode(f, -1)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to decode GGUF: %v", err), http.StatusInternalServerError)
		return
	}

	layers := meta.Tensors().GroupLayers()
	layerKey := fmt.Sprintf("blk.%d", layerIdx)
	layer, ok := layers[layerKey]
	if !ok {
		http.Error(w, fmt.Sprintf("layer %d not found", layerIdx), http.StatusNotFound)
		return
	}

	var bigTensor *ggml.Tensor
	for _, t := range layer {
		if bigTensor == nil || t.Size() > bigTensor.Size() {
			bigTensor = t
		}
	}

	if bigTensor == nil {
		http.Error(w, "no tensors in layer", http.StatusNotFound)
		return
	}

	if len(bigTensor.Shape) < 2 {
		http.Error(w, "tensor has fewer than 2 dimensions", http.StatusBadRequest)
		return
	}

	rows := int(bigTensor.Shape[len(bigTensor.Shape)-2])
	cols := int(bigTensor.Shape[len(bigTensor.Shape)-1])

	absOffset := int64(meta.Tensors().Offset + bigTensor.Offset)
	data := make([]byte, bigTensor.Size())
	if _, err := f.ReadAt(data, absOffset); err != nil {
		http.Error(w, fmt.Sprintf("failed to read tensor data: %v", err), http.StatusInternalServerError)
		return
	}

	floatData := make([]float32, rows*cols)
	elemSize := bigTensor.Size() / bigTensor.Elements()
	for i := 0; i < len(floatData) && i*int(elemSize) < len(data); i++ {
		switch elemSize {
		case 2:
			val := uint16(data[i*2]) | uint16(data[i*2+1])<<8
			floatData[i] = float32(int16(val)) / 32768.0
		case 4:
			val := uint32(data[i*4]) | uint32(data[i*4+1])<<8 | uint32(data[i*4+2])<<16 | uint32(data[i*4+3])<<24
			floatData[i] = float32(int32(val)) / 2147483648.0
		default:
			floatData[i] = float32(data[i]) / 127.5
		}
	}

	img := weightimage.WeightsToImage(floatData, rows, cols)

	format := r.URL.Query().Get("format")
	switch format {
	case "png":
		pngData, err := img.ToPNG()
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to encode PNG: %v", err), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "image/png")
		w.Write(pngData)
	case "heatmap":
		colors := img.HeatmapColors()
		rgbaImg := img.ToRGBA()
		for i, c := range colors {
			x := i % img.Width
			y := i / img.Width
			if y < rgbaImg.Bounds().Dy() && x < rgbaImg.Bounds().Dx() {
				rgbaImg.SetRGBA(x, y, c)
			}
		}
		pngData, _ := weightimage.PNGFromRGBA(rgbaImg)
		w.Header().Set("Content-Type", "image/png")
		w.Write(pngData)
	default:
		min, max, mean, stddev := img.Stats()
		response := gin.H{
			"layer":             layerIdx,
			"name":              bigTensor.Name,
			"shape":             []int{rows, cols},
			"image_width":       img.Width,
			"image_height":      img.Height,
			"tensor_bytes":      bigTensor.Size(),
			"image_bytes":       img.Width * img.Height * 4,
			"compression_ratio": float64(rows*cols*4) / float64(img.Width*img.Height*4),
			"stats": gin.H{
				"min":    min,
				"max":    max,
				"mean":   mean,
				"stddev": stddev,
			},
			"format": "rgb_f32_to_uint8",
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}
}

func (h *WeightLayerImageHandler) formatLayerInfo(layer map[string]*ggml.Tensor) gin.H {
	var totalSize uint64
	var tensorNames []string
	var biggestTensor string
	var biggestSize uint64

	for name, t := range layer {
		totalSize += t.Size()
		tensorNames = append(tensorNames, name)
		if t.Size() > biggestSize {
			biggestSize = t.Size()
			biggestTensor = name
		}
	}

	return gin.H{
		"tensor_count":   len(layer),
		"total_bytes":    totalSize,
		"biggest_tensor": biggestTensor,
		"biggest_bytes":  biggestSize,
		"tensors":        tensorNames,
	}
}

type WeightImageStatsHandler struct{}

func NewWeightImageStatsHandler() *WeightImageStatsHandler {
	return &WeightImageStatsHandler{}
}

func (h *WeightImageStatsHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	modelPath := r.URL.Query().Get("model")
	if modelPath == "" {
		http.Error(w, "model path required", http.StatusBadRequest)
		return
	}

	modelFile := filepath.Join(modelPath, "model.gguf")
	f, err := os.Open(modelFile)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer f.Close()

	meta, err := ggml.Decode(f, 1024)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to decode GGUF: %v", err), http.StatusInternalServerError)
		return
	}

	tensors := meta.Tensors().Items()
	layers := meta.Tensors().GroupLayers()

	stats := gin.H{
		"architecture":  meta.KV().Architecture(),
		"block_count":   meta.KV().BlockCount(),
		"total_tensors": len(tensors),
	}

	var totalWeightBytes uint64
	var layerStats []gin.H

	for name, layer := range layers {
		var layerSize uint64
		var bigSize uint64
		var bigName string
		var shapes [][]int

		for tName, t := range layer {
			layerSize += t.Size()
			shapes = append(shapes, []int{int(t.Shape[len(t.Shape)-2]), int(t.Shape[len(t.Shape)-1])})
			if t.Size() > bigSize {
				bigSize = t.Size()
				bigName = tName
			}
		}

		totalWeightBytes += layerSize

		layerStats = append(layerStats, gin.H{
			"name":          name,
			"size_bytes":    layerSize,
			"biggest":       bigName,
			"biggest_bytes": bigSize,
			"shapes":        shapes,
		})
	}

	stats["total_weight_bytes"] = totalWeightBytes
	stats["layers"] = layerStats

	estCompressed := float64(totalWeightBytes) / 4.0
	stats["estimated_bc7_compressed_bytes"] = estCompressed
	stats["estimated_compression_ratio"] = float64(totalWeightBytes) / estCompressed

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}
