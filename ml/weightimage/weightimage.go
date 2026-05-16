package weightimage

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
)

type ImageFormat uint8

const (
	FormatF32 ImageFormat = iota
	FormatRGB
	FormatRGBA
	FormatBC4
	FormatBC7
	FormatDCT
)

type WeightImage struct {
	Width  int
	Height int
	Format ImageFormat
	Data   []float32
}

func WeightsToImage(tensor []float32, rows, cols int) *WeightImage {
	size := nextPow2(int(math.Sqrt(float64(rows * cols))))
	if size < 8 {
		size = 8
	}

	total := size * size
	data := make([]float32, total)

	copy(data, tensor)

	normalize := func(v float32) float32 {
		return (v + 1.0) / 2.0
	}

	if len(tensor) < total {
		for i := len(tensor); i < total; i++ {
			data[i] = normalize(tensor[i%len(tensor)])
		}
	}

	return &WeightImage{
		Width:  size,
		Height: size,
		Format: FormatRGB,
		Data:   data,
	}
}

func (w *WeightImage) ToRGBA() *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w.Width, w.Height))
	for y := 0; y < w.Height; y++ {
		for x := 0; x < w.Width; x++ {
			idx := y*w.Width + x
			if idx >= len(w.Data) {
				continue
			}
			val := uint8(math.Max(0, math.Min(255, float64(w.Data[idx])*255)))
			img.SetRGBA(x, y, color.RGBA{val, val, val, 255})
		}
	}
	return img
}

func (w *WeightImage) ToPNG() ([]byte, error) {
	img := w.ToRGBA()
	var buf bytes.Buffer
	err := png.Encode(&buf, img)
	return buf.Bytes(), err
}

func (w *WeightImage) String() string {
	return fmt.Sprintf("WeightImage{%dx%d, format=%v, data_len=%d}",
		w.Width, w.Height, w.Format, len(w.Data))
}

func nextPow2(n int) int {
	if n <= 0 {
		return 1
	}
	n--
	n |= n >> 1
	n |= n >> 2
	n |= n >> 4
	n |= n >> 8
	n |= n >> 16
	return n + 1
}

type LayerWeights struct {
	Name     string
	Shape    [2]int
	Data     []float32
	Image    *WeightImage
	Offset   uint64
	Size     uint64
	DType    string
}

func NewLayerWeights(name string, data []float32, rows, cols int) *LayerWeights {
	lw := &LayerWeights{
		Name:  name,
		Shape: [2]int{rows, cols},
		Data:  data,
	}
	lw.Image = WeightsToImage(data, rows, cols)
	return lw
}

func (lw *LayerWeights) Elements() int {
	return lw.Shape[0] * lw.Shape[1]
}

func (lw *LayerWeights) AspectRatio() float64 {
	return float64(lw.Shape[0]) / float64(lw.Shape[1])
}

func WeightsFromTensor(tensor []float32, shape []uint64) (*WeightImage, error) {
	if len(shape) < 2 {
		return nil, fmt.Errorf("tensor must have at least 2 dimensions, got %d", len(shape))
	}

	rows := int(shape[len(shape)-2])
	cols := int(shape[len(shape)-1])

	return WeightsToImage(tensor, rows, cols), nil
}

func (w *WeightImage) Stats() (min, max, mean, stddev float64) {
	if len(w.Data) == 0 {
		return 0, 0, 0, 0
	}

	min = float64(w.Data[0])
	max = float64(w.Data[0])
	var sum float64

	for _, v := range w.Data {
		fv := float64(v)
		if fv < min {
			min = fv
		}
		if fv > max {
			max = fv
		}
		sum += fv
	}

	mean = sum / float64(len(w.Data))

	var variance float64
	for _, v := range w.Data {
		diff := float64(v) - mean
		variance += diff * diff
	}
	stddev = math.Sqrt(variance / float64(len(w.Data)))

	return min, max, mean, stddev
}

func (w *WeightImage) HeatmapColors() []color.RGBA {
	colors := make([]color.RGBA, len(w.Data))

	min, max, _, _ := w.Stats()
	range_ := max - min
	if range_ == 0 {
		range_ = 1
	}

	for i, v := range w.Data {
		normalized := (float64(v) - min) / range_

		var r, g, b uint8
		if normalized < 0.5 {
			t := normalized * 2
			r = uint8(255 * (1 - t))
			g = uint8(255 * t)
			b = 0
		} else {
			t := (normalized - 0.5) * 2
			r = uint8(255 * t)
			g = uint8(255 * (1 - t))
			b = 0
		}

		colors[i] = color.RGBA{r, g, b, 255}
	}

	return colors
}

func PNGFromRGBA(img *image.RGBA) ([]byte, error) {
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func (w *WeightImage) ToHeatmapPNG() ([]byte, error) {
	colors := w.HeatmapColors()
	rgbaImg := image.NewRGBA(image.Rect(0, 0, w.Width, w.Height))

	for i, c := range colors {
		x := i % w.Width
		y := i / w.Width
		if y < rgbaImg.Bounds().Dy() && x < rgbaImg.Bounds().Dx() {
			rgbaImg.SetRGBA(x, y, c)
		}
	}

	return PNGFromRGBA(rgbaImg)
}