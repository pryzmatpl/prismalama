package weightimage

import (
	"fmt"
	"math"
)

type CompressionFormat uint8

const (
	CompressionF32 CompressionFormat = iota
	CompressionBC4
	CompressionBC7
	CompressionDCT
)

type CompressedWeights struct {
	Format  CompressionFormat
	Width   int
	Height  int
	BlocksX int
	BlocksY int
	Data    []byte
}

type BC4Block struct {
	Endpoints [8]byte
	Indices   [16]byte
}

func CompressBC4(weights []float32, w, h int) (*CompressedWeights, error) {
	blockW := 4
	blockH := 4
	blockSize := 8

	blocksX := (w + blockW - 1) / blockW
	blocksY := (h + blockH - 1) / blockH

	data := make([]byte, blocksX*blocksY*blockSize)

	for by := 0; by < blocksY; by++ {
		for bx := 0; bx < blocksX; bx++ {
			blockData := make([]float32, 16)
			blockIdx := 0

			for ky := 0; ky < blockH; ky++ {
				for kx := 0; kx < blockW; kx++ {
					px := bx*blockW + kx
					py := by*blockH + ky

					idx := py*w + px
					if idx < len(weights) {
						blockData[blockIdx] = weights[idx]
					} else {
						blockData[blockIdx] = 0
					}
					blockIdx++
				}
			}

			block := compressBC4Block(blockData)
			copy(data[(by*blocksX+bx)*blockSize:], block[:])
		}
	}

	return &CompressedWeights{
		Format:  CompressionBC4,
		Width:   w,
		Height:  h,
		BlocksX: blocksX,
		BlocksY: blocksY,
		Data:    data,
	}, nil
}

func compressBC4Block(pixels []float32) [8]byte {
	minVal := float32(1)
	maxVal := float32(0)

	for _, p := range pixels {
		pNorm := (p + 1.0) / 2.0
		if pNorm < minVal {
			minVal = pNorm
		}
		if pNorm > maxVal {
			maxVal = pNorm
		}
	}

	if maxVal-minVal < 1.0/256.0 {
		maxVal = minVal + 1.0/256.0
	}

	quantMin := uint8(minVal * 255)
	quantMax := uint8(maxVal * 255)

	var block [8]byte
	block[0] = quantMin
	block[1] = quantMax

	range_ := maxVal - minVal
	if range_ == 0 {
		range_ = 1
	}

	indices := [16]byte{}
	for i, p := range pixels {
		pNorm := (float64(p) + 1.0) / 2.0
		t := (pNorm - float64(minVal)) / float64(range_)
		t = math.Max(0, math.Min(1, t))

		idx := byte(t * 7.0)
		if idx > 7 {
			idx = 7
		}

		if i%2 == 0 {
			indices[i/2] = idx
		} else {
			indices[i/2] |= idx << 4
		}
	}

	copy(block[2:], indices[:])

	return block
}

func DecompressBC4(cw *CompressedWeights) ([]float32, error) {
	if cw.Format != CompressionBC4 {
		return nil, fmt.Errorf("not a BC4 compressed block")
	}

	weights := make([]float32, cw.Width*cw.Height)
	blockW := 4
	blockH := 4

	for by := 0; by < cw.BlocksY; by++ {
		for bx := 0; bx < cw.BlocksX; bx++ {
			blockOffset := (by*cw.BlocksX + bx) * 8
			block := cw.Data[blockOffset : blockOffset+8]

			minVal := float32(block[0]) / 255.0
			maxVal := float32(block[1]) / 255.0

			indices := [16]byte{}
			copy(indices[:], block[2:8])

			blockIdx := 0
			for ky := 0; ky < blockH; ky++ {
				for kx := 0; kx < blockW; kx++ {
					px := bx*blockW + kx
					py := by*blockH + ky

					if px >= cw.Width || py >= cw.Height {
						continue
					}

					idx := indices[blockIdx/2]
					if blockIdx%2 == 0 {
						idx = idx & 0x0F
					} else {
						idx = idx >> 4
					}

					t := float32(idx) / 7.0
					val := minVal + t*(maxVal-minVal)
					weights[py*cw.Width+px] = val*2.0 - 1.0
					blockIdx++
				}
			}
		}
	}

	return weights, nil
}

func (cw *CompressedWeights) CompressionRatio() float64 {
	originalSize := cw.Width * cw.Height * 4
	compressedSize := len(cw.Data)
	if compressedSize == 0 {
		return 0
	}
	return float64(originalSize) / float64(compressedSize)
}

func (cw *CompressedWeights) String() string {
	return fmt.Sprintf("CompressedWeights{%dx%d, format=%v, ratio=%.2fx, blocks=%dx%d}",
		cw.Width, cw.Height, cw.Format, cw.CompressionRatio(), cw.BlocksX, cw.BlocksY)
}

type DCTWeights struct {
	Width    int
	Height   int
	DCTData  []float32
	Quant    []float32
	Frequency []float32
}

func CompressDCT(weights []float32, w, h int, quality float64) (*DCTWeights, error) {
	if quality <= 0 || quality > 1 {
		quality = 0.8
	}

	dct := &DCTWeights{
		Width:   w,
		Height:  h,
		DCTData: make([]float32, w*h),
		Quant:   make([]float32, w*h),
	}

	dctData := dct.dct2d(weights, w, h)

	dct.createQuantizationTable(quality)

	quantized := make([]float32, w*h)
	for i := 0; i < w*h && i < len(dctData) && i < len(dct.Quant); i++ {
		quantized[i] = dctData[i] / dct.Quant[i]
	}
	dct.DCTData = quantized

	var energy float64
	for _, v := range dct.DCTData {
		energy += float64(v * v)
	}
	dct.Frequency = []float32{float32(energy), float32(energy * 0.1), float32(energy * 0.01)}

	return dct, nil
}

func (d *DCTWeights) dct2d(input []float32, w, h int) []float32 {
	result := make([]float32, w*h)

	for u := 0; u < w; u++ {
		for v := 0; v < h; v++ {
			var sum float32
			for x := 0; x < w; x++ {
				for y := 0; y < h; y++ {
					idx := y*w + x
					if idx >= len(input) {
						continue
					}
					val := input[idx]

					cosX := math.Cos(float64(2*x+1) * float64(u) * math.Pi / float64(2*w))
					cosY := math.Cos(float64(2*y+1) * float64(v) * math.Pi / float64(2*h))

					sum += val * float32(cosX*cosY)
				}
			}

			alphaU := float32(math.Sqrt(2.0/float64(w)))
			alphaV := float32(math.Sqrt(2.0/float64(h)))

			if u == 0 {
				alphaU = float32(1.0 / float64(w))
			}
			if v == 0 {
				alphaV = float32(1.0 / float64(h))
			}

			result[v*w+u] = sum * alphaU * alphaV
		}
	}

	return result
}

func (d *DCTWeights) createQuantizationTable(quality float64) {
	for i := 0; i < d.Width*d.Height; i++ {
		x := i % d.Width
		y := i / d.Width

		freq := math.Sqrt(float64(x*x + y*y))
		maxFreq := math.Sqrt(float64(d.Width*d.Width + d.Height*d.Height))

		normFreq := freq / maxFreq

		baseQ := 1.0 + normFreq*quality*9.0

		d.Quant[i] = float32(baseQ)
	}
}

func (d *DCTWeights) Decompress() []float32 {
	dequantized := make([]float32, d.Width*d.Height)
	for i := 0; i < d.Width*d.Height && i < len(d.DCTData) && i < len(d.Quant); i++ {
		dequantized[i] = d.DCTData[i] * d.Quant[i]
	}

	return d.idct2d(dequantized, d.Width, d.Height)
}

func (d *DCTWeights) idct2d(input []float32, w, h int) []float32 {
	result := make([]float32, w*h)

	for x := 0; x < w; x++ {
		for y := 0; y < h; y++ {
			var sum float32
			for u := 0; u < w; u++ {
				for v := 0; v < h; v++ {
					idx := v*w + u
					if idx >= len(input) {
						continue
					}
					val := input[idx]

					cosU := math.Cos(float64(2*x+1) * float64(u) * math.Pi / float64(2*w))
					cosV := math.Cos(float64(2*y+1) * float64(v) * math.Pi / float64(2*h))

					sum += val * float32(cosU*cosV)
				}
			}

			alphaU := float32(math.Sqrt(2.0/float64(w)))
			alphaV := float32(math.Sqrt(2.0/float64(h)))

			if x == 0 {
				alphaU = float32(1.0 / float64(w))
			}
			if y == 0 {
				alphaV = float32(1.0 / float64(h))
			}

			result[y*w+x] = sum * alphaU * alphaV
		}
	}

	return result
}

func CompressWeights(weights []float32, w, h int, format CompressionFormat) (interface{}, error) {
	switch format {
	case CompressionBC4:
		return CompressBC4(weights, w, h)
	case CompressionDCT:
		return CompressDCT(weights, w, h, 0.8)
	default:
		return nil, fmt.Errorf("unsupported compression format: %v", format)
	}
}