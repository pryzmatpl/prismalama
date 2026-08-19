package weightimage

import (
	"math"
	"testing"
)

func TestWeightsToImage(t *testing.T) {
	weights := make([]float32, 4096)
	for i := range weights {
		weights[i] = float32(i%256)/127.5 - 1.0
	}

	img := WeightsToImage(weights, 64, 64)

	if img.Width != img.Height {
		t.Errorf("expected square image, got %dx%d", img.Width, img.Height)
	}

	if img.Format != FormatRGB {
		t.Errorf("expected FormatRGB, got %v", img.Format)
	}

	if len(img.Data) != img.Width*img.Height {
		t.Errorf("expected %d data points, got %d", img.Width*img.Height, len(img.Data))
	}
}

func TestWeightImageStats(t *testing.T) {
	weights := []float32{-1.0, -0.5, 0.0, 0.5, 1.0}
	img := WeightsToImage(weights, 1, 5)

	min, max, _, stddev := img.Stats()

	if math.Abs(min+1.0) > 0.01 {
		t.Errorf("expected min -1.0, got %v", min)
	}
	if math.Abs(max-1.0) > 0.01 {
		t.Errorf("expected max 1.0, got %v", max)
	}
	if stddev < 0.1 {
		t.Errorf("expected positive stddev, got %v", stddev)
	}
}

func TestCompressBC4(t *testing.T) {
	weights := make([]float32, 64)
	for i := range weights {
		weights[i] = float32(i) / 64.0
	}

	cw, err := CompressBC4(weights, 8, 8)
	if err != nil {
		t.Fatalf("compression failed: %v", err)
	}

	if cw.Format != CompressionBC4 {
		t.Errorf("expected CompressionBC4, got %v", cw.Format)
	}

	if cw.Width != 8 || cw.Height != 8 {
		t.Errorf("expected 8x8, got %dx%d", cw.Width, cw.Height)
	}

	ratio := cw.CompressionRatio()
	if ratio < 1.0 {
		t.Errorf("expected ratio >= 1.0 (compression), got %v", ratio)
	}

	t.Logf("BC4 compression ratio: %.2fx", ratio)
}

func TestDecompressBC4(t *testing.T) {
	weights := make([]float32, 64)
	for i := range weights {
		weights[i] = float32(i%64) / 63.0
	}

	cw, err := CompressBC4(weights, 8, 8)
	if err != nil {
		t.Fatalf("compression failed: %v", err)
	}

	decompressed, err := DecompressBC4(cw)
	if err != nil {
		t.Fatalf("decompression failed: %v", err)
	}

	if len(decompressed) != len(weights) {
		t.Errorf("expected %d values, got %d", len(weights), len(decompressed))
	}

	maxDiff := 0.0
	for i := range weights {
		diff := math.Abs(float64(decompressed[i] - weights[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	if maxDiff > 0.5 {
		t.Errorf("decompression error too high: %v (BC4 is lossy)", maxDiff)
	}
}

func TestCompressDCT(t *testing.T) {
	weights := make([]float32, 256)
	for i := range weights {
		weights[i] = float32(math.Sin(float64(i) * 0.1))
	}

	dct, err := CompressDCT(weights, 16, 16, 0.8)
	if err != nil {
		t.Fatalf("DCT compression failed: %v", err)
	}

	if len(dct.DCTData) != 256 {
		t.Errorf("expected 256 DCT coefficients, got %d", len(dct.DCTData))
	}

	recovered := dct.Decompress()
	if len(recovered) != len(weights) {
		t.Errorf("expected %d recovered values, got %d", len(weights), len(recovered))
	}
}

func TestTextureLayout(t *testing.T) {
	layout := ReshapeToTextureLayout(1024, 1024)

	if layout.Width != layout.Height {
		t.Errorf("expected square layout, got %dx%d", layout.Width, layout.Height)
	}

	if layout.Channels != 4 {
		t.Errorf("expected 4 channels, got %d", layout.Channels)
	}

	if layout.ElementSize() != 4 {
		t.Errorf("expected element size 4, got %d", layout.ElementSize())
	}
}

func TestBilinearSampler(t *testing.T) {
	layout := &TextureLayout{
		Width:     4,
		Height:    4,
		Channels:  4,
		RowStride: 16,
	}

	data := make([]byte, 4*4*4)
	for i := 0; i < len(data); i += 4 {
		data[i] = 0
		data[i+1] = 0
		data[i+2] = 0
		data[i+3] = 255
	}

	sampler := NewBilinearSampler(layout, data)

	_, _, _, a := sampler.Sample(0.5, 0.5)
	if a < 0.9 || a > 1.0 {
		t.Errorf("expected alpha ~1.0, got %v", a)
	}

	r, _, _, _ := sampler.Sample(0.5, 0.5)
	if r < -0.1 || r > 0.1 {
		t.Errorf("expected r ~0.0 (black pixel), got %v", r)
	}
}

func TestCompressWeights(t *testing.T) {
	weights := make([]float32, 256)
	for i := range weights {
		weights[i] = float32(i) / 255.0
	}

	result, err := CompressWeights(weights, 16, 16, CompressionBC4)
	if err != nil {
		t.Fatalf("CompressWeights failed: %v", err)
	}

	cw, ok := result.(*CompressedWeights)
	if !ok {
		t.Fatalf("expected *CompressedWeights, got %T", result)
	}

	if cw.CompressionRatio() < 1.0 {
		t.Errorf("BC4 should compress: ratio %v", cw.CompressionRatio())
	}
}

func BenchmarkWeightsToImage(b *testing.B) {
	weights := make([]float32, 1024*1024)
	for i := range weights {
		weights[i] = float32(i%256) / 127.5
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		WeightsToImage(weights, 1024, 1024)
	}
}

func BenchmarkCompressBC4(b *testing.B) {
	weights := make([]float32, 1024*1024)
	for i := range weights {
		weights[i] = float32(i%256) / 255.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CompressBC4(weights, 1024, 1024)
	}
}
