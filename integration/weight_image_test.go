package integration

import (
	"testing"

	"github.com/ollama/ollama/ml/weightimage"
)

func TestWeightImagePackage(t *testing.T) {
	t.Run("WeightsToImage", func(t *testing.T) {
		weights := make([]float32, 4096)
		for i := range weights {
			weights[i] = float32(i%256)/127.5 - 1.0
		}

		img := weightimage.WeightsToImage(weights, 64, 64)
		if img.Width != img.Height {
			t.Errorf("expected square image")
		}
	})

	t.Run("BC4Compression", func(t *testing.T) {
		weights := make([]float32, 256)
		for i := range weights {
			weights[i] = float32(i) / 255.0
		}

		cw, err := weightimage.CompressBC4(weights, 16, 16)
		if err != nil {
			t.Fatalf("compression failed: %v", err)
		}

		ratio := cw.CompressionRatio()
		if ratio < 1.0 {
			t.Errorf("BC4 should compress, got ratio %v", ratio)
		}
	})

	t.Run("DCTCompression", func(t *testing.T) {
		weights := make([]float32, 256)
		for i := range weights {
			weights[i] = float32(i%64) / 63.0
		}

		dct, err := weightimage.CompressDCT(weights, 16, 16, 0.8)
		if err != nil {
			t.Fatalf("DCT failed: %v", err)
		}

		recovered := dct.Decompress()
		if len(recovered) != len(weights) {
			t.Errorf("DCT recovery size mismatch")
		}
	})
}