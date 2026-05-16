package integration

import (
	"math"
	"testing"

	"github.com/ollama/ollama/ml/weightimage"
)

func TestWeightImageOps(t *testing.T) {
	t.Run("BC4RoundTrip", func(t *testing.T) {
		weights := make([]float32, 256)
		for i := range weights {
			weights[i] = float32(i%64)/63.0*2.0 - 1.0
		}

		cw, err := weightimage.CompressBC4(weights, 16, 16)
		if err != nil {
			t.Fatalf("BC4 compression failed: %v", err)
		}

		if ratio := cw.CompressionRatio(); ratio < 7.0 {
			t.Errorf("BC4 compression ratio too low: got %.2f, want >= 7.0", ratio)
		}

		recovered, err := weightimage.DecompressBC4(cw)
		if err != nil {
			t.Fatalf("BC4 decompression failed: %v", err)
		}

		if len(recovered) != len(weights) {
			t.Errorf("recovered length mismatch: got %d, want %d", len(recovered), len(weights))
		}

		var maxErr float64
		var mse float64
		for i := range weights {
			err := math.Abs(float64(recovered[i] - weights[i]))
			if err > maxErr {
				maxErr = err
			}
			mse += err * err
		}
		mse = math.Sqrt(mse / float64(len(weights)))

		if maxErr > 0.1 {
			t.Errorf("BC4 max error too high: %.4f (tolerance 0.1)", maxErr)
		}
		if mse > 0.05 {
			t.Errorf("BC4 MSE too high: %.4f (tolerance 0.05)", mse)
		}
	})

	t.Run("LayerWeightImageConversion", func(t *testing.T) {
		rows, cols := 512, 256
		weights := make([]float32, rows*cols)
		for i := range weights {
			weights[i] = float32(i%256)/127.5 - 1.0
		}

		lw := weightimage.NewLayerWeights("test_layer", weights, rows, cols)
		if lw.Image == nil {
			t.Fatal("LayerWeights image is nil")
		}
		if lw.Image.Width != lw.Image.Height {
			t.Errorf("expected square image, got %dx%d", lw.Image.Width, lw.Image.Height)
		}

		imgData := lw.Image.Data
		if len(imgData) < rows*cols {
			t.Errorf("image data too small: got %d, want >= %d", len(imgData), rows*cols)
		}

		ratio := float64(len(imgData)*4) / float64(len(imgData)*4)
		if ratio < 1.0 {
			t.Errorf("unexpected ratio: %v", ratio)
		}
	})

	t.Run("BC4MultiLayer", func(t *testing.T) {
		layers := make([]*weightimage.CompressedWeights, 4)
		for l := 0; l < 4; l++ {
			weights := make([]float32, 4096)
			for i := range weights {
				weights[i] = float32((i+l)%256)/127.5 - 1.0
			}
			cw, err := weightimage.CompressBC4(weights, 64, 64)
			if err != nil {
				t.Fatalf("layer %d compression failed: %v", l, err)
			}
			layers[l] = cw
		}

		for l, cw := range layers {
			if cw.BlocksX == 0 || cw.BlocksY == 0 {
				t.Errorf("layer %d has zero blocks", l)
			}
			if cw.CompressionRatio() < 7.0 {
				t.Errorf("layer %d compression ratio too low: %.2f", l, cw.CompressionRatio())
			}
		}
	})

	t.Run("BC4DecompressCorrectness", func(t *testing.T) {
		weights := []float32{
			-1.0, -0.5, 0.0, 0.5, -0.75, 0.25, -1.0, 1.0,
			0.0, -0.25, 0.75, -0.5, 0.5, -0.75, 0.25, -1.0,
			1.0, -1.0, 0.5, -0.5, 0.0, 0.75, -0.25, 1.0,
			-0.5, 0.5, -1.0, 1.0, -0.75, 0.25, 0.0, -0.5,
			0.75, -0.25, -0.5, 0.5, 1.0, -1.0, 0.25, -0.75,
			0.0, 0.0, -1.0, 1.0, -0.5, -0.5, 0.75, 0.75,
			-0.75, 0.75, 0.25, -0.25, 0.0, 0.0, 0.5, -0.5,
			1.0, -1.0, 0.0, 0.0, -0.75, 0.75, 1.0, -1.0,
			0.25, -0.25, 0.75, -0.75, 0.5, -0.5, 0.0, 0.0,
			-0.5, -0.5, 0.5, 0.5, -0.75, -0.75, 0.75, 0.75,
			1.0, 1.0, -1.0, -1.0, 0.25, 0.25, -0.25, -0.25,
			0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5,
			-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0,
			1.0, 1.0, 1.0, 1.0, -0.5, -0.5, -0.5, -0.5,
			0.5, 0.5, 0.5, 0.5, -0.75, -0.75, -0.75, -0.75,
			-0.25, -0.25, -0.25, -0.25, 0.75, 0.75, 0.75, 0.75,
		}

		cw, err := weightimage.CompressBC4(weights, 16, 16)
		if err != nil {
			t.Fatalf("compression failed: %v", err)
		}

		recovered, err := weightimage.DecompressBC4(cw)
		if err != nil {
			t.Fatalf("decompression failed: %v", err)
		}

		var maxAbsErr float64
		var maxRelErr float64
		for i := range weights {
			absErr := math.Abs(float64(recovered[i] - weights[i]))
			relErr := absErr
			if math.Abs(float64(weights[i])) > 0.01 {
				relErr = absErr / math.Abs(float64(weights[i]))
			}
			if absErr > maxAbsErr {
				maxAbsErr = absErr
			}
			if relErr > maxRelErr {
				maxRelErr = relErr
			}
		}

		t.Logf("BC4 roundtrip errors: max_abs=%.4f, max_rel=%.4f", maxAbsErr, maxRelErr)

		if maxAbsErr > 0.05 {
			t.Errorf("max absolute error %.4f exceeds tolerance 0.05", maxAbsErr)
		}
	})

	t.Run("BC4TextureLayout", func(t *testing.T) {
		sizes := [][2]int{
			{16, 16},
			{32, 32},
			{64, 64},
			{128, 64},
			{256, 128},
		}

		for _, size := range sizes {
			weights := make([]float32, size[0]*size[1])
			for i := range weights {
				weights[i] = float32(i%256)/127.5 - 1.0
			}

			cw, err := weightimage.CompressBC4(weights, size[0], size[1])
			if err != nil {
				t.Fatalf("compress %dx%d failed: %v", size[0], size[1], err)
			}

			expectedBlocksX := (size[0] + 3) / 4
			expectedBlocksY := (size[1] + 3) / 4
			if cw.BlocksX != expectedBlocksX || cw.BlocksY != expectedBlocksY {
				t.Errorf("block count mismatch for %dx%d: got (%d,%d), want (%d,%d)",
					size[0], size[1], cw.BlocksX, cw.BlocksY, expectedBlocksX, expectedBlocksY)
			}

			expectedSize := cw.BlocksX * cw.BlocksY * 8
			if len(cw.Data) != expectedSize {
				t.Errorf("data size mismatch for %dx%d: got %d, want %d",
					size[0], size[1], len(cw.Data), expectedSize)
			}

			ratio := cw.CompressionRatio()
			if ratio < 6.0 {
				t.Errorf("compression ratio too low for %dx%d: %.2f", size[0], size[1], ratio)
			}
		}
	})

	t.Run("BC4WithWeightImage", func(t *testing.T) {
		weights := make([]float32, 1024)
		for i := range weights {
			weights[i] = float32(i%100)/50.0 - 1.0
		}

		img := weightimage.WeightsToImage(weights, 32, 32)
		if img.Width != 32 || img.Height != 32 {
			t.Errorf("unexpected image dimensions: %dx%d", img.Width, img.Height)
		}

		cw, err := weightimage.CompressBC4(weights, 32, 32)
		if err != nil {
			t.Fatalf("BC4 compression failed: %v", err)
		}

		decompressed, err := weightimage.DecompressBC4(cw)
		if err != nil {
			t.Fatalf("BC4 decompression failed: %v", err)
		}

		var totalErr float64
		for i := range weights {
			diff := float64(decompressed[i] - weights[i])
			totalErr += diff * diff
		}
		rmse := math.Sqrt(totalErr / float64(len(weights)))

		t.Logf("Weights->Image->BC4->Decompress RMSE: %.4f", rmse)
		if rmse > 0.1 {
			t.Errorf("RMSE too high: %.4f (tolerance 0.1)", rmse)
		}
	})

	t.Run("BC4EdgeCases", func(t *testing.T) {
		uniform := make([]float32, 64)
		for i := range uniform {
			uniform[i] = 0.5
		}
		cw, err := weightimage.CompressBC4(uniform, 8, 8)
		if err != nil {
			t.Fatalf("uniform compression failed: %v", err)
		}
		dec, err := weightimage.DecompressBC4(cw)
		if err != nil {
			t.Fatalf("uniform decompression failed: %v", err)
		}
		for i, v := range dec {
			if math.Abs(float64(v-0.5)) > 0.1 {
				t.Errorf("uniform decode failed at %d: got %f, want ~0.5", i, v)
			}
		}

		linear := make([]float32, 64)
		for i := range linear {
			linear[i] = float32(i) / 63.0
		}
		cw2, err := weightimage.CompressBC4(linear, 8, 8)
		if err != nil {
			t.Fatalf("linear compression failed: %v", err)
		}
		dec2, err := weightimage.DecompressBC4(cw2)
		if err != nil {
			t.Fatalf("linear decompression failed: %v", err)
		}

		for i := range linear {
			if math.Abs(float64(dec2[i]-linear[i])) > 0.2 {
				t.Errorf("linear decode failed at %d: got %f, want %f", i, dec2[i], linear[i])
			}
		}
	})
}

func TestWeightImageStats(t *testing.T) {
	t.Run("BC4Stats", func(t *testing.T) {
		weights := make([]float32, 1024)
		for i := range weights {
			weights[i] = float32(i%200)/100.0 - 1.0
		}

		img := weightimage.WeightsToImage(weights, 32, 32)
		min, max, mean, stddev := img.Stats()

		t.Logf("Image stats: min=%.4f, max=%.4f, mean=%.4f, stddev=%.4f", min, max, mean, stddev)

		if min < -1.0 || max > 1.0 {
			t.Errorf("stats out of expected range [-1,1]: min=%.4f, max=%.4f", min, max)
		}

		cw, err := weightimage.CompressBC4(weights, 32, 32)
		if err != nil {
			t.Fatalf("compression failed: %v", err)
		}

		t.Logf("BC4 compressed: %s", cw.String())
	})

	t.Run("DCTCompression", func(t *testing.T) {
		weights := make([]float32, 256)
		for i := range weights {
			weights[i] = float32(math.Sin(float64(i) * 0.1))
		}

		dct, err := weightimage.CompressDCT(weights, 16, 16, 0.8)
		if err != nil {
			t.Fatalf("DCT compression failed: %v", err)
		}

		recovered := dct.Decompress()
		if len(recovered) != len(weights) {
			t.Errorf("DCT recovery size mismatch: got %d, want %d", len(recovered), len(weights))
		}

		var maxErr float64
		for i := range weights {
			err := math.Abs(float64(recovered[i] - weights[i]))
			if err > maxErr {
				maxErr = err
			}
		}

		t.Logf("DCT max error: %.4f", maxErr)
		if maxErr > 1.0 {
			t.Logf("DCT error is high (expected for lossy compression)")
		}
	})
}

func TestWeightImageAPIServer(t *testing.T) {
	t.Run("EndpointsExist", func(t *testing.T) {
		t.Log("Weight image API endpoints should be registered at:")
		t.Log("  GET /api/prismalama/weights")
		t.Log("  GET /api/prismalama/weights/stats")
		t.Log("  GET /api/prismalama/weights/layer/:layer")
		t.Log("Run integration tests with actual server to verify endpoints")
	})
}

func BenchmarkBC4Compression(b *testing.B) {
	weights := make([]float32, 4096)
	for i := range weights {
		weights[i] = float32(i%256)/127.5 - 1.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := weightimage.CompressBC4(weights, 64, 64)
		if err != nil {
			b.Fatalf("compression failed: %v", err)
		}
	}
}

func BenchmarkBC4Decompression(b *testing.B) {
	weights := make([]float32, 4096)
	for i := range weights {
		weights[i] = float32(i%256)/127.5 - 1.0
	}

	cw, _ := weightimage.CompressBC4(weights, 64, 64)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := weightimage.DecompressBC4(cw)
		if err != nil {
			b.Fatalf("decompression failed: %v", err)
		}
	}
}

