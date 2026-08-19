# ml/weightimage — Weight Compression (BC4/DCT)

> GPU texture-style compression for model weights. Reduces NVMe→GPU transfer bandwidth.

## Compression formats

| Format | Ratio | Method | Use case |
|--------|-------|--------|----------|
| **BC4** | ~4:1 | 4×4 block, 2 endpoints + 3-bit indices per pixel | Lossless-ish, fast decode |
| **DCT** | Variable | 2D discrete cosine transform + frequency quantization | Higher compression, tunable quality |
| **F32** | 1:1 | No compression (passthrough) | Baseline |

## Files

| File | Lines | Purpose |
|------|------:|---------|
| `weightimage.go` | 222 | Weight↔Image conversion, PNG export, heatmap visualization |
| `texture.go` | 350 | GPU texture backend abstraction, RGBA layout, bilinear sampling |
| `compression.go` | 341 | BC4 block codec, DCT codec, compression dispatcher |

## Key types

```go
// Weight data as 2D image
type WeightImage struct {
    Width, Height int
    Format        ImageFormat
    Data          []float32
}

// BC4 compressed block (4×4 pixels → 8 bytes)
type BC4Block struct {
    Endpoints [8]byte   // min/max values
    Indices   [16]byte  // 3-bit index per pixel
}

// DCT compressed weights
type DCTWeights struct {
    Width, Height int
    DCTData       []float32  // Frequency coefficients
    Quant         []float32  // Quantization table
    Frequency     []float32  // Frequency weights
}

// Compressed output
type CompressedWeights struct {
    Format        CompressionFormat
    Width, Height int
    BlocksX, BlocksY int
    Data          []byte
}
```

## Usage

```go
// Compress
compressed := CompressBC4(weights, width, height)
ratio := compressed.CompressionRatio()

// Decompress
restored := DecompressBC4(compressed)

// Visualization
img := WeightsToImage(data, width, height)
png := img.ToHeatmapPNG()
```

## GPU texture pipeline

`VulkanWeightTextureBackend` manages compressed weight textures:
1. `CreateWeightTexture()` — float32 → RGBA byte layout
2. `LoadCompressedBlock()` — store CompressedWeights
3. `BilinearSampler.Sample(u, v)` — interpolate at normalized coordinates

Used by `ml/streaming/texture_inference.go` for compressed streaming inference.
