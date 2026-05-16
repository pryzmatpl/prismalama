package weightimage

import (
	"fmt"
	"unsafe"

	"github.com/ollama/ollama/ml"
)

type GGMLTensorShim struct {
	Ne []uint64
}

type GGMLTextureAdapter struct {
	backend *VulkanWeightTextureBackend
}

func NewGGMLTextureAdapter() *GGMLTextureAdapter {
	return &GGMLTextureAdapter{
		backend: NewVulkanWeightTextureBackend("texture", "weightimage"),
	}
}

func (a *GGMLTextureAdapter) AdaptTensor(name string, tensor *GGMLTensorShim, data []byte) (*WeightTexture, error) {
	shape := tensor.Ne
	if len(shape) < 2 {
		return nil, fmt.Errorf("tensor must have at least 2 dimensions")
	}

	rows := int(shape[0])
	cols := int(shape[1])

	floatData := make([]float32, len(data)/4)
	for i := 0; i < len(floatData); i++ {
		floatData[i] = float32(data[i*4])/255.0*2.0 - 1.0
	}

	wt, err := a.backend.CreateWeightTexture(name, cols, rows, floatData)
	if err != nil {
		return nil, err
	}

	return wt, nil
}

func (a *GGMLTextureAdapter) CompressToBC4(name string, weights []float32, w, h int) (*CompressedWeights, error) {
	cw, err := CompressBC4(weights, w, h)
	if err != nil {
		return nil, err
	}

	err = a.backend.LoadCompressedBlock(cw)
	if err != nil {
		return nil, err
	}

	return cw, nil
}

type TextureBackend interface {
	ml.Backend
	CreateWeightTexture(name string, width, height int, data []float32) (WeightTexture, error)
	LoadCompressedBlock(block *CompressedWeights) error
}

type WeightTexture struct {
	Name      string
	Width     int
	Height    int
	Format    ImageFormat
	Bytes     []byte
	TextureID uintptr
}

type VulkanWeightTextureBackend struct {
	deviceID       string
	library        string
	textures       map[string]*WeightTexture
	compressedData map[string]*CompressedWeights
}

func NewVulkanWeightTextureBackend(deviceID, library string) *VulkanWeightTextureBackend {
	return &VulkanWeightTextureBackend{
		deviceID:       deviceID,
		library:        library,
		textures:       make(map[string]*WeightTexture),
		compressedData: make(map[string]*CompressedWeights),
	}
}

func (b *VulkanWeightTextureBackend) Close() error {
	b.textures = nil
	b.compressedData = nil
	return nil
}

func (b *VulkanWeightTextureBackend) Get(name string) ml.Tensor {
	return nil
}

func (b *VulkanWeightTextureBackend) NewContext() ml.Context {
	return nil
}

func (b *VulkanWeightTextureBackend) CreateWeightTexture(name string, width, height int, data []float32) (*WeightTexture, error) {
	img := WeightsToImage(data, height, width)

	wt := &WeightTexture{
		Name:   name,
		Width:  img.Width,
		Height: img.Height,
		Format: img.Format,
		Bytes:  make([]byte, len(img.Data)*4),
	}

	for i, v := range img.Data {
		offset := i * 4
		val := uint8(v * 255)
		wt.Bytes[offset] = val
		wt.Bytes[offset+1] = val
		wt.Bytes[offset+2] = val
		wt.Bytes[offset+3] = 255
	}

	b.textures[name] = wt
	return wt, nil
}

func (b *VulkanWeightTextureBackend) LoadCompressedBlock(block *CompressedWeights) error {
	name := fmt.Sprintf("compressed_%dx%d", block.Width, block.Height)
	b.compressedData[name] = block
	return nil
}

func (b *VulkanWeightTextureBackend) GetCompressedData(name string) (*CompressedWeights, bool) {
	block, ok := b.compressedData[name]
	return block, ok
}

func (wt *WeightTexture) IsCompressed() bool {
	return wt.Format == FormatBC4 || wt.Format == FormatBC7 || wt.Format == FormatDCT
}

func (wt *WeightTexture) CompressionRatio() float64 {
	if wt.IsCompressed() {
		return float64(wt.Width*wt.Height*4) / float64(len(wt.Bytes))
	}
	return 1.0
}

func (wt *WeightTexture) Metadata() map[string]interface{} {
	return map[string]interface{}{
		"name":       wt.Name,
		"width":      wt.Width,
		"height":     wt.Height,
		"format":     wt.Format,
		"compressed": wt.IsCompressed(),
		"bytes":      len(wt.Bytes),
		"ratio":      wt.CompressionRatio(),
	}
}

type TextureLayout struct {
	Width     int
	Height    int
	Channels  int
	RowStride int
}

func (tl *TextureLayout) ElementSize() int {
	switch tl.Channels {
	case 1:
		return 1
	case 3:
		return 3
	case 4:
		return 4
	default:
		return tl.Channels
	}
}

func (tl *TextureLayout) TotalBytes() int {
	return tl.Height * tl.RowStride
}

func ReshapeToTextureLayout(rows, cols int) *TextureLayout {
	size := nextPow2(int(float64(rows*cols) / float64(cols)))
	if size < 4 {
		size = 4
	}

	height := (rows*cols + size - 1) / size
	if height < 4 {
		height = 4
	}

	return &TextureLayout{
		Width:     size,
		Height:    height,
		Channels:  4,
		RowStride: size * 4,
	}
}

func (tl *TextureLayout) Reshape(weights []float32) []byte {
	result := make([]byte, tl.TotalBytes())

	for i := 0; i < len(weights) && i < tl.Width*tl.Height; i++ {
		offset := i * 4
		if offset+3 >= len(result) {
			break
		}
		val := uint8(weights[i]*127.5 + 127.5)
		result[offset] = val
		result[offset+1] = val
		result[offset+2] = val
		result[offset+3] = 255
	}

	return result
}

type WeightTextureHandle struct {
	texture unsafe.Pointer
	layout  *TextureLayout
}

func (h *WeightTextureHandle) Texture() unsafe.Pointer {
	return h.texture
}

func (h *WeightTextureHandle) Layout() *TextureLayout {
	return h.layout
}

func (h *WeightTextureHandle) Sample(x, y float32) (r, g, b, a float32) {
	return 0.5, 0.5, 0.5, 1.0
}

type TextureSampler struct {
	MinFilter int
	MagFilter int
	WrapS     int
	WrapT     int
}

const (
	FilterNearest = 0
	FilterLinear  = 1
	FilterBilinear = 2
	FilterTrilinear = 3

	WrapClamp = 0
	WrapRepeat = 1
	WrapMirror = 2
)

func NewTextureSampler(minFilter, magFilter, wrapS, wrapT int) *TextureSampler {
	return &TextureSampler{
		MinFilter: minFilter,
		MagFilter: magFilter,
		WrapS:     wrapS,
		WrapT:     wrapT,
	}
}

type BilinearSampler struct {
	layout *TextureLayout
	data   []byte
}

func NewBilinearSampler(layout *TextureLayout, data []byte) *BilinearSampler {
	return &BilinearSampler{
		layout: layout,
		data:   data,
	}
}

func (s *BilinearSampler) Sample(u, v float32) (r, g, b, a float32) {
	x := u * float32(s.layout.Width-1)
	y := v * float32(s.layout.Height-1)

	x0 := int(x)
	y0 := int(y)
	x1 := x0 + 1
	y1 := y0 + 1

	if x1 >= s.layout.Width {
		x1 = s.layout.Width - 1
	}
	if y1 >= s.layout.Height {
		y1 = s.layout.Height - 1
	}

	 fx := x - float32(x0)
	 fy := y - float32(y0)

	p00 := s.samplePixel(x0, y0)
	p10 := s.samplePixel(x1, y0)
	p01 := s.samplePixel(x0, y1)
	p11 := s.samplePixel(x1, y1)

	 r = lerp(lerp(p00.r, p10.r, fx), lerp(p01.r, p11.r, fx), fy)
	 g = lerp(lerp(p00.g, p10.g, fx), lerp(p01.g, p11.g, fx), fy)
	 b = lerp(lerp(p00.b, p10.b, fx), lerp(p01.b, p11.b, fx), fy)
	 a = lerp(lerp(p00.a, p10.a, fx), lerp(p01.a, p11.a, fx), fy)

	return r, g, b, a
}

func (s *BilinearSampler) samplePixel(x, y int) (c rgba) {
	idx := (y*s.layout.Width + x) * 4
	if idx+3 >= len(s.data) {
		return rgba{0.5, 0.5, 0.5, 1.0}
	}
	return rgba{
		float32(s.data[idx]) / 255.0,
		float32(s.data[idx+1]) / 255.0,
		float32(s.data[idx+2]) / 255.0,
		float32(s.data[idx+3]) / 255.0,
	}
}

type rgba struct {
	r, g, b, a float32
}

func lerp(a, b, t float32) float32 {
	return a + t*(b-a)
}

func (s *BilinearSampler) SampleMatrix(uStart, vStart, uEnd, vEnd float32, outRows, outCols int) []float32 {
	result := make([]float32, outRows*outCols)

	uStep := (uEnd - uStart) / float32(outCols)
	vStep := (vEnd - vStart) / float32(outRows)

	for row := 0; row < outRows; row++ {
		for col := 0; col < outCols; col++ {
			u := uStart + float32(col)*uStep
			v := vStart + float32(row)*vStep

			r, g, b, a := s.Sample(u, v)
			result[row*outCols+col] = (r + g + b) / 3.0 * a
		}
	}

	return result
}