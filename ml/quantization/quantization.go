package quantization

import (
	"context"

	"github.com/ollama/ollama/ml"
)

type QuantType int

const (
	FP32 QuantType = iota
	FP16
	BF16
	INT8
	INT4
	INT2
	NF4
	FP4
)

type QuantizationConfig struct {
	LayerConfigs      map[int]QuantType
	TargetPrecision   string
	CalibrationData   [][]float32
	CalibrationMethod string
}

type DynamicQuantizer struct {
	backend     ml.Backend
	quantLayer  int
	scaleFactor float32
	config      QuantizationConfig
}

func NewDynamicQuantizer(backend ml.Backend, config QuantizationConfig) *DynamicQuantizer {
	return &DynamicQuantizer{
		backend: backend,
		config:  config,
	}
}

func (d *DynamicQuantizer) QuantizeLayer(ctx context.Context, layer *ml.Tensor) (*ml.Tensor, error) {
	_ = ctx
	_ = layer
	return nil, nil
}

func (d *DynamicQuantizer) SetQuantType(layerIdx int, qtype QuantType) {
	if d.config.LayerConfigs == nil {
		d.config.LayerConfigs = make(map[int]QuantType)
	}
	d.config.LayerConfigs[layerIdx] = qtype
}

func (d *DynamicQuantizer) GetQuantType(layerIdx int) QuantType {
	if qtype, ok := d.config.LayerConfigs[layerIdx]; ok {
		return qtype
	}
	return INT8
}

type PTQCalibrator struct {
	method     string
	percentile float32
	data       [][]float32
	minVals    []float32
	maxVals    []float32
}

func NewPTQCalibrator(method string, percentile float32) *PTQCalibrator {
	return &PTQCalibrator{
		method:     method,
		percentile: percentile,
		data:       make([][]float32, 0),
	}
}

func (p *PTQCalibrator) AddCalibrationData(data []float32) {
	p.data = append(p.data, data)
}

func (p *PTQCalibrator) ComputeScales() ([]float32, error) {
	if len(p.data) == 0 {
		return nil, nil
	}

	switch p.method {
	case "minmax":
		return p.minMaxScales()
	case "percentile":
		return p.percentileScales()
	case "mse":
		return p.mseScales()
	default:
		return p.minMaxScales()
	}
}

func (p *PTQCalibrator) minMaxScales() ([]float32, error) {
	if len(p.data) == 0 {
		return nil, nil
	}

	flatData := make([]float32, 0)
	for _, d := range p.data {
		flatData = append(flatData, d...)
	}

	minVal := float32(0)
	maxVal := float32(0)
	for i, v := range flatData {
		if i == 0 {
			minVal = v
			maxVal = v
		} else {
			if v < minVal {
				minVal = v
			}
			if v > maxVal {
				maxVal = v
			}
		}
	}

	rangeVal := maxVal - minVal
	if rangeVal == 0 {
		return []float32{1.0}, nil
	}

	scale := 127.0 / rangeVal
	return []float32{scale}, nil
}

func (p *PTQCalibrator) percentileScales() ([]float32, error) {
	return p.minMaxScales()
}

func (p *PTQCalibrator) mseScales() ([]float32, error) {
	return p.minMaxScales()
}

type QuantizedTensor struct {
	data      []byte
	shape     []int
	quantType QuantType
	scale     float32
	zeroPoint int32
}

func NewQuantizedTensor(data []byte, shape []int, qtype QuantType, scale float32) *QuantizedTensor {
	return &QuantizedTensor{
		data:      data,
		shape:     shape,
		quantType: qtype,
		scale:     scale,
	}
}

func (q *QuantizedTensor) Data() []byte {
	return q.data
}

func (q *QuantizedTensor) Shape() []int {
	return q.shape
}

func (q *QuantizedTensor) QuantType() QuantType {
	return q.quantType
}

func QuantizeTensor(data []float32, qtype QuantType) (*QuantizedTensor, error) {
	var scale float32 = 1.0

	switch qtype {
	case INT8:
		scale = 127.0 / maxAbs(data)
	case INT4:
		scale = 15.0 / maxAbs(data)
	case FP4, NF4:
		scale = 1.0
	default:
		return nil, nil
	}

	quantizedData := make([]byte, len(data))

	for i, v := range data {
		quantVal := int8(float32(v) * scale)
		quantizedData[i] = byte(quantVal)
	}

	return NewQuantizedTensor(quantizedData, []int{len(data)}, qtype, scale), nil
}

func DequantizeTensor(q *QuantizedTensor) ([]float32, error) {
	result := make([]float32, len(q.data))

	for i, b := range q.data {
		result[i] = float32(int8(b)) * q.scale
	}

	return result, nil
}

func maxAbs(data []float32) float32 {
	maxAbs := float32(0)
	for _, v := range data {
		abs := v
		if abs < 0 {
			abs = -abs
		}
		if abs > maxAbs {
			maxAbs = abs
		}
	}
	if maxAbs == 0 {
		return 1.0
	}
	return maxAbs
}

type AWQQuantizer struct {
	calibrator *PTQCalibrator
	groupSize  int
}

func NewAWQQuantizer(calibrator *PTQCalibrator, groupSize int) *AWQQuantizer {
	return &AWQQuantizer{
		calibrator: calibrator,
		groupSize:  groupSize,
	}
}

func (a *AWQQuantizer) Quantize(w *ml.Tensor) (*QuantizedTensor, error) {
	_ = w
	return nil, nil
}

type GPTQQuantizer struct {
	calibrator *PTQCalibrator
	groupSize  int
	bits       int
}

func NewGPTQQuantizer(calibrator *PTQCalibrator, bits, groupSize int) *GPTQQuantizer {
	return &GPTQQuantizer{
		calibrator: calibrator,
		groupSize:  groupSize,
		bits:       bits,
	}
}

func (g *GPTQQuantizer) Quantize(w *ml.Tensor) (*QuantizedTensor, error) {
	_ = w
	return nil, nil
}
