package quantization

import (
	"testing"
)

func TestQuantType(t *testing.T) {
	if FP32 != 0 {
		t.Errorf("expected FP32 = 0, got %d", FP32)
	}
	if FP16 != 1 {
		t.Errorf("expected FP16 = 1, got %d", FP16)
	}
	if BF16 != 2 {
		t.Errorf("expected BF16 = 2, got %d", BF16)
	}
	if INT8 != 3 {
		t.Errorf("expected INT8 = 3, got %d", INT8)
	}
	if INT4 != 4 {
		t.Errorf("expected INT4 = 4, got %d", INT4)
	}
}

func TestQuantizationConfig(t *testing.T) {
	config := QuantizationConfig{
		LayerConfigs:      map[int]QuantType{0: FP16, 1: INT8},
		TargetPrecision:   "int8",
		CalibrationData:   [][]float32{{1.0, 2.0}, {3.0, 4.0}},
		CalibrationMethod: "minmax",
	}

	if config.TargetPrecision != "int8" {
		t.Errorf("expected int8, got %s", config.TargetPrecision)
	}
	if config.LayerConfigs[0] != FP16 {
		t.Errorf("expected FP16 for layer 0, got %d", config.LayerConfigs[0])
	}
}

func TestQuantizedTensor(t *testing.T) {
	data := []byte{1, 2, 3, 4}
	qt := NewQuantizedTensor(data, []int{4}, INT8, 0.1)
	if qt == nil {
		t.Fatal("expected non-nil QuantizedTensor")
	}
	if qt.QuantType() != INT8 {
		t.Errorf("expected INT8, got %d", qt.QuantType())
	}
	if len(qt.Data()) != 4 {
		t.Errorf("expected data length 4, got %d", len(qt.Data()))
	}
}

func TestQuantizeTensor_INT8(t *testing.T) {
	data := []float32{0.0, 0.5, -0.5, 1.0}
	qt, err := QuantizeTensor(data, INT8)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if qt == nil {
		t.Fatal("expected non-nil QuantizedTensor")
	}
	if qt.QuantType() != INT8 {
		t.Errorf("expected INT8, got %d", qt.QuantType())
	}
}

func TestQuantizeTensor_INT4(t *testing.T) {
	data := []float32{0.0, 0.5, -0.5, 1.0}
	qt, err := QuantizeTensor(data, INT4)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if qt == nil {
		t.Fatal("expected non-nil QuantizedTensor")
	}
	if qt.QuantType() != INT4 {
		t.Errorf("expected INT4, got %d", qt.QuantType())
	}
}

func TestDequantizeTensor(t *testing.T) {
	data := []byte{0, 127}
	qt := NewQuantizedTensor(data, []int{2}, INT8, 1.0/127.0)
	result, err := DequantizeTensor(qt)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(result) != 2 {
		t.Errorf("expected length 2, got %d", len(result))
	}
}

func TestPTQCalibrator_New(t *testing.T) {
	calibrator := NewPTQCalibrator("minmax", 99.0)
	if calibrator == nil {
		t.Fatal("expected non-nil PTQCalibrator")
	}
	if calibrator.method != "minmax" {
		t.Errorf("expected method minmax, got %s", calibrator.method)
	}
	if calibrator.percentile != 99.0 {
		t.Errorf("expected percentile 99.0, got %f", calibrator.percentile)
	}
}

func TestPTQCalibrator_AddCalibrationData(t *testing.T) {
	calibrator := NewPTQCalibrator("minmax", 99.0)
	calibrator.AddCalibrationData([]float32{1.0, 2.0, 3.0})
	calibrator.AddCalibrationData([]float32{-1.0, 0.0, 4.0})

	if len(calibrator.data) != 2 {
		t.Errorf("expected 2 calibration sets, got %d", len(calibrator.data))
	}
}

func TestPTQCalibrator_ComputeScales_MinMax(t *testing.T) {
	calibrator := NewPTQCalibrator("minmax", 99.0)
	calibrator.AddCalibrationData([]float32{-1.0, 0.0, 1.0})

	scales, err := calibrator.ComputeScales()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if scales == nil {
		t.Fatal("expected non-nil scales")
	}
}

func TestPTQCalibrator_ComputeScales_Empty(t *testing.T) {
	calibrator := NewPTQCalibrator("minmax", 99.0)

	scales, err := calibrator.ComputeScales()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if scales != nil {
		t.Error("expected nil scales for empty data")
	}
}

func TestAWQQuantizer_New(t *testing.T) {
	calibrator := NewPTQCalibrator("minmax", 99.0)
	awq := NewAWQQuantizer(calibrator, 128)
	if awq == nil {
		t.Fatal("expected non-nil AWQQuantizer")
	}
	if awq.groupSize != 128 {
		t.Errorf("expected groupSize 128, got %d", awq.groupSize)
	}
}

func TestGPTQQuantizer_New(t *testing.T) {
	calibrator := NewPTQCalibrator("minmax", 99.0)
	gptq := NewGPTQQuantizer(calibrator, 4, 128)
	if gptq == nil {
		t.Fatal("expected non-nil GPTQQuantizer")
	}
	if gptq.bits != 4 {
		t.Errorf("expected bits 4, got %d", gptq.bits)
	}
	if gptq.groupSize != 128 {
		t.Errorf("expected groupSize 128, got %d", gptq.groupSize)
	}
}
