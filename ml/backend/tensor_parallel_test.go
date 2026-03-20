package backend

import (
	"testing"

	"github.com/ollama/ollama/ml"
)

func TestTensorParallelConfig(t *testing.T) {
	config := TensorParallelConfig{
		TPWorldSize:       4,
		TPRank:            1,
		AllReduceStrategy: "Ring",
		ColumnParallel:    true,
		RowParallel:       false,
	}

	if config.TPWorldSize != 4 {
		t.Errorf("expected TPWorldSize 4, got %d", config.TPWorldSize)
	}
	if config.TPRank != 1 {
		t.Errorf("expected TPRank 1, got %d", config.TPRank)
	}
	if config.AllReduceStrategy != "Ring" {
		t.Errorf("expected AllReduceStrategy Ring, got %s", config.AllReduceStrategy)
	}
}

func TestTensorParallelAllReduce_New(t *testing.T) {
	config := TensorParallelConfig{
		TPWorldSize:       2,
		TPRank:            0,
		AllReduceStrategy: "Ring",
	}

	allReduce := NewTensorParallelAllReduce(config, nil)
	if allReduce == nil {
		t.Fatal("expected non-nil TensorParallelAllReduce")
	}
	if allReduce.config.TPWorldSize != 2 {
		t.Errorf("expected TPWorldSize 2, got %d", allReduce.config.TPWorldSize)
	}
}

func TestTensorParallelAllReduce_AllReduce(t *testing.T) {
	config := TensorParallelConfig{
		TPWorldSize:       2,
		TPRank:            0,
		AllReduceStrategy: "Ring",
	}

	allReduce := NewTensorParallelAllReduce(config, nil)

	err := allReduce.AllReduce(nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestTensorParallelAllReduce_TreeAllReduce(t *testing.T) {
	config := TensorParallelConfig{
		TPWorldSize:       4,
		TPRank:            0,
		AllReduceStrategy: "Tree",
	}

	allReduce := NewTensorParallelAllReduce(config, nil)

	err := allReduce.AllReduce(nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestTensorParallelAllReduce_ButterflyAllReduce(t *testing.T) {
	config := TensorParallelConfig{
		TPWorldSize:       4,
		TPRank:            0,
		AllReduceStrategy: "Butterfly",
	}

	allReduce := NewTensorParallelAllReduce(config, nil)

	err := allReduce.AllReduce(nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestPipelineStage_New(t *testing.T) {
	stage := NewPipelineStage(0, 10, ml.DeviceID{ID: "0"})
	if stage == nil {
		t.Fatal("expected non-nil PipelineStage")
	}
	if stage.FirstLayer != 0 {
		t.Errorf("expected FirstLayer 0, got %d", stage.FirstLayer)
	}
	if stage.LastLayer != 10 {
		t.Errorf("expected LastLayer 10, got %d", stage.LastLayer)
	}
}

func TestPipelineStage_SetMicrobatchConfig(t *testing.T) {
	stage := NewPipelineStage(0, 10, ml.DeviceID{ID: "0"})
	stage.SetMicrobatchConfig(4, 8)

	if stage.MicrobatchSize != 4 {
		t.Errorf("expected MicrobatchSize 4, got %d", stage.MicrobatchSize)
	}
	if stage.NumMicrobatches != 8 {
		t.Errorf("expected NumMicrobatches 8, got %d", stage.NumMicrobatches)
	}
}

func TestColumnParallelLinear_New(t *testing.T) {
	config := TensorParallelConfig{
		TPWorldSize: 2,
		TPRank:      0,
	}

	linear := NewColumnParallelLinear(1024, 2048, config)
	if linear == nil {
		t.Fatal("expected non-nil ColumnParallelLinear")
	}
	if linear.inputSize != 1024 {
		t.Errorf("expected inputSize 1024, got %d", linear.inputSize)
	}
}

func TestColumnParallelLinear_Forward(t *testing.T) {
	config := TensorParallelConfig{
		TPWorldSize: 2,
		TPRank:      0,
	}

	linear := NewColumnParallelLinear(1024, 2048, config)
	result, err := linear.Forward(nil, nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if result != nil {
		t.Error("expected nil result for nil input")
	}
}

func TestRowParallelLinear_New(t *testing.T) {
	config := TensorParallelConfig{
		TPWorldSize: 2,
		TPRank:      0,
	}

	linear := NewRowParallelLinear(1024, 2048, config)
	if linear == nil {
		t.Fatal("expected non-nil RowParallelLinear")
	}
	if linear.inputSize != 1024 {
		t.Errorf("expected inputSize 1024, got %d", linear.inputSize)
	}
}

func TestRowParallelLinear_Forward(t *testing.T) {
	config := TensorParallelConfig{
		TPWorldSize: 2,
		TPRank:      0,
	}

	linear := NewRowParallelLinear(1024, 2048, config)
	result, err := linear.Forward(nil, nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if result != nil {
		t.Error("expected nil result for nil input")
	}
}

func TestTensorParallelScheduler_New(t *testing.T) {
	config := TensorParallelConfig{
		TPWorldSize: 2,
		TPRank:      0,
	}

	scheduler := NewTensorParallelScheduler(config, 32)
	if scheduler == nil {
		t.Fatal("expected non-nil TensorParallelScheduler")
	}
	if len(scheduler.stages) != 2 {
		t.Errorf("expected 2 stages, got %d", len(scheduler.stages))
	}
}

func TestTensorParallelScheduler_MinimizePipelineBubble(t *testing.T) {
	config := TensorParallelConfig{
		TPWorldSize: 4,
		TPRank:      0,
	}

	scheduler := NewTensorParallelScheduler(config, 32)
	bubble := scheduler.MinimizePipelineBubble()

	if bubble != 8.0 {
		t.Errorf("expected bubble 8.0, got %f", bubble)
	}
}

func TestTensorParallelScheduler_GetStage(t *testing.T) {
	config := TensorParallelConfig{
		TPWorldSize: 2,
		TPRank:      0,
	}

	scheduler := NewTensorParallelScheduler(config, 32)

	stage := scheduler.GetStage(5)
	if stage == nil {
		t.Fatal("expected non-nil stage")
	}
	if stage.FirstLayer != 0 {
		t.Errorf("expected FirstLayer 0, got %d", stage.FirstLayer)
	}
}
