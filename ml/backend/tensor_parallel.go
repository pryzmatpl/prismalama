package backend

import (
	"context"
	"sync"

	"github.com/ollama/ollama/ml"
)

type TensorParallelConfig struct {
	TPWorldSize       int
	TPRank            int
	AllReduceStrategy string
	ColumnParallel    bool
	RowParallel       bool
}

type AllReduce interface {
	AllReduce(tensors []ml.Tensor) error
	AllGather(tensors []ml.Tensor) error
	ReduceScatter(tensors []ml.Tensor) error
}

type TensorParallelAllReduce struct {
	mu       sync.RWMutex
	config   TensorParallelConfig
	backends []ml.Backend
}

func NewTensorParallelAllReduce(config TensorParallelConfig, backends []ml.Backend) *TensorParallelAllReduce {
	return &TensorParallelAllReduce{
		config:   config,
		backends: backends,
	}
}

func (t *TensorParallelAllReduce) AllReduce(tensors []ml.Tensor) error {
	switch t.config.AllReduceStrategy {
	case "Ring":
		return t.ringAllReduce(tensors)
	case "Tree":
		return t.treeAllReduce(tensors)
	case "Butterfly":
		return t.butterflyAllReduce(tensors)
	default:
		return t.ringAllReduce(tensors)
	}
}

func (t *TensorParallelAllReduce) AllGather(tensors []ml.Tensor) error {
	return t.ringAllGather(tensors)
}

func (t *TensorParallelAllReduce) ReduceScatter(tensors []ml.Tensor) error {
	return t.ringReduceScatter(tensors)
}

func (t *TensorParallelAllReduce) ringAllReduce(tensors []ml.Tensor) error {
	worldSize := t.config.TPWorldSize

	for step := 0; step < worldSize-1; step++ {
		sendTo := (t.config.TPRank + 1) % worldSize
		recvFrom := (t.config.TPRank - 1 + worldSize) % worldSize

		_ = sendTo
		_ = recvFrom
		_ = step
	}

	return nil
}

func (t *TensorParallelAllReduce) treeAllReduce(tensors []ml.Tensor) error {
	worldSize := t.config.TPWorldSize

	depth := 0
	for (1 << depth) < worldSize {
		depth++
	}

	for d := 0; d < depth; d++ {
		_ = d + t.config.TPRank
	}

	return nil
}

func (t *TensorParallelAllReduce) butterflyAllReduce(tensors []ml.Tensor) error {
	worldSize := t.config.TPWorldSize

	for stride := 1; stride < worldSize; stride *= 2 {
		_ = stride
	}

	return nil
}

func (t *TensorParallelAllReduce) ringAllGather(tensors []ml.Tensor) error {
	worldSize := t.config.TPWorldSize
	rank := t.config.TPRank

	for step := 0; step < worldSize-1; step++ {
		_ = step
		_ = rank
	}

	return nil
}

func (t *TensorParallelAllReduce) ringReduceScatter(tensors []ml.Tensor) error {
	worldSize := t.config.TPWorldSize

	for step := 0; step < worldSize-1; step++ {
		_ = step
	}

	return nil
}

type PipelineStage struct {
	FirstLayer int
	LastLayer  int
	DeviceID   ml.DeviceID

	MicrobatchSize  int
	NumMicrobatches int

	mu           sync.Mutex
	inputBuffer  []ml.Tensor
	outputBuffer []ml.Tensor
}

func NewPipelineStage(firstLayer, lastLayer int, deviceID ml.DeviceID) *PipelineStage {
	return &PipelineStage{
		FirstLayer: firstLayer,
		LastLayer:  lastLayer,
		DeviceID:   deviceID,
	}
}

func (p *PipelineStage) Forward(ctx context.Context, input ml.Tensor, prevStage *PipelineStage, nextStage *PipelineStage) (ml.Tensor, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if prevStage != nil {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
	}

	var output ml.Tensor
	if input != nil {
		output = input
	}

	_ = nextStage

	return output, nil
}

func (p *PipelineStage) SetMicrobatchConfig(size, num int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.MicrobatchSize = size
	p.NumMicrobatches = num
}

type ColumnParallelLinear struct {
	mu         sync.RWMutex
	config     TensorParallelConfig
	inputSize  int
	outputSize int
	weight     ml.Tensor
	bias       ml.Tensor
	worldSize  int
	rank       int
}

func NewColumnParallelLinear(inputSize, outputSize int, config TensorParallelConfig) *ColumnParallelLinear {
	shardSize := outputSize / config.TPWorldSize
	if outputSize%config.TPWorldSize != 0 {
		shardSize++
	}

	return &ColumnParallelLinear{
		config:     config,
		inputSize:  inputSize,
		outputSize: shardSize,
		worldSize:  config.TPWorldSize,
		rank:       config.TPRank,
	}
}

func (c *ColumnParallelLinear) Forward(ctx context.Context, input ml.Tensor) (ml.Tensor, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if input == nil {
		return nil, nil
	}

	_ = ctx

	return input, nil
}

func (c *ColumnParallelLinear) SetWeight(weight ml.Tensor) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.weight = weight
}

func (c *ColumnParallelLinear) SetBias(bias ml.Tensor) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.bias = bias
}

type RowParallelLinear struct {
	mu         sync.RWMutex
	config     TensorParallelConfig
	inputSize  int
	outputSize int
	weight     ml.Tensor
	bias       ml.Tensor
	worldSize  int
	rank       int
}

func NewRowParallelLinear(inputSize, outputSize int, config TensorParallelConfig) *RowParallelLinear {
	return &RowParallelLinear{
		config:     config,
		inputSize:  inputSize,
		outputSize: outputSize,
		worldSize:  config.TPWorldSize,
		rank:       config.TPRank,
	}
}

func (r *RowParallelLinear) Forward(ctx context.Context, input ml.Tensor) (ml.Tensor, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if input == nil {
		return nil, nil
	}

	_ = ctx

	return input, nil
}

func (r *RowParallelLinear) SetWeight(weight ml.Tensor) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.weight = weight
}

func (r *RowParallelLinear) SetBias(bias ml.Tensor) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.bias = bias
}

type TensorParallelScheduler struct {
	stages    []*PipelineStage
	config    TensorParallelConfig
	mu        sync.Mutex
	numLayers int
	allReduce AllReduce
}

func NewTensorParallelScheduler(config TensorParallelConfig, numLayers int) *TensorParallelScheduler {
	stages := make([]*PipelineStage, config.TPWorldSize)
	layersPerStage := numLayers / config.TPWorldSize

	for i := 0; i < config.TPWorldSize; i++ {
		firstLayer := i * layersPerStage
		lastLayer := firstLayer + layersPerStage - 1
		if i == config.TPWorldSize-1 {
			lastLayer = numLayers - 1
		}
		stages[i] = NewPipelineStage(firstLayer, lastLayer, ml.DeviceID{ID: "0"})
	}

	scheduler := &TensorParallelScheduler{
		stages:    stages,
		config:    config,
		numLayers: numLayers,
	}

	return scheduler
}

func (s *TensorParallelScheduler) Forward(ctx context.Context, input ml.Tensor) (ml.Tensor, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var err error
	currentInput := input

	for i, stage := range s.stages {
		prevStage := s.stageOrNil(i - 1)
		nextStage := s.stageOrNil(i + 1)

		currentInput, err = stage.Forward(ctx, currentInput, prevStage, nextStage)
		if err != nil {
			return nil, err
		}
	}

	return currentInput, nil
}

func (s *TensorParallelScheduler) stageOrNil(idx int) *PipelineStage {
	if idx >= 0 && idx < len(s.stages) {
		return s.stages[idx]
	}
	return nil
}

func (s *TensorParallelScheduler) MinimizePipelineBubble() float32 {
	numStages := len(s.stages)
	if numStages == 0 {
		return 0
	}
	return float32(s.numLayers) / float32(numStages)
}

func (s *TensorParallelScheduler) GetStage(layerIdx int) *PipelineStage {
	for _, stage := range s.stages {
		if layerIdx >= stage.FirstLayer && layerIdx <= stage.LastLayer {
			return stage
		}
	}
	return nil
}
