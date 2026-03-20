package ml

import "sync"

type DeviceCapability struct {
	DeviceID          DeviceID
	ComputeType       string
	MemoryBandwidth   float64
	ComputeThroughput float64
	LatencyProfile    string
}

type OffloadPolicy struct {
	LayerDevices []DeviceID
	KVCacheOnCPU bool
	AttentionOn  DeviceID
	EmbeddingOn  DeviceID
}

type Scheduler interface {
	ComputeOffloadPolicy(model *Model, availableDevices []DeviceInfo, memoryBudget uint64) (*OffloadPolicy, error)
}

type HeterogeneousScheduler struct {
	deviceCapabilities map[DeviceID]DeviceCapability
	layerCosts         map[int]LayerCost
	mu                 sync.RWMutex
}

type LayerCost struct {
	ComputeCost float64
	MemoryCost  uint64
}

func NewHeterogeneousScheduler() *HeterogeneousScheduler {
	return &HeterogeneousScheduler{
		deviceCapabilities: make(map[DeviceID]DeviceCapability),
		layerCosts:         make(map[int]LayerCost),
	}
}

func (s *HeterogeneousScheduler) RegisterDeviceCapability(cap DeviceCapability) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.deviceCapabilities[cap.DeviceID] = cap
}

func (s *HeterogeneousScheduler) SetLayerCost(layerIdx int, cost LayerCost) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.layerCosts[layerIdx] = cost
}

func (s *HeterogeneousScheduler) ComputeOffloadPolicy(model Model, availableDevices []DeviceInfo, memoryBudget uint64) (*OffloadPolicy, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	numLayers := model.NumLayers()
	policy := &OffloadPolicy{
		LayerDevices: make([]DeviceID, numLayers),
	}

	totalMemoryUsed := uint64(0)
	availableMemory := memoryBudget

	cpuDevice := DeviceID{ID: "cpu", Library: "cpu"}
	policy.EmbeddingOn = cpuDevice
	policy.AttentionOn = cpuDevice

	for i := 0; i < model.NumLayers(); i++ {
		layerCost, ok := s.layerCosts[i]
		if !ok {
			layerCost = LayerCost{ComputeCost: 1.0, MemoryCost: 100 * 1024 * 1024}
		}

		var bestDevice DeviceID
		bestScore := float64(0)

		for _, dev := range availableDevices {
			cap, ok := s.deviceCapabilities[dev.DeviceID]
			if !ok {
				continue
			}

			if cap.MemoryBandwidth <= 0 {
				continue
			}

			score := cap.ComputeThroughput / layerCost.ComputeCost
			if score > bestScore && totalMemoryUsed+layerCost.MemoryCost <= availableMemory {
				bestScore = score
				bestDevice = dev.DeviceID
			}
		}

		if bestDevice.ID == "" {
			bestDevice = cpuDevice
		} else {
			totalMemoryUsed += layerCost.MemoryCost
		}

		policy.LayerDevices[i] = bestDevice
	}

	if len(availableDevices) > 0 && totalMemoryUsed > memoryBudget/2 {
		policy.KVCacheOnCPU = true
	}

	return policy, nil
}

func (s *HeterogeneousScheduler) GetDeviceCapability(deviceID DeviceID) (DeviceCapability, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	cap, ok := s.deviceCapabilities[deviceID]
	return cap, ok
}

type Model interface {
	NumLayers() int
	LayerSize(layerIdx int) uint64
}

type SimpleModel struct {
	numLayers  int
	layerSizes []uint64
}

func NewSimpleModel(numLayers int, layerSizes []uint64) *SimpleModel {
	return &SimpleModel{
		numLayers:  numLayers,
		layerSizes: layerSizes,
	}
}

func (m *SimpleModel) NumLayers() int {
	return m.numLayers
}

func (m *SimpleModel) LayerSize(layerIdx int) uint64 {
	if layerIdx >= 0 && layerIdx < len(m.layerSizes) {
		return m.layerSizes[layerIdx]
	}
	return 0
}
