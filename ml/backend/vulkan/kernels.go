package vulkan

import (
	"errors"
	"fmt"

	"github.com/ollama/ollama/ml"
)

// errNotImplemented is returned by Vulkan backend operations that have not yet been
// implemented with real Vulkan compute kernels.
var errNotImplemented = errors.New("vulkan backend: operation not implemented; " +
	"build with Vulkan compute support (GGML_VULKAN=1) and ensure GPU device is available")

type VulkanBackend struct {
	device              Device
	computePipeline     *VulkanComputePipeline
	memoryPool          *VulkanMemoryPool
	kernelCache         map[string]VulkanPipeline
	optimizedKernels    *VulkanOptimizedKernels
	descriptorPool      DescriptorPool
	descriptorSetLayout DescriptorSetLayout
}

type Device interface {
	Handle() uint64
	Name() string
}

type VulkanComputePipeline struct {
	device         Device
	computeShader  DescriptorSetLayout
	descriptorPool DescriptorPool
	kernelCache    map[string]VulkanPipeline
}

type VulkanPipeline interface {
	Handle() uint64
}

type DescriptorPool interface {
	Handle() uint64
}

type DescriptorSetLayout interface {
	Handle() uint64
}

type VulkanMemoryPool struct {
	device         Device
	allocations    map[uint64]*VulkanMemory
	totalAllocated uint64
	maxAllocated   uint64
	mu             int
}

type VulkanMemory struct {
	device       Device
	memoryHandle uint64
	size         uint64
	offset       uint64
	mappable     bool
}

type VulkanOptimizedKernels struct {
	backend *VulkanBackend
}

func NewVulkanBackend(device Device) *VulkanBackend {
	backend := &VulkanBackend{
		device:           device,
		kernelCache:      make(map[string]VulkanPipeline),
		optimizedKernels: &VulkanOptimizedKernels{},
	}
	backend.optimizedKernels.backend = backend
	return backend
}

// GroupedQueryAttention performs grouped query attention on the GPU via Vulkan compute shaders.
// Returns errNotImplemented until real Vulkan GQA kernels are implemented.
func (v *VulkanBackend) GroupedQueryAttention(ctx ml.Context, q, k, vTensor, o *ml.Tensor, group int) error {
	return fmt.Errorf("GroupedQueryAttention: %w", errNotImplemented)
}

// FusedMLP performs a fused MLP (up-projection → silu → down-projection) on the GPU via Vulkan.
// Returns errNotImplemented until real Vulkan MLP kernels are implemented.
func (v *VulkanBackend) FusedMLP(ctx ml.Context, up, gate, down *ml.Tensor) error {
	return fmt.Errorf("FusedMLP: %w", errNotImplemented)
}

// FlashAttention performs fused flash attention on the GPU via Vulkan compute shaders.
// Returns errNotImplemented until real Vulkan flash attention kernels are implemented.
func (v *VulkanBackend) FlashAttention(ctx ml.Context, q, k, vTensor, o *ml.Tensor, softmaxScale float32, isCausal bool) error {
	return fmt.Errorf("FlashAttention: %w", errNotImplemented)
}

// AllocateLayerMemory allocates GPU memory for a model layer via Vulkan.
// Returns errNotImplemented until real Vulkan memory allocation is implemented.
func (v *VulkanBackend) AllocateLayerMemory(layerIdx int, size uint64, hint VulkanMemoryHint) (*VulkanMemory, error) {
	return nil, fmt.Errorf("AllocateLayerMemory: %w", errNotImplemented)
}

// GetOrCreatePipeline retrieves a cached Vulkan compute pipeline or creates a new one from shaderData.
// Returns errNotImplemented until real Vulkan pipeline creation is implemented.
func (v *VulkanBackend) GetOrCreatePipeline(name string, shaderData []byte) (VulkanPipeline, error) {
	if pipeline, ok := v.kernelCache[name]; ok {
		return pipeline, nil
	}
	return nil, fmt.Errorf("GetOrCreatePipeline(%q): %w", name, errNotImplemented)
}

type VulkanMemoryHint struct {
	PreferredLocation  MemoryPropertyFlags
	ExplicitFlush      bool
	CoherentWithDevice bool
}

type MemoryPropertyFlags uint32

const (
	MemoryPropertyDeviceLocal     MemoryPropertyFlags = 0x00000001
	MemoryPropertyHostVisible     MemoryPropertyFlags = 0x00000002
	MemoryPropertyHostCoherent    MemoryPropertyFlags = 0x00000004
	MemoryPropertyHostCached      MemoryPropertyFlags = 0x00000008
	MemoryPropertyLazilyAllocated MemoryPropertyFlags = 0x00000010
)

func (v *VulkanBackend) Close() {
	v.kernelCache = make(map[string]VulkanPipeline)
}

type VulkanMemoryPoolAllocation struct {
	memory *VulkanMemory
	offset uint64
	size   uint64
}

func NewVulkanMemoryPool(device Device, maxMemory uint64) *VulkanMemoryPool {
	return &VulkanMemoryPool{
		device:       device,
		allocations:  make(map[uint64]*VulkanMemory),
		maxAllocated: maxMemory,
	}
}

// Allocate reserves GPU memory from the Vulkan memory pool.
// Returns nil until real Vulkan memory pool allocation is implemented.
func (p *VulkanMemoryPool) Allocate(size uint64, hint VulkanMemoryHint) (*VulkanMemoryPoolAllocation, error) {
	return nil, fmt.Errorf("VulkanMemoryPool.Allocate: %w", errNotImplemented)
}

func (p *VulkanMemoryPool) Deallocate(alloc *VulkanMemoryPoolAllocation) {
	_ = alloc
	// No-op until real allocation tracking is implemented.
}

func (p *VulkanMemoryPool) TotalAllocated() uint64 {
	return p.totalAllocated
}
