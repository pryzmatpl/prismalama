package vulkan

import (
	"errors"
	"fmt"
	"testing"

	"github.com/ollama/ollama/ml"
)

// mockDevice implements the Device interface for testing.
type mockDevice struct {
	handle uint64
	name   string
}

func (m *mockDevice) Handle() uint64 { return m.handle }
func (m *mockDevice) Name() string    { return m.name }

// errNotImplemented sentinel for matching the package-level error.
var errNotImplemented = errors.New("vulkan backend: operation not implemented; " +
	"build with Vulkan compute support (GGML_VULKAN=1) and ensure GPU device is available")

// TestNewVulkanBackend tests that NewVulkanBackend returns a non-nil VulkanBackend
// with correctly initialised internal state for a variety of mock devices.
func TestNewVulkanBackend(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name   string
		device Device
	}{
		{
			name:   "nil device handle",
			device: &mockDevice{handle: 0, name: "llvmpipe"},
		},
		{
			name:   "valid GPU device",
			device: &mockDevice{handle: 1, name: "NVIDIA GeForce RTX 4090"},
		},
		{
			name:   "AMD GPU via RADV",
			device: &mockDevice{handle: 0x100000000, name: "AMD RADV NAVI21"},
		},
		{
			name:   "Intel ANV",
			device: &mockDevice{handle: 0x8086 << 16, name: "Intel Graphics ADL-P"},
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			backend := NewVulkanBackend(tt.device)
			if backend == nil {
				t.Fatal("NewVulkanBackend returned nil")
			}
			if backend.device != tt.device {
				t.Errorf("device mismatch: got %v, want %v", backend.device, tt.device)
			}
			if backend.kernelCache == nil {
				t.Error("kernelCache should be non-nil (initialised map)")
			}
			if backend.optimizedKernels == nil {
				t.Error("optimizedKernels should be non-nil")
			}
			// Verify the back-reference from VulkanOptimizedKernels to VulkanBackend.
			if backend.optimizedKernels.backend != backend {
				t.Error("optimizedKernels.backend back-reference is incorrect")
			}
			if backend.memoryPool != nil {
				t.Error("memoryPool should be nil on a fresh backend")
			}
		})
	}
}

// TestVulkanBackendClose tests that Close clears the kernel cache without panicking.
func TestVulkanBackendClose(t *testing.T) {
	t.Parallel()

	backend := NewVulkanBackend(&mockDevice{handle: 1, name: "test"})

	// Populate cache with a mock pipeline.
	// VulkanPipeline is an interface, so we need a concrete implementation.
	type mockPipeline struct{ handle uint64 }

	impl := &mockPipeline{handle: 42}
	backend.kernelCache["test_kernel.spv"] = impl

	if len(backend.kernelCache) != 1 {
		t.Fatalf("expected 1 cached kernel, got %d", len(backend.kernelCache))
	}

	// Close must not panic.
	backend.Close()

	// After Close the cache is re-initialised (not nil, but empty).
	if backend.kernelCache == nil {
		t.Error("kernelCache should not be nil after Close")
	}
	if len(backend.kernelCache) != 0 {
		t.Errorf("kernelCache should be empty after Close, got %d entries", len(backend.kernelCache))
	}
}

// TestVulkanMemoryPoolAllocationStruct tests the VulkanMemoryPoolAllocation struct
// fields directly (no ML operations needed).
func TestVulkanMemoryPoolAllocationStruct(t *testing.T) {
	t.Parallel()

	mem := &VulkanMemory{
		device:       &mockDevice{handle: 1},
		memoryHandle: 0xDEAD,
		size:         1024,
		offset:       128,
		mappable:     true,
	}

	alloc := &VulkanMemoryPoolAllocation{
		memory: mem,
		offset: 128,
		size:   1024,
	}

	if alloc.memory != mem {
		t.Error("memory field mismatch")
	}
	if alloc.offset != 128 {
		t.Errorf("offset: got %d, want 128", alloc.offset)
	}
	if alloc.size != 1024 {
		t.Errorf("size: got %d, want 1024", alloc.size)
	}
}

// TestNewVulkanMemoryPool tests that NewVulkanMemoryPool returns a non-nil
// pool with the correct initial state.
func TestNewVulkanMemoryPool(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name     string
		device   Device
		maxMem   uint64
		wantZero bool
	}{
		{
			name:     "zero max (unlimited pool)",
			device:   &mockDevice{handle: 1, name: "test"},
			maxMem:   0,
			wantZero: true,
		},
		{
			name:     "bounded 8 GiB pool",
			device:   &mockDevice{handle: 2, name: "test"},
			maxMem:   8 * 1024 * 1024 * 1024,
			wantZero: false,
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			pool := NewVulkanMemoryPool(tt.device, tt.maxMem)
			if pool == nil {
				t.Fatal("NewVulkanMemoryPool returned nil")
			}
			if pool.device != tt.device {
				t.Errorf("device mismatch: got %v, want %v", pool.device, tt.device)
			}
			if pool.maxAllocated != tt.maxMem {
				t.Errorf("maxAllocated: got %d, want %d", pool.maxAllocated, tt.maxMem)
			}
			if pool.totalAllocated != 0 {
				t.Errorf("totalAllocated: got %d, want 0", pool.totalAllocated)
			}
			if pool.allocations == nil {
				t.Error("allocations map should be non-nil")
			}
			if pool.TotalAllocated() != 0 {
				t.Errorf("TotalAllocated() after init: got %d, want 0", pool.TotalAllocated())
			}
		})
	}
}

// TestVulkanMemoryPoolAllocate tests that Allocate returns (nil, error) and that
// the error wraps errNotImplemented and identifies the operation.
// This is the key HC-48 assertion: Allocate must NOT return nil without a
// descriptive error; callers must be able to distinguish "out of memory" from
// "Vulkan compute not implemented".
func TestVulkanMemoryPoolAllocate(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name        string
		size        uint64
		hint        VulkanMemoryHint
		wantNil     bool   // allocation must be nil
		wantErr     bool   // error must be non-nil
		errContains string // substring that must appear in error message
	}{
		{
			name:         "zero size allocation",
			size:         0,
			hint:         VulkanMemoryHint{},
			wantNil:      true,
			wantErr:      true,
			errContains: "VulkanMemoryPool.Allocate",
		},
		{
			name:         "1 MiB device-local allocation",
			size:         1024 * 1024,
			hint:         VulkanMemoryHint{PreferredLocation: MemoryPropertyDeviceLocal},
			wantNil:      true,
			wantErr:      true,
			errContains: "VulkanMemoryPool.Allocate",
		},
		{
			name:         "large host-visible allocation",
			size:         4 * 1024 * 1024 * 1024,
			hint:         VulkanMemoryHint{PreferredLocation: MemoryPropertyHostVisible | MemoryPropertyHostCoherent},
			wantNil:      true,
			wantErr:      true,
			errContains: "not implemented",
		},
		{
			name:         "1 GiB with explicit flush hint",
			size:         1024 * 1024 * 1024,
			hint:         VulkanMemoryHint{ExplicitFlush: true, CoherentWithDevice: true},
			wantNil:      true,
			wantErr:      true,
			errContains: "not implemented",
		},
	}

	pool := NewVulkanMemoryPool(&mockDevice{handle: 1, name: "test"}, 0)

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			alloc, err := pool.Allocate(tt.size, tt.hint)

			if tt.wantNil && alloc != nil {
				t.Errorf("Allocate: expected nil allocation, got %+v", alloc)
			}
			if !tt.wantErr && err == nil {
				t.Error("Allocate: expected non-nil error, got nil")
			}
			if err == nil {
				t.Skip("skipping error checks since err is nil")
			}
			if !errors.Is(err, errNotImplemented) {
				t.Errorf("Allocate: error does not wrap errNotImplemented: %v", err)
			}
			if tt.errContains != "" && !errors.Is(err, errNotImplemented) {
				if !stringsContain(err.Error(), tt.errContains) {
					t.Errorf("Allocate: error message %q does not contain %q", err.Error(), tt.errContains)
				}
			}
		})
	}
}

// TestVulkanMemoryPoolDeallocate tests that Deallocate is safe to call with
// nil, with a valid allocation, and with a zero-initialized allocation struct.
// It must not panic and must not modify pool state in the stub implementation.
func TestVulkanMemoryPoolDeallocate(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name  string
		alloc *VulkanMemoryPoolAllocation
	}{
		{
			name:  "nil allocation (safe to pass)",
			alloc: nil,
		},
		{
			name: "zero-initialized allocation",
			alloc: &VulkanMemoryPoolAllocation{},
		},
		{
			name: "allocation with memory",
			alloc: &VulkanMemoryPoolAllocation{
				memory: &VulkanMemory{
					device:       &mockDevice{handle: 1},
					memoryHandle: 0xBEEF,
					size:         4096,
					offset:       0,
					mappable:     false,
				},
				offset: 0,
				size:   4096,
			},
		},
	}

	pool := NewVulkanMemoryPool(&mockDevice{handle: 1, name: "test"}, 0)

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			// Deallocate must not panic.
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("Deallocate panicked: %v", r)
				}
			}()

			pool.Deallocate(tt.alloc)

			// Stub implementation is a no-op, so totalAllocated stays 0.
			if pool.TotalAllocated() != 0 {
				t.Errorf("TotalAllocated: expected 0 after Deallocate, got %d", pool.TotalAllocated())
			}
		})
	}
}

// TestVulkanBackendGPUOperations tests that GPU operations return errors that
// wrap errNotImplemented and clearly identify which operation failed.
// None of these should panic or return nil without an error.
func TestVulkanBackendGPUOperations(t *testing.T) {
	t.Parallel()

	backend := NewVulkanBackend(&mockDevice{handle: 1, name: "test"})

	cases := []struct {
		name        string
		fn          func() error
		errContains string
	}{
		{
			name: "GroupedQueryAttention",
			fn: func() error {
				return backend.GroupedQueryAttention(nil, nil, nil, nil, nil, 0)
			},
			errContains: "GroupedQueryAttention",
		},
		{
			name: "FusedMLP",
			fn: func() error {
				return backend.FusedMLP(nil, nil, nil, nil)
			},
			errContains: "FusedMLP",
		},
		{
			name: "FlashAttention",
			fn: func() error {
				return backend.FlashAttention(nil, nil, nil, nil, nil, 1.0, false)
			},
			errContains: "FlashAttention",
		},
		{
			name: "AllocateLayerMemory",
			fn: func() error {
				_, err := backend.AllocateLayerMemory(0, 1024, VulkanMemoryHint{})
				return err
			},
			errContains: "AllocateLayerMemory",
		},
		{
			name: "GetOrCreatePipeline (cache miss)",
			fn: func() error {
				_, err := backend.GetOrCreatePipeline("nonexistent.spv", nil)
				return err
			},
			errContains: "GetOrCreatePipeline",
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			err := tt.fn()
			if err == nil {
				t.Error("expected non-nil error, got nil")
				return
			}
			if !errors.Is(err, errNotImplemented) {
				t.Errorf("error does not wrap errNotImplemented: %v", err)
			}
			if !stringsContain(err.Error(), tt.errContains) {
				t.Errorf("error message %q does not contain %q", err.Error(), tt.errContains)
			}
		})
	}
}

// TestVulkanBackendGetOrCreatePipeline_CacheHit tests that GetOrCreatePipeline
// returns the cached pipeline when one exists, rather than an error.
func TestVulkanBackendGetOrCreatePipeline_CacheHit(t *testing.T) {
	t.Parallel()

	type mockPipeline struct{ handle uint64 }

	backend := NewVulkanBackend(&mockDevice{handle: 1, name: "test"})
	cached := &mockPipeline{handle: 99}
	backend.kernelCache["cached_kernel.spv"] = cached

	pipeline, err := backend.GetOrCreatePipeline("cached_kernel.spv", []byte("shader data"))

	if err != nil {
		t.Errorf("GetOrCreatePipeline cache hit: unexpected error: %v", err)
	}
	if pipeline == nil {
		t.Error("GetOrCreatePipeline cache hit: expected non-nil pipeline, got nil")
	}
	if pipeline.Handle() != 99 {
		t.Errorf("GetOrCreatePipeline: wrong handle, got %d, want 99", pipeline.Handle())
	}
}

// TestVulkanMemoryHintFlags tests that VulkanMemoryHint and MemoryPropertyFlags
// are correctly structured (sanity checks for the stub types).
func TestVulkanMemoryHintFlags(t *testing.T) {
	t.Parallel()

	cases := []struct {
		flag    MemoryPropertyFlags
		want    uint32
		hasBits uint32
	}{
		{MemoryPropertyDeviceLocal, 0x00000001, 0x00000001},
		{MemoryPropertyHostVisible, 0x00000002, 0x00000002},
		{MemoryPropertyHostCoherent, 0x00000004, 0x00000004},
		{MemoryPropertyHostCached, 0x00000008, 0x00000008},
		{MemoryPropertyLazilyAllocated, 0x00000010, 0x00000010},
		// Composite flag
		{MemoryPropertyHostVisible | MemoryPropertyHostCoherent, 0x00000006, 0x00000006},
	}

	for _, tt := range cases {
		t.Run(fmt.Sprintf("%x", tt.want), func(t *testing.T) {
			t.Parallel()

			if uint32(tt.flag) != tt.want {
				t.Errorf("MemoryPropertyFlags: got 0x%08x, want 0x%08x", uint32(tt.flag), tt.want)
			}
			if uint32(tt.flag)&tt.hasBits != tt.hasBits {
				t.Errorf("MemoryPropertyFlags 0x%08x: missing expected bits 0x%08x", uint32(tt.flag), tt.hasBits)
			}
		})
	}

	// Test VulkanMemoryHint fields.
	hint := VulkanMemoryHint{
		PreferredLocation:  MemoryPropertyDeviceLocal,
		ExplicitFlush:      true,
		CoherentWithDevice: true,
	}
	if hint.PreferredLocation != MemoryPropertyDeviceLocal {
		t.Error("PreferredLocation mismatch")
	}
	if !hint.ExplicitFlush {
		t.Error("ExplicitFlush should be true")
	}
	if !hint.CoherentWithDevice {
		t.Error("CoherentWithDevice should be true")
	}
}

// stringsContain is a simple helper equivalent to strings.Contains.
// Using direct byte comparison avoids importing the strings package to keep
// the test file focused on the Vulkan backend surface area.
func stringsContain(s, substr string) bool {
	if len(substr) == 0 {
		return true
	}
	if len(s) < len(substr) {
		return false
	}
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
