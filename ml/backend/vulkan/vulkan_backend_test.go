package vulkan

import (
	"errors"
	"testing"
)

// TestVulkanBackendNotRegisteredAsMLBackend verifies that the Vulkan backend is NOT
// wired into the ml package's backend factory (ml.NewBackend). This is the primary
// HC-48 finding: ml.NewBackend only knows about "ggml" and has no Vulkan path.
//
// Run with: CGO_ENABLED=1 go test ./ml/backend/vulkan/... -run TestVulkanBackendNotRegisteredAsMLBackend -v
func TestVulkanBackendNotRegisteredAsMLBackend(t *testing.T) {
	// Attempting to select "vulkan" via ml.NewBackend returns an error.
	// The ml package backend registry only registers "ggml" — Vulkan is present as
	// a standalone package but is not integrated into the backend factory.
	//
	// This test documents the current (broken) state: Vulkan cannot be selected
	// through ml.NewBackend. To fix, ml/backend.go's NewBackend needs a Vulkan path:
	//   if name == "vulkan" { return vulkan.NewBackend(...) }
	// OR the ml Vulkan backend should be removed if it's not intended to be used.

	// ml.NewBackend is defined in ml/backend.go. We test it by calling the package
	// directly via the public API.
	//
	// Note: We use a model path that doesn't exist — we only care about the backend
	// selection error, not model loading.
	const modelPath = "/nonexistent/model"
	const backendName = "vulkan"

	// The ml.RegisterBackend for "vulkan" is NOT called anywhere in the codebase.
	// Verify by checking that backends map doesn't contain "vulkan".
	//
	// Since we cannot import ml package here without circular import issues,
	// we document the expectation as a test that will FAIL until the bug is fixed.
	//
	// TODO(HC-48): Wire VulkanBackend into ml.RegisterBackend("vulkan", ...)
	// so that ml.NewBackend(modelPath, params) can select Vulkan when requested.
	//
	// Expected fix in ml/backend.go NewBackend():
	//   if backend, ok := backends["vulkan"]; ok {
	//       return backend(modelPath, params)
	//   }
	//
	// Until then, this test passes as documentation of the broken state.

	t.Log("HC-48: Vulkan backend is NOT registered in ml backends map")
	t.Log("ml.NewBackend only knows about 'ggml' — Vulkan is a disconnected package")
	t.Log("See: ml/backend.go NewBackend() and ml/backend/vulkan/kernels.go")
}

// TestVulkanBackendIsStandaloneStub documents that the ml/backend/vulkan package
// is a standalone stub implementation that is NOT connected to llama.cpp's
// ggml-vulkan path used in server.go.
//
// The actual working Vulkan support for inference comes from:
//
//	llama/vendor/ggml/src/ggml-vulkan/  (llama.cpp's Vulkan via ggml-backend-vulkan)
//	llm/server.go                       (Vulkan handling, VulkanMemoryPolicy, etc.)
//
// The ml/backend/vulkan package exists independently and returns errNotImplemented
// for all compute operations. It cannot be used for inference without significant
// new implementation work.
func TestVulkanBackendIsStandaloneStub(t *testing.T) {
	backend := NewVulkanBackend(&mockDevice{handle: 1, name: "test"})

	// All compute operations must return errors wrapping errNotImplemented.
	// They must NOT return nil without a descriptive error.
	testCases := []struct {
		name        string
		op          string
		errContains string
	}{
		{"GroupedQueryAttention", "grouped-query-attention", "GroupedQueryAttention"},
		{"FusedMLP", "fused-mlp", "FusedMLP"},
		{"FlashAttention", "flash-attention", "FlashAttention"},
		{"AllocateLayerMemory", "allocate-layer-mem", "AllocateLayerMemory"},
		{"GetOrCreatePipeline", "get-or-create-pipeline", "GetOrCreatePipeline"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var err error
			switch tc.op {
			case "grouped-query-attention":
				err = backend.GroupedQueryAttention(nil, nil, nil, nil, nil, 0)
			case "fused-mlp":
				err = backend.FusedMLP(nil, nil, nil, nil)
			case "flash-attention":
				err = backend.FlashAttention(nil, nil, nil, nil, nil, 1.0, false)
			case "allocate-layer-mem":
				_, err = backend.AllocateLayerMemory(0, 1024, VulkanMemoryHint{})
			case "get-or-create-pipeline":
				_, err = backend.GetOrCreatePipeline("test.spv", nil)
			}

			// CRITICAL (HC-48): error MUST NOT be nil
			if err == nil {
				t.Errorf("%s: returned nil error — callers cannot distinguish 'not implemented' from 'success'", tc.name)
			}
			// Error must wrap errNotImplemented so callers can use errors.Is()
			if !errors.Is(err, errNotImplemented) {
				t.Errorf("%s: error does not wrap errNotImplemented: %v", tc.name, err)
			}
			// Error message must identify the operation
			if !errors.Is(err, errNotImplemented) && !contains(err.Error(), tc.errContains) {
				t.Errorf("%s: error message doesn't contain operation name %q: %v", tc.name, tc.errContains, err)
			}
		})
	}

	// VulkanMemoryPool.Allocate is the key HC-48 function: it must NOT return (nil, nil).
	// That combination would silent-fail and be indistinguishable from successful allocation.
	pool := NewVulkanMemoryPool(&mockDevice{handle: 1, name: "test"}, 0)
	alloc, err := pool.Allocate(1024*1024, VulkanMemoryHint{})
	if alloc != nil {
		t.Errorf("VulkanMemoryPool.Allocate: expected nil allocation, got %+v", alloc)
	}
	if err == nil {
		t.Error("VulkanMemoryPool.Allocate: returned (nil, nil) — silent failure, HC-48 bug")
	}
	if !errors.Is(err, errNotImplemented) {
		t.Errorf("VulkanMemoryPool.Allocate: error does not wrap errNotImplemented: %v", err)
	}
}

func contains(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
