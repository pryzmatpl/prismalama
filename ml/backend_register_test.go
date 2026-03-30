package ml

import (
	"testing"
)

// TestOnlyGGMLBackendIsRegistered verifies that only the "ggml" backend is
// registered in the ml backend registry. This documents the current state where
// "vulkan" is not a selectable backend through ml.NewBackend.
//
// Run with: CGO_ENABLED=1 go test ./ml/... -run TestOnlyGGMLBackendIsRegistered -v
func TestOnlyGGMLBackendIsRegistered(t *testing.T) {
	registeredBackends := make([]string, 0, len(backends))
	for name := range backends {
		registeredBackends = append(registeredBackends, name)
	}

	if len(registeredBackends) != 1 {
		t.Errorf("expected exactly 1 registered backend (ggml), got %d: %v", len(registeredBackends), registeredBackends)
	}
	if len(registeredBackends) > 0 && registeredBackends[0] != "ggml" {
		t.Errorf("expected registered backend to be 'ggml', got %q", registeredBackends[0])
	}

	// Verify Vulkan is NOT registered (HC-48)
	for _, name := range registeredBackends {
		if name == "vulkan" {
			t.Error("Vulkan backend should NOT be registered (HC-48: not wired into ml.NewBackend)")
		}
	}
}
