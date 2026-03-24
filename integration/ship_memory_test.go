//go:build integration

package integration

import (
	"testing"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
)

// TestShipMemoryPolicyEnv locks the Prismalama default: unset OLLAMA_MEMORY_POLICY must
// mean performance-tier VRAM defaults (snappy small/medium models). Balanced is explicit.
func TestShipMemoryPolicyEnv(t *testing.T) {
	cases := []struct {
		set  string
		want string
	}{
		{"", "performance"},
		{"balanced", "balanced"},
		{"performance", "performance"},
		{"BALANCED", "balanced"},
	}
	for _, tc := range cases {
		t.Run(tc.set, func(t *testing.T) {
			t.Setenv("OLLAMA_MEMORY_POLICY", tc.set)
			if g := envconfig.MemoryPolicy(); g != tc.want {
				t.Fatalf("OLLAMA_MEMORY_POLICY=%q: got %q want %q", tc.set, g, tc.want)
			}
		})
	}
}

// TestShipAdaptiveMemoryEnv ensures load-time num_ctx adaptation defaults on and can be disabled.
func TestShipAdaptiveMemoryEnv(t *testing.T) {
	t.Setenv("OLLAMA_ADAPTIVE_MEMORY", "")
	if !envconfig.AdaptiveMemory(true) {
		t.Fatal("unset OLLAMA_ADAPTIVE_MEMORY must default to true (adapt when RAM tight)")
	}
	t.Setenv("OLLAMA_ADAPTIVE_MEMORY", "false")
	if envconfig.AdaptiveMemory(true) {
		t.Fatal("OLLAMA_ADAPTIVE_MEMORY=false must disable adaptive clamp")
	}
	t.Setenv("OLLAMA_ADAPTIVE_MEMORY", "true")
	if !envconfig.AdaptiveMemory(true) {
		t.Fatal("OLLAMA_ADAPTIVE_MEMORY=true must enable adaptive clamp")
	}
}

// TestShipGpuOverheadDefault reserves 2 GiB per GPU for compositor/desktop unless disabled.
func TestShipGpuOverheadDefault(t *testing.T) {
	t.Setenv("OLLAMA_GPU_OVERHEAD", "")
	if g := envconfig.GpuOverhead(); g != 3*format.GibiByte {
		t.Fatalf("unset OLLAMA_GPU_OVERHEAD must default to 3 GiB, got %d", g)
	}
	t.Setenv("OLLAMA_GPU_OVERHEAD", "0")
	if envconfig.GpuOverhead() != 0 {
		t.Fatal("OLLAMA_GPU_OVERHEAD=0 must disable reserved VRAM")
	}
}

func TestShipVulkanMmapDefault(t *testing.T) {
	t.Setenv("OLLAMA_VULKAN_MMAP", "")
	if !envconfig.VulkanMmap(true) {
		t.Fatal("unset OLLAMA_VULKAN_MMAP must default to true")
	}
}
