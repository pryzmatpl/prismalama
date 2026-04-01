//go:build integration

package integration

import (
	"testing"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
)

// TestShipLayerStreamingEnvDefault ensures layer streaming is opt-in.
func TestShipLayerStreamingEnvDefault(t *testing.T) {
	t.Setenv("OLLAMA_LAYER_STREAMING", "")
	if envconfig.LayerStreaming() {
		t.Fatal("unset OLLAMA_LAYER_STREAMING must default to false")
	}
}

func TestShipLayerStreamingEnvEnable(t *testing.T) {
	t.Setenv("OLLAMA_LAYER_STREAMING", "1")
	if !envconfig.LayerStreaming() {
		t.Fatal("OLLAMA_LAYER_STREAMING=1 must enable streaming")
	}
}

func TestShipStreamingBudgetDefault(t *testing.T) {
	t.Setenv("OLLAMA_STREAMING_BUDGET", "")
	if got := envconfig.StreamingBudgetBytes(); got != 4*format.GibiByte {
		t.Fatalf("default streaming budget: got %d, want %d", got, 4*format.GibiByte)
	}
}

func TestShipStreamingBudgetOverride(t *testing.T) {
	t.Setenv("OLLAMA_STREAMING_BUDGET", "8589934592")
	if got := envconfig.StreamingBudgetBytes(); got != 8*format.GibiByte {
		t.Fatalf("streaming budget override: got %d, want %d", got, 8*format.GibiByte)
	}
}
