package llm

import (
	"testing"

	"github.com/ollama/ollama/ml"
)

func TestAdaptiveUsableBytes(t *testing.T) {
	u := adaptiveUsableBytes(ml.SystemInfo{FreeMemory: 12 * 1024 * 1024 * 1024, FreeSwap: 4 * 1024 * 1024 * 1024})
	if u == 0 || u > 12*1024*1024*1024 {
		t.Fatalf("unexpected usable: %d", u)
	}
}
