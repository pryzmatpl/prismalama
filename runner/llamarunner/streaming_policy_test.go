package llamarunner

import (
	"testing"
)

func TestUseMmapWithLayerStreaming(t *testing.T) {
	t.Parallel()
	if useMmapWithLayerStreaming(false, 0, true) != true {
		t.Fatal("layer streaming should force mmap when no LoRA")
	}
	if useMmapWithLayerStreaming(true, 1, true) != false {
		t.Fatal("LoRA must still disable mmap")
	}
	if useMmapWithLayerStreaming(true, 0, false) != true {
		t.Fatal("requested mmap should pass through")
	}
	if useMmapWithLayerStreaming(false, 0, false) != false {
		t.Fatal("no streaming and no request should leave mmap off")
	}
}
