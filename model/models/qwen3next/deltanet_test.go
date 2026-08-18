package qwen3next

import "testing"

func TestGatedDeltaNetPackedElems(t *testing.T) {
	t.Parallel()
	// llama.cpp concat: output S*H*T*B + state S*S*H*B
	got := gatedDeltaNetPackedElems(128, 32, 6, 1)
	want := 128*32*6 + 128*128*32
	if got != want {
		t.Fatalf("packed elems = %d, want %d", got, want)
	}
	if gatedDeltaNetPackedElems(4, 2, 1, 1) != 4*2*1+4*4*2 {
		t.Fatal("AR packed size mismatch")
	}
}
