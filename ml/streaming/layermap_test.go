package streaming

import (
	"bytes"
	"os"
	"testing"

	"github.com/ollama/ollama/fs/ggml"
)

func writeTestGGUF(t *testing.T) string {
	t.Helper()
	data := bytes.NewBuffer(make([]byte, 7*2*3*4)) // 7 tensors × 2×3 float32 = 168 bytes

	ts := []*ggml.Tensor{
		{Name: "token_embd.weight", Kind: uint32(ggml.TensorTypeF32), Shape: []uint64{2, 3}, WriterTo: data},
		{Name: "blk.0.attn_q.weight", Kind: uint32(ggml.TensorTypeF32), Shape: []uint64{2, 3}, WriterTo: data},
		{Name: "blk.0.attn_v.weight", Kind: uint32(ggml.TensorTypeF32), Shape: []uint64{2, 3}, WriterTo: data},
		{Name: "blk.1.attn_q.weight", Kind: uint32(ggml.TensorTypeF32), Shape: []uint64{2, 3}, WriterTo: data},
		{Name: "blk.1.attn_v.weight", Kind: uint32(ggml.TensorTypeF32), Shape: []uint64{2, 3}, WriterTo: data},
		{Name: "output_norm.weight", Kind: uint32(ggml.TensorTypeF32), Shape: []uint64{3, 2}, WriterTo: data},
		{Name: "output.weight", Kind: uint32(ggml.TensorTypeF32), Shape: []uint64{3, 2}, WriterTo: data},
	}

	f, err := os.CreateTemp(t.TempDir(), "test-*.gguf")
	if err != nil {
		t.Fatal(err)
	}

	kv := ggml.KV{
		"general.architecture": "test",
		"test.block_count":     uint32(2),
	}
	if err := ggml.WriteGGUF(f, kv, ts); err != nil {
		t.Fatal(err)
	}
	f.Close()
	return f.Name()
}

func decodeTestGGUF(t *testing.T, path string) *ggml.GGML {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	g, err := ggml.Decode(f, -1)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func TestBuildLayerMap_FullTensorNames(t *testing.T) {
	path := writeTestGGUF(t)
	g := decodeTestGGUF(t, path)

	lm, err := BuildLayerMap(g)
	if err != nil {
		t.Fatal(err)
	}

	if lm.BlockCount != 2 {
		t.Fatalf("block count: want 2, got %d", lm.BlockCount)
	}
	if len(lm.Layers) < 3 {
		t.Fatalf("want >= 3 layers (2 blocks + output), got %d", len(lm.Layers))
	}

	wantBlock0 := map[string]bool{
		"blk.0.attn_q.weight": true,
		"blk.0.attn_v.weight": true,
	}
	wantBlock1 := map[string]bool{
		"blk.1.attn_q.weight": true,
		"blk.1.attn_v.weight": true,
	}
	wantOutput := map[string]bool{
		"token_embd.weight":   true,
		"output_norm.weight":  true,
		"output.weight":       true,
	}

	assertLayerTensors(t, lm.Layers[0], "blk.0", wantBlock0)
	assertLayerTensors(t, lm.Layers[1], "blk.1", wantBlock1)
	assertLayerTensors(t, lm.Layers[2], "output", wantOutput)
}

func assertLayerTensors(t *testing.T, li LayerInfo, wantName string, wantTensors map[string]bool) {
	t.Helper()
	if li.Name != wantName {
		t.Errorf("layer name: want %q, got %q", wantName, li.Name)
	}
	if li.TensorCount != len(wantTensors) {
		t.Errorf("layer %s: want %d tensors, got %d", wantName, len(wantTensors), li.TensorCount)
	}
	for _, ref := range li.Tensors {
		if !wantTensors[ref.Name] {
			t.Errorf("layer %s: unexpected tensor %q (not a full GGUF name?)", wantName, ref.Name)
		}
		if ref.Size == 0 {
			t.Errorf("layer %s tensor %s: zero size", wantName, ref.Name)
		}
	}
}

func TestBuildLayerMap_ReadLayerTensors(t *testing.T) {
	path := writeTestGGUF(t)
	g := decodeTestGGUF(t, path)

	lm, err := BuildLayerMap(g)
	if err != nil {
		t.Fatal(err)
	}

	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	for i := range lm.Layers {
		tensors, err := lm.ReadLayerTensors(f, i)
		if err != nil {
			t.Fatalf("ReadLayerTensors(%d): %v", i, err)
		}
		if len(tensors) != lm.Layers[i].TensorCount {
			t.Fatalf("layer %d: read %d tensors, expected %d", i, len(tensors), lm.Layers[i].TensorCount)
		}
		for name, data := range tensors {
			if len(data) == 0 {
				t.Fatalf("layer %d tensor %s: empty data", i, name)
			}
		}
	}
}

func TestBuildLayerMap_NilGGML(t *testing.T) {
	_, err := BuildLayerMap(nil)
	if err == nil {
		t.Fatal("expected error for nil GGML")
	}
}

func TestLayerMap_FitsInBudget(t *testing.T) {
	path := writeTestGGUF(t)
	g := decodeTestGGUF(t, path)

	lm, err := BuildLayerMap(g)
	if err != nil {
		t.Fatal(err)
	}

	if got := lm.FitsInBudget(0); got != 0 {
		t.Fatalf("FitsInBudget(0): want 0, got %d", got)
	}
	if got := lm.FitsInBudget(lm.TotalWeightBytes); got != len(lm.Layers) {
		t.Fatalf("FitsInBudget(total): want %d, got %d", len(lm.Layers), got)
	}
}

func TestLayerMap_String(t *testing.T) {
	path := writeTestGGUF(t)
	g := decodeTestGGUF(t, path)

	lm, err := BuildLayerMap(g)
	if err != nil {
		t.Fatal(err)
	}

	s := lm.String()
	if s == "" {
		t.Fatal("String() returned empty")
	}
}
