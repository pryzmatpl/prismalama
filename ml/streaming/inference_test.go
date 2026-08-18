package streaming

import (
	"context"
	"os"
	"testing"
)

func TestNewInferenceStreamer(t *testing.T) {
	path := writeTestGGUF(t)
	g := decodeTestGGUF(t, path)

	lm, err := BuildLayerMap(g)
	if err != nil {
		t.Fatal(err)
	}

	is := NewInferenceStreamer(nil, path, lm)
	if is == nil {
		t.Fatal("expected non-nil InferenceStreamer")
	}
	if is.totalBlocks != 2 {
		t.Errorf("totalBlocks = %d, want 2", is.totalBlocks)
	}
	if is.currentBlock != -1 {
		t.Errorf("initial currentBlock = %d, want -1", is.currentBlock)
	}
}

func TestInferenceStreamerLayerMapIntegrity(t *testing.T) {
	path := writeTestGGUF(t)
	g := decodeTestGGUF(t, path)

	lm, err := BuildLayerMap(g)
	if err != nil {
		t.Fatal(err)
	}

	if len(lm.Layers) != 3 {
		t.Fatalf("expected 3 layers (2 blocks + output), got %d", len(lm.Layers))
	}

	for _, li := range lm.Layers {
		if len(li.Tensors) == 0 {
			t.Errorf("layer %q has no tensors", li.Name)
		}
		for _, ref := range li.Tensors {
			f, err := os.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			buf := make([]byte, ref.Size)
			_, err = f.ReadAt(buf, int64(lm.TensorDataBase+ref.Offset))
			f.Close()
			if err != nil {
				t.Errorf("failed to read tensor %s: %v", ref.Name, err)
			}
		}
	}
}

func TestInferenceStreamerPrepareForInference(t *testing.T) {
	path := writeTestGGUF(t)
	g := decodeTestGGUF(t, path)

	lm, err := BuildLayerMap(g)
	if err != nil {
		t.Fatal(err)
	}

	is := NewInferenceStreamer(nil, path, lm)
	defer is.Close()

	err = is.PrepareForInference(context.Background())
	if err == nil {
		t.Log("PrepareForInference succeeded (nil backend means no tensor loading)")
	}

	if is.file == nil {
		t.Error("expected file to be opened after PrepareForInference")
	}
}

func TestOnBlockDoneAdvancesBlocks(t *testing.T) {
	path := writeTestGGUF(t)
	g := decodeTestGGUF(t, path)

	lm, err := BuildLayerMap(g)
	if err != nil {
		t.Fatal(err)
	}

	is := NewInferenceStreamer(nil, path, lm)
	is.currentBlock = 0
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	is.file = f
	defer is.Close()

	cont := is.OnBlockDone(0)
	if !cont {
		t.Error("OnBlockDone(0) returned false, expected true to continue")
	}
	if is.currentBlock != 1 {
		t.Errorf("currentBlock = %d after OnBlockDone(0), want 1", is.currentBlock)
	}

	cont = is.OnBlockDone(1)
	if !cont {
		t.Error("OnBlockDone(1) returned false, expected true to continue")
	}
	if is.currentBlock != 2 {
		t.Errorf("currentBlock = %d after OnBlockDone(1), want 2", is.currentBlock)
	}
}

func TestOnBlockDoneKeepsBlocksWhenModelFitsBudget(t *testing.T) {
	path := writeTestGGUF(t)
	g := decodeTestGGUF(t, path)
	lm, err := BuildLayerMap(g)
	if err != nil {
		t.Fatal(err)
	}

	is := NewInferenceStreamer(nil, path, lm)
	is.SetBudgetBytes(1 << 40)
	is.loadedWeights[0] = true
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	is.file = f
	defer is.Close()

	if !is.OnBlockDone(0) {
		t.Fatal("OnBlockDone(0) returned false")
	}
	if !is.loadedWeights[0] {
		t.Fatal("expected block 0 to stay resident when the model fits the budget")
	}
}

func TestOnBlockDoneEvictsWhenOverBudget(t *testing.T) {
	path := writeTestGGUF(t)
	g := decodeTestGGUF(t, path)
	lm, err := BuildLayerMap(g)
	if err != nil {
		t.Fatal(err)
	}

	is := NewInferenceStreamer(nil, path, lm)
	is.SetBudgetBytes(1)
	is.loadedWeights[0] = true
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	is.file = f
	defer is.Close()

	if !is.OnBlockDone(0) {
		t.Fatal("OnBlockDone(0) returned false")
	}
	if is.loadedWeights[0] {
		t.Fatal("expected block 0 to be evicted when over budget")
	}
}

func TestSetupAndMapFromFile(t *testing.T) {
	path := writeTestGGUF(t)
	lm, err := MapFromFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if lm.BlockCount != 2 {
		t.Fatalf("BlockCount = %d want 2", lm.BlockCount)
	}

	is, err := Setup(context.Background(), nil, path, 1<<40)
	if err != nil {
		t.Fatal(err)
	}
	defer is.Close()
	if is.LayerMap() == nil || is.LayerMap().BlockCount != 2 {
		t.Fatal("Setup should retain LayerMap")
	}
	if is.budgetBytes != 1<<40 {
		t.Fatalf("budgetBytes = %d", is.budgetBytes)
	}
}
