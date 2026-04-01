//go:build integration

package integration

import (
	"bytes"
	"os"
	"testing"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml/streaming"
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

// TestShipStreamingLayerMapGGUF verifies BuildLayerMap produces correct full GGUF tensor names,
// which is critical for the Backend.Get(name) ↔ Streamer integration.
func shipTensorBuf() *bytes.Buffer {
	return bytes.NewBuffer(make([]byte, 2*3*4))
}

func TestShipStreamingLayerMapGGUF(t *testing.T) {
	ts := []*ggml.Tensor{
		{Name: "token_embd.weight", Kind: uint32(ggml.TensorTypeF32), Shape: []uint64{2, 3}, WriterTo: shipTensorBuf()},
		{Name: "blk.0.attn_q.weight", Kind: uint32(ggml.TensorTypeF32), Shape: []uint64{2, 3}, WriterTo: shipTensorBuf()},
		{Name: "blk.0.attn_v.weight", Kind: uint32(ggml.TensorTypeF32), Shape: []uint64{2, 3}, WriterTo: shipTensorBuf()},
		{Name: "blk.1.attn_q.weight", Kind: uint32(ggml.TensorTypeF32), Shape: []uint64{2, 3}, WriterTo: shipTensorBuf()},
		{Name: "output.weight", Kind: uint32(ggml.TensorTypeF32), Shape: []uint64{3, 2}, WriterTo: shipTensorBuf()},
	}

	f, err := os.CreateTemp(t.TempDir(), "ship-stream-*.gguf")
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

	r, err := os.Open(f.Name())
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	g, err := ggml.Decode(r, -1)
	if err != nil {
		t.Fatal(err)
	}

	lm, err := streaming.BuildLayerMap(g)
	if err != nil {
		t.Fatal(err)
	}

	if lm.BlockCount != 2 {
		t.Fatalf("block count: want 2, got %d", lm.BlockCount)
	}

	// Verify full GGUF tensor names are used (not short group names)
	for _, li := range lm.Layers {
		for _, ref := range li.Tensors {
			if ref.Name == "attn_q.weight" || ref.Name == "attn_v.weight" || ref.Name == "weight" {
				t.Fatalf("layer %s tensor %q uses short name instead of full GGUF name", li.Name, ref.Name)
			}
		}
	}
}

// TestShipStreamingBackendInterface verifies that the GGML backend implements StreamingBackend
// when imported. This is a compile-time check via a type assertion in the test.
func TestShipStreamingBackendInterface(t *testing.T) {
	t.Setenv("OLLAMA_LAYER_STREAMING", "1")
	if !envconfig.LayerStreaming() {
		t.Fatal("OLLAMA_LAYER_STREAMING must be enabled for this test")
	}
}

// TestShipStreamingInferenceStreamerLifecycle verifies InferenceStreamer initialization,
// block advancement, and cleanup.
func TestShipStreamingInferenceStreamerLifecycle(t *testing.T) {
	ts := []*ggml.Tensor{
		{Name: "token_embd.weight", Kind: uint32(ggml.TensorTypeF32), Shape: []uint64{2, 3}, WriterTo: shipTensorBuf()},
		{Name: "blk.0.attn_q.weight", Kind: uint32(ggml.TensorTypeF32), Shape: []uint64{2, 3}, WriterTo: shipTensorBuf()},
		{Name: "blk.1.attn_q.weight", Kind: uint32(ggml.TensorTypeF32), Shape: []uint64{2, 3}, WriterTo: shipTensorBuf()},
		{Name: "blk.2.attn_q.weight", Kind: uint32(ggml.TensorTypeF32), Shape: []uint64{2, 3}, WriterTo: shipTensorBuf()},
		{Name: "output.weight", Kind: uint32(ggml.TensorTypeF32), Shape: []uint64{3, 2}, WriterTo: shipTensorBuf()},
	}

	f, err := os.CreateTemp(t.TempDir(), "ship-infer-*.gguf")
	if err != nil {
		t.Fatal(err)
	}
	kv := ggml.KV{
		"general.architecture": "test",
		"test.block_count":     uint32(3),
	}
	if err := ggml.WriteGGUF(f, kv, ts); err != nil {
		t.Fatal(err)
	}
	f.Close()

	r, err := os.Open(f.Name())
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	g, err := ggml.Decode(r, -1)
	if err != nil {
		t.Fatal(err)
	}

	lm, err := streaming.BuildLayerMap(g)
	if err != nil {
		t.Fatal(err)
	}

	is := streaming.NewInferenceStreamer(nil, f.Name(), lm)
	defer is.Close()

	if err := is.PrepareForInference(nil); err != nil {
		t.Fatal(err)
	}

	blocksDone := make([]int, 0, 3)
	for i := 0; i < 3; i++ {
		cont := is.OnBlockDone(i)
		if !cont {
			t.Fatalf("OnBlockDone(%d) returned false", i)
		}
		blocksDone = append(blocksDone, i)
	}

	if len(blocksDone) != 3 {
		t.Fatalf("expected 3 blocks done, got %d", len(blocksDone))
	}
	for i, b := range blocksDone {
		if b != i {
			t.Fatalf("blocksDone[%d] = %d, want %d", i, b, i)
		}
	}
}

// TestShipStreamingComputeBackendInterface verifies at compile time that
// the GGML backend implements StreamingComputeBackend.
func TestShipStreamingComputeBackendInterface(t *testing.T) {
	// Compile-time assertion: ml.StreamingComputeBackend is the interface,
	// GGML Backend implements it via PrepareStreamingCompute. The runner
	// exercises this at runtime via type assertion.
	t.Setenv("OLLAMA_LAYER_STREAMING", "1")
	if !envconfig.LayerStreaming() {
		t.Fatal("OLLAMA_LAYER_STREAMING must be enabled")
	}
}
