package ggml

// #cgo CPPFLAGS: -I${SRCDIR}/../../../llama/llama.cpp/ggml/include
// #include <stdlib.h>
// #include "ggml.h"
// #include "ggml-backend.h"
// void ggml_streaming_backend_dev_reset(ggml_backend_dev_t device) { (void)device; }
import "C"

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
	"unicode"
	"unsafe"

	"github.com/ollama/ollama/format"
	fsggml "github.com/ollama/ollama/fs/ggml"
)

// LoadStreaming loads model weights layer-by-layer instead of concurrently.
// This bounds peak I/O and enables observable per-layer progress. The end state
// is identical to Load — all tensors populated — but the sequential pattern is
// NVMe-friendly and provides the foundation for runtime layer streaming.
func (b *Backend) LoadStreaming(ctx context.Context, progress func(float32)) error {
	if !b.allocMemory {
		return fmt.Errorf("cannot stream-load model without memory allocation")
	}

	b.logOffloadingInfo()

	file, err := os.Open(b.modelPath)
	if err != nil {
		return fmt.Errorf("open model: %w", err)
	}
	defer file.Close()

	tensorsByLayer := b.groupTensorsByLayer()
	totalBytes := uint64(b.meta.Length) - b.meta.Tensors().Offset
	var doneBytes atomic.Uint64

	reportProgress := func(n uint64) {
		if progress != nil {
			done := doneBytes.Add(n)
			progress(float32(done) / float32(totalBytes))
		}
	}

	slog.Info("streaming load: loading non-block tensors first")
	if err := b.loadTensorGroup(ctx, file, tensorsByLayer[-1], reportProgress); err != nil {
		return fmt.Errorf("load non-block tensors: %w", err)
	}

	blockCount := int(b.meta.KV().BlockCount())
	for i := 0; i < blockCount; i++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		tensors := tensorsByLayer[i]
		if len(tensors) == 0 {
			continue
		}
		if err := b.loadTensorGroup(ctx, file, tensors, reportProgress); err != nil {
			return fmt.Errorf("stream layer %d: %w", i, err)
		}
		slog.Info("streaming load: layer ready", "layer", i, "of", blockCount)
	}

	b.cleanupUnusedDevices()

	slog.Info("streaming load: complete", "layers", blockCount, "total_bytes", format.HumanBytes2(totalBytes))
	return nil
}

// logOffloadingInfo logs the same GPU/CPU offloading summary as Load.
func (b *Backend) logOffloadingInfo() {
	gpuLayers := 0
	for _, layer := range b.layers {
		switch C.ggml_backend_dev_type(layer.d) {
		case C.GGML_BACKEND_DEVICE_TYPE_GPU, C.GGML_BACKEND_DEVICE_TYPE_IGPU:
			gpuLayers++
		}
	}
	slog.Info(fmt.Sprintf("offloading %d repeating layers to GPU", gpuLayers))

	switch C.ggml_backend_dev_type(b.output) {
	case C.GGML_BACKEND_DEVICE_TYPE_CPU:
		slog.Info("offloading output layer to CPU")
	case C.GGML_BACKEND_DEVICE_TYPE_GPU, C.GGML_BACKEND_DEVICE_TYPE_IGPU:
		slog.Info("offloading output layer to GPU")
		gpuLayers++
	case C.GGML_BACKEND_DEVICE_TYPE_ACCEL:
		slog.Info("offloading output layer to ACCEL")
	}
	slog.Info(fmt.Sprintf("offloaded %d/%d compute layers to GPU", gpuLayers, len(b.layers)+1))
	if b.moeExpertCPU > 0 {
		slog.Info("MoE split: attn/GDN on GPU; routed experts partially on CPU",
			"expert_layers_cpu", b.moeExpertCPU,
			"repeating_layers", len(b.layers))
	}
}

// cleanupUnusedDevices frees backend state for devices not assigned to the scheduler.
func (b *Backend) cleanupUnusedDevices() {
	for _, d := range append(gpus, append(accels, cpus...)...) {
		used := false
		for _, backend := range b.schedBackends {
			if d == C.ggml_backend_get_device(backend) {
				used = true
				break
			}
		}
		if !used {
			C.ggml_streaming_backend_dev_reset(d)
		}
	}
}

// groupTensorsByLayer classifies GGUF tensors by their block index. Non-block
// tensors (embeddings, norms, output) are grouped under key -1.
func (b *Backend) groupTensorsByLayer() map[int][]*fsggml.Tensor {
	groups := make(map[int][]*fsggml.Tensor)
	for _, t := range b.meta.Tensors().Items() {
		idx := parseTensorBlockIndex(t.Name)
		groups[idx] = append(groups[idx], t)
	}
	return groups
}

// parseTensorBlockIndex extracts the block number from a GGUF tensor name.
// "blk.5.attn_q.weight" → 5, "output_norm.weight" → -1.
func parseTensorBlockIndex(name string) int {
	fields := strings.FieldsFunc(name, func(r rune) bool { return !unicode.IsNumber(r) })
	if len(fields) > 0 {
		if i, err := strconv.Atoi(fields[0]); err == nil && strings.Contains(name, "blk.") {
			return i
		}
	}
	return -1
}

// loadTensorGroup loads a set of GGUF tensors sequentially from file.
// Handles MXFP4 byte transforms and BF16→FP32 conversions.
func (b *Backend) loadTensorGroup(
	ctx context.Context,
	file *os.File,
	tensors []*fsggml.Tensor,
	reportProgress func(uint64),
) error {
	for _, t := range tensors {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := b.loadOneTensor(file, t, reportProgress); err != nil {
			return err
		}
	}
	return nil
}

// loadOneTensor reads a single tensor from the GGUF file and writes it into
// the GGML backend buffer(s), applying format transforms as needed.
func (b *Backend) loadOneTensor(
	file *os.File,
	t *fsggml.Tensor,
	reportProgress func(uint64),
) error {
	targets := b.resolveTensorTargets(t.Name)
	if len(targets) == 0 {
		return fmt.Errorf("unassigned tensor: %s", t.Name)
	}

	sr := io.NewSectionReader(file, int64(b.meta.Tensors().Offset+t.Offset), int64(t.Size()))

	if t.Kind == 4 && targets[0]._type == 39 {
		return b.loadMXFP4Tensor(sr, t, targets, reportProgress)
	}
	if strings.HasSuffix(t.Name, "_exps.bias") && t.Kind == 30 && targets[0]._type == 0 {
		return b.loadBF16ToFP32Tensor(sr, t, targets, reportProgress)
	}
	return b.loadRawTensor(sr, t, targets, reportProgress)
}

func (b *Backend) resolveTensorTargets(name string) []*C.struct_ggml_tensor {
	mapping := b.tensorLoadTargets[name]
	tts := make([]*C.struct_ggml_tensor, max(1, len(mapping)))
	for i := range tts {
		target := ""
		if i < len(mapping) {
			target = mapping[i]
		}
		if target == "" {
			target = name
		}
		tt, ok := b.tensors[target]
		if !ok {
			return nil
		}
		tts[i] = tt
	}
	return tts
}

func (b *Backend) loadRawTensor(
	sr *io.SectionReader,
	t *fsggml.Tensor,
	tts []*C.struct_ggml_tensor,
	reportProgress func(uint64),
) error {
	bts := make([]byte, 1*format.MebiByte)
	var s uint64
	for s < t.Size() {
		n, err := io.ReadFull(sr, bts[:min(len(bts), int(t.Size()-s))])
		if err != nil {
			return fmt.Errorf("read %s at offset %d: %w", t.Name, s, err)
		}
		for _, tt := range tts {
			C.ggml_backend_tensor_set(tt, unsafe.Pointer(&bts[0]), C.size_t(s), C.size_t(n))
		}
		s += uint64(n)
		reportProgress(uint64(n))
	}
	return nil
}

func (b *Backend) loadMXFP4Tensor(
	sr *io.SectionReader,
	t *fsggml.Tensor,
	tts []*C.struct_ggml_tensor,
	reportProgress func(uint64),
) error {
	const BS = 17
	bts := make([]byte, 8*BS*format.KibiByte)
	var s uint64
	var tmp [16]byte
	for s < t.Size() {
		n, err := io.ReadFull(sr, bts[:min(len(bts), int(t.Size()-s))])
		if err != nil {
			return fmt.Errorf("read MXFP4 %s: %w", t.Name, err)
		}
		for j := range n / BS {
			for i := 1; i < 9; i++ {
				a, b := bts[j*BS+i], bts[j*BS+i+8]
				tmp[2*(i-1)] = (a & 0x0F) | (b << 4)
				tmp[2*(i-1)+1] = (a >> 4) | (b & 0xF0)
			}
			copy(bts[j*BS+1:j*BS+17], tmp[:])
		}
		for _, tt := range tts {
			C.ggml_backend_tensor_set(tt, unsafe.Pointer(&bts[0]), C.size_t(s), C.size_t(n))
		}
		s += uint64(n)
		reportProgress(uint64(n))
	}
	return nil
}

func (b *Backend) loadBF16ToFP32Tensor(
	sr *io.SectionReader,
	t *fsggml.Tensor,
	tts []*C.struct_ggml_tensor,
	reportProgress func(uint64),
) error {
	bts := make([]byte, 128*format.KibiByte)
	var e uint64
	for e < t.Elements() {
		n, err := io.ReadFull(sr, bts[:min(len(bts), int(t.Elements()-e)*2)])
		if err != nil {
			return fmt.Errorf("read BF16 %s: %w", t.Name, err)
		}
		fp32 := ConvertToF32(bts, uint32(fsggml.TensorTypeBF16), uint64(n/2))
		for _, tt := range tts {
			C.ggml_backend_tensor_set(tt, unsafe.Pointer(&fp32[0]), C.size_t(e*4), C.size_t(n*2))
		}
		e += uint64(n / 2)
		reportProgress(uint64(n))
	}
	return nil
}
