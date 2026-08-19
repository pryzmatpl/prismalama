package ggml

/*
#cgo CPPFLAGS: -I${SRCDIR}/../../../llama/llama.cpp/ggml/include
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "ggml.h"
#include "ggml-backend.h"

extern _Bool goStreamingEvalCB(struct ggml_tensor *t, _Bool ask, uintptr_t handle);

static _Bool prismalama_eval_cb_trampoline(struct ggml_tensor *t, _Bool ask, void *ud) {
    return goStreamingEvalCB(t, ask, (uintptr_t)ud);
}

static inline void prismalama_set_eval_cb(ggml_backend_sched_t sched, uintptr_t handle) {
    ggml_backend_sched_set_eval_callback(sched,
        prismalama_eval_cb_trampoline, (void*)handle);
}

static inline void prismalama_clear_eval_cb(ggml_backend_sched_t sched) {
    ggml_backend_sched_set_eval_callback(sched, NULL, NULL);
}

static inline const char* prismalama_tensor_name_safe(struct ggml_tensor *t) {
    return t ? ggml_get_name(t) : "";
}

static inline struct ggml_tensor* prismalama_src(struct ggml_tensor *t, int i) {
    return (t && i >= 0 && i < GGML_MAX_SRC) ? t->src[i] : NULL;
}
*/
import "C"

import (
	"fmt"
	"log/slog"
	"regexp"
	"runtime/cgo"
	"sort"
	"strconv"

	"github.com/ollama/ollama/ml"
)

var blkRe = regexp.MustCompile(`blk\.(\d+)\.`)

// BlockBoundary records the last graph node index for a transformer block.
type BlockBoundary struct {
	Block       int
	LastNodeIdx int
	NodePtr     *C.struct_ggml_tensor
}

// ScanBlockBoundaries walks the context's graph and returns, for each block,
// the index and pointer of its last compute node. This is used to set the
// eval callback observation points for streaming inference.
func (c *Context) ScanBlockBoundaries() []BlockBoundary {
	if c.graph == nil {
		return nil
	}

	lastForBlock := make(map[int]BlockBoundary)
	nNodes := int(C.ggml_graph_n_nodes(c.graph))

	for i := 0; i < nNodes; i++ {
		node := C.ggml_graph_node(c.graph, C.int(i))
		if node == nil {
			continue
		}
		blk := nodeBlock(node)
		if blk >= 0 {
			lastForBlock[blk] = BlockBoundary{
				Block:       blk,
				LastNodeIdx: i,
				NodePtr:     node,
			}
		}
	}

	out := make([]BlockBoundary, 0, len(lastForBlock))
	for _, bb := range lastForBlock {
		out = append(out, bb)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Block < out[j].Block })

	slog.Info("streaming: scanned graph", "nodes", nNodes, "block_boundaries", len(out))
	return out
}

// nodeBlock returns the transformer block index that a graph node belongs to,
// or -1 for non-block nodes (embeddings, output).
func nodeBlock(node *C.struct_ggml_tensor) int {
	for i := 0; i < C.GGML_MAX_SRC; i++ {
		src := C.prismalama_src(node, C.int(i))
		if src == nil {
			continue
		}
		name := C.GoString(C.prismalama_tensor_name_safe(src))
		if m := blkRe.FindStringSubmatch(name); len(m) > 1 {
			if idx, err := strconv.Atoi(m[1]); err == nil {
				return idx
			}
		}
	}
	return -1
}

// --- Eval callback bridge ---

// StreamingEvalFn is called at block boundaries during graph compute.
// blockIdx is the block whose last node was just computed.
// The function should load the NEXT block's weights and optionally evict the current block.
// Return false to cancel the remaining graph compute.
type StreamingEvalFn func(blockIdx int) bool

type evalState struct {
	boundaryPtrs map[*C.struct_ggml_tensor]int
	onBoundary   StreamingEvalFn
}

//export goStreamingEvalCB
func goStreamingEvalCB(t *C.struct_ggml_tensor, ask C.bool, handle C.uintptr_t) C.bool {
	h := cgo.Handle(handle)
	st := h.Value().(*evalState)

	if bool(ask) {
		_, isBoundary := st.boundaryPtrs[t]
		return C.bool(isBoundary)
	}

	blk, ok := st.boundaryPtrs[t]
	if ok {
		return C.bool(st.onBoundary(blk))
	}
	return true
}

// PrepareStreamingCompute implements ml.StreamingComputeBackend.
func (b *Backend) PrepareStreamingCompute(ctx ml.Context, onBlockDone func(blockIdx int) bool) (int, func(), error) {
	gc, ok := ctx.(*Context)
	if !ok || gc.graph == nil {
		return 0, func() {}, fmt.Errorf("context has no graph for streaming compute")
	}
	boundaries := gc.ScanBlockBoundaries()
	if len(boundaries) == 0 {
		return 0, func() {}, nil
	}
	cleanup := b.setStreamingEvalCallback(boundaries, StreamingEvalFn(onBlockDone))
	return len(boundaries), cleanup, nil
}

// setStreamingEvalCallback installs an eval callback on the backend scheduler.
// At each block boundary (the last node of blk.N), the scheduler pauses compute,
// calls onBlockDone(N), then continues with the next block.
//
// Caller MUST call the returned cleanup function after Compute finishes.
func (b *Backend) setStreamingEvalCallback(boundaries []BlockBoundary, onBlockDone StreamingEvalFn) func() {
	ptrs := make(map[*C.struct_ggml_tensor]int, len(boundaries))
	for _, bb := range boundaries {
		ptrs[bb.NodePtr] = bb.Block
	}

	h := cgo.NewHandle(&evalState{boundaryPtrs: ptrs, onBoundary: onBlockDone})
	C.prismalama_set_eval_cb(b.sched, C.uintptr_t(h))

	return func() {
		C.prismalama_clear_eval_cb(b.sched)
		h.Delete()
	}
}
