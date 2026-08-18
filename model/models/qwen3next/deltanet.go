package qwen3next

import (
	"errors"
	"log/slog"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

// convKernel wraps the 1D convolution kernel tensor
type convKernel struct {
	Weight ml.Tensor `gguf:"weight"`
}

// GatedDeltaNet implements linear attention with SSM convolution and recurrent state.
// It implements the Operator interface directly.
type GatedDeltaNet struct {
	// Optimized path: pre-split QKV and gate
	SSMQKV       *nn.Linear  `gguf:"attn_qkv"`  // -> Q, K, V (concatenated)
	SSMQKVGate   *nn.Linear  `gguf:"attn_gate"` // -> Z gate
	SSMBetaAlpha *nn.Linear  `gguf:"ssm_ba"`    // -> beta, alpha (qwen3next)
	SSMBeta      *nn.Linear  `gguf:"ssm_beta"`  // qwen35 / qwen35moe
	SSMAlpha     *nn.Linear  `gguf:"ssm_alpha"` // qwen35 / qwen35moe
	SSMConv1D    *convKernel `gguf:"ssm_conv1d"`
	SSMDT        ml.Tensor   `gguf:"ssm_dt.bias,alt:ssm_dt"` // alpha bias (llama.cpp / published GGUFs)
	SSMA         ml.Tensor   `gguf:"ssm_a"`                  // -A_log.exp()
	SSMNorm      *nn.RMSNorm `gguf:"ssm_norm"`
	SSMOut       *nn.Linear  `gguf:"ssm_out"`

	// Layer index for cache access (set during model construction)
	Layer int
}

func (gdn *GatedDeltaNet) Forward(ctx ml.Context, hiddenStates, _ ml.Tensor, cache *HybridCache, opts *Options) (ml.Tensor, error) {
	layer := gdn.Layer
	nSeqTokens := hiddenStates.Dim(1)
	nSeqs := hiddenStates.Dim(2)
	if cache != nil && cache.IsSupportedForBatch() {
		seqTokens := cache.seqTokens()
		seqs := cache.numSeqs()
		if seqTokens > 0 && seqs > 0 {
			if nSeqs > 1 {
				if nSeqTokens != seqTokens || nSeqs != seqs {
					return nil, ErrUnsupportedBatchLayout
				}
			} else {
				if nSeqTokens != seqTokens*seqs {
					return nil, ErrUnsupportedBatchLayout
				}
				hiddenStates = hiddenStates.Reshape(ctx, hiddenStates.Dim(0), seqTokens, seqs)
				nSeqTokens = seqTokens
				nSeqs = seqs
			}
		}
	}

	headKDim := opts.ssmDState
	numKHeads := opts.ssmNGroup
	numVHeads := opts.ssmDtRank
	headVDim := opts.ssmDInner / numVHeads
	convKernelSize := opts.convKernelSize

	qkvDim := headKDim*numKHeads*2 + headVDim*numVHeads

	if gdn.SSMQKV == nil || gdn.SSMQKVGate == nil {
		return nil, errors.New("qwen3next: missing attn_qkv/attn_gate projections (legacy ssm_in is not supported)")
	}
	// Optimized path: pre-split QKV and gate
	qkvMixed := gdn.SSMQKV.Forward(ctx, hiddenStates).Reshape(ctx, qkvDim, nSeqTokens, nSeqs)
	z := gdn.SSMQKVGate.Forward(ctx, hiddenStates)

	var beta, alpha ml.Tensor
	betaPreSigmoid := false
	switch {
	case gdn.SSMBetaAlpha != nil:
		mixedBA := gdn.SSMBetaAlpha.Forward(ctx, hiddenStates)
		baNewDim := 2 * numVHeads / numKHeads
		mixedBAReshaped := mixedBA.Reshape(ctx, baNewDim, numKHeads, nSeqTokens, nSeqs)

		betaSize := numVHeads / numKHeads
		alphaSize := numVHeads / numKHeads

		b := mixedBAReshaped.Slice(ctx, 0, 0, betaSize, 1)
		a := mixedBAReshaped.Slice(ctx, 0, betaSize, betaSize+alphaSize, 1)

		beta = b.Contiguous(ctx, numVHeads, 1, nSeqTokens, nSeqs)
		alpha = a.Contiguous(ctx, numVHeads, nSeqTokens, nSeqs)
	case gdn.SSMBeta != nil && gdn.SSMAlpha != nil:
		// qwen35moe applies sigmoid to beta before the delta-net update
		beta = gdn.SSMBeta.Forward(ctx, hiddenStates)
		beta = beta.Reshape(ctx, 1, numVHeads, nSeqTokens, nSeqs)
		beta = beta.SigmoidOut(ctx)
		betaPreSigmoid = true

		alpha = gdn.SSMAlpha.Forward(ctx, hiddenStates)
		alpha = alpha.Contiguous(ctx, numVHeads, nSeqTokens, nSeqs)
	default:
		return nil, errors.New("qwen3next: missing ssm_ba or ssm_beta/ssm_alpha projections")
	}

	if gdn.SSMDT == nil || gdn.SSMA == nil {
		return nil, errors.New("qwen3next: missing ssm_dt.bias/ssm_a tensors")
	}

	// Compute gate: softplus(alpha + dt_bias) * -A
	alphaBiased := alpha.Add(ctx, gdn.SSMDT)
	alphaSoftplus := alphaBiased.Softplus(ctx)
	gate := alphaSoftplus.Mul(ctx, gdn.SSMA)
	// llama.cpp build_layer_attn_linear: reshape gate to [1, num_v_heads, n_seq_tokens, n_seqs]
	// before delta-net. Leaving it as [heads, tokens, seqs] makes the chunked permute
	// scramble the time/head axes (prefill garbage).
	gate = gate.Reshape(ctx, 1, numVHeads, nSeqTokens, nSeqs)
	qkvMixed = qkvMixed.Permute(ctx, 1, 0, 2, 3)

	// Get conv state from cache
	convStates, err := cache.ConvState(ctx, layer)
	if err != nil {
		// Log this - if it happens, short-term context will be lost
		slog.Warn("qwen3next: failed to get conv state, using zeros", "layer", layer, "error", err)
		convStates = ctx.Input().Zeros(ml.DTypeF32, convKernelSize-1, qkvDim, nSeqs)
	}

	// Reshape conv states
	convStates = convStates.Reshape(ctx, convKernelSize-1, qkvDim, nSeqs)

	// Concatenate with input for convolution
	convInput := convStates.Concat(ctx, qkvMixed, 0)

	// Save new conv state (last convKernelSize-1 tokens)
	lastConvStates := convInput.Slice(ctx, 0, nSeqTokens, nSeqTokens+convKernelSize-1, 1)
	cache.UpdateConvState(ctx, layer, lastConvStates)

	// Apply SSM convolution (kernel must be F32 for Metal)
	convOutput := convInput.SSMConv(ctx, gdn.SSMConv1D.Weight)
	convOutput = convOutput.SILU(ctx)

	// Reshape for extraction
	convQKVMix := convOutput.Contiguous(ctx, qkvDim, nSeqTokens*nSeqs)

	// Extract convolved Q, K, V
	qConv := convQKVMix.Slice(ctx, 0, 0, headKDim*numKHeads, 1)
	kConv := convQKVMix.Slice(ctx, 0, headKDim*numKHeads, 2*headKDim*numKHeads, 1)
	vConv := convQKVMix.Slice(ctx, 0, 2*headKDim*numKHeads, qkvDim, 1)

	// Reshape to 4D
	qConv = qConv.Contiguous(ctx, headKDim, numKHeads, nSeqTokens, nSeqs)
	kConv = kConv.Contiguous(ctx, headKDim, numKHeads, nSeqTokens, nSeqs)
	vConv = vConv.Contiguous(ctx, headVDim, numVHeads, nSeqTokens, nSeqs)

	// Get delta state from cache
	state, err := cache.DeltaState(ctx, layer, headVDim, numVHeads)
	if err != nil {
		// Log this - if it happens frequently, context will degrade
		slog.Warn("qwen3next: failed to get delta state, using zeros", "layer", layer, "error", err)
		state = ctx.Input().Zeros(ml.DTypeF32, headVDim, headVDim*numVHeads, nSeqs)
	}
	state = state.Reshape(ctx, headVDim, headVDim*numVHeads, 1, nSeqs)

	// llama.cpp fused GDN broadcasts K-heads internally. Do not Repeat4D first.
	attnOut := gdn.deltaNetFused(ctx, qConv, kConv, vConv, gate, beta, state, opts, layer, cache, betaPreSigmoid)

	// Apply gated normalization
	attnOut2D := attnOut.Contiguous(ctx, headVDim, numVHeads*nSeqTokens*nSeqs)
	z2D := z.Contiguous(ctx, headVDim, numVHeads*nSeqTokens*nSeqs)

	// norm(attnOut, z) = RMSNorm(attnOut) * silu(z)
	attnOutNorm := gdn.SSMNorm.Forward(ctx, attnOut2D, opts.eps)
	zSilu := z2D.SILU(ctx)
	attnOutGated := attnOutNorm.Mul(ctx, zSilu)

	// Reshape for output projection
	finalOutput := attnOutGated.Reshape(ctx, headVDim*numVHeads, nSeqTokens, nSeqs)

	out := gdn.SSMOut.Forward(ctx, finalOutput)
	return out.Reshape(ctx, out.Dim(0), nSeqTokens*nSeqs), nil
}

// deltaNetFused is llama.cpp build_delta_net_fused: L2-norm Q/K, then
// GGML_OP_GATED_DELTA_NET (scale + recurrence live in the kernel).
func (gdn *GatedDeltaNet) deltaNetFused(
	ctx ml.Context,
	q, k, v, gate, beta, state ml.Tensor,
	opts *Options,
	layer int,
	cache *HybridCache,
	betaPreSigmoid bool,
) ml.Tensor {
	headVDim := v.Dim(0)
	numVHeads := v.Dim(1)
	nTokens := q.Dim(2)
	nSeqs := q.Dim(3)

	q = q.L2Norm(ctx, opts.eps).Contiguous(ctx)
	k = k.L2Norm(ctx, opts.eps).Contiguous(ctx)
	v = v.Contiguous(ctx)
	if !betaPreSigmoid {
		beta = beta.Sigmoid(ctx)
	}
	gate = gate.Contiguous(ctx)
	beta = beta.Contiguous(ctx)
	state = state.Reshape(ctx, headVDim, headVDim, numVHeads, nSeqs).Contiguous(ctx)

	packed := q.GatedDeltaNet(ctx, k, v, gate, beta, state)
	out, newState := splitGatedDeltaNet(packed, ctx, headVDim, numVHeads, nTokens, nSeqs)
	cache.UpdateDeltaState(ctx, layer, newState.Reshape(ctx, headVDim, headVDim*numVHeads, nSeqs))
	return out
}

// splitGatedDeltaNet views fused GDN packed output the same way llama.cpp does:
// result is [S*H, T*B + S*B]; first slab is attn out [S,H,T,B], second is
// new state [S,S,H,B].
func splitGatedDeltaNet(packed ml.Tensor, ctx ml.Context, s, h, tokens, seqs int) (out, state ml.Tensor) {
	es := packed.Stride(0)
	if es < 1 {
		es = 4
	}
	outNB1 := s * es
	outNB2 := s * h * es
	outNB3 := s * h * tokens * es
	out = packed.View(ctx, 0, s, outNB1, h, outNB2, tokens, outNB3, seqs)
	stateOff := s * h * tokens * seqs * es
	stNB1 := s * es
	stNB2 := s * s * es
	stNB3 := s * s * h * es
	state = packed.View(ctx, stateOff, s, stNB1, s, stNB2, h, stNB3, seqs)
	return out, state
}

func gatedDeltaNetPackedElems(s, h, tokens, seqs int) int {
	return s * h * (tokens*seqs + s*seqs)
}
