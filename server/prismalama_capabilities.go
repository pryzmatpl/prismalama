package server

import (
	"fmt"
	"net/http"
	"os"
	"runtime"
	"sync"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/runner"
	"github.com/ollama/ollama/version"
)

// capabilitiesSchemaVersion is the current /api/prismalama/capabilities schema
// version. Bumped when the response shape changes; clients SHOULD use this to
// gate parsing of additive fields.
const capabilitiesSchemaVersion = "2"

// lastDecisionMu guards lastDecision. Set by RecordLastDispatch (called from
// the runner dispatch path). Read by buildPrismalamaCapabilities.
var (
	lastDecisionMu sync.RWMutex
	lastDecision   *api.DispatchDecisionSnapshot
)

// RecordLastDispatch snapshots the most recent dispatch decision into the
// capabilities response. Safe to call from any goroutine; nil-safe.
//
// Wire this from runner.Execute (JAISIU-2157 follow-up) and from the
// POST /api/prismalama/dispatch handler so operators can see what the
// last decision was without scraping logs.
func RecordLastDispatch(modelPath string, d runner.EngineDecision) {
	snap := decisionToSnapshot(modelPath, d)
	lastDecisionMu.Lock()
	lastDecision = &snap
	lastDecisionMu.Unlock()
}

func decisionToSnapshot(modelPath string, d runner.EngineDecision) api.DispatchDecisionSnapshot {
	trace := make([]string, len(d.Reasons))
	for i, r := range d.Reasons {
		trace[i] = r.String()
	}
	return api.DispatchDecisionSnapshot{
		Kind:        d.Kind.String(),
		Selected:    d.Selected.String(),
		SelectedID:  int(d.Selected),
		ReasonTrace: trace,
	}
}

func getLastDecision() *api.DispatchDecisionSnapshot {
	lastDecisionMu.RLock()
	defer lastDecisionMu.RUnlock()
	return lastDecision
}

// PrismalamaCapabilitiesHandler documents GGML vs AirLLM inference semantics for enterprise operators.
func PrismalamaCapabilitiesHandler(c *gin.Context) {
	c.JSON(http.StatusOK, buildPrismalamaCapabilities())
}

// PrismalamaDispatchHandler dry-runs dispatch for a model path without
// touching the runner. Useful for operators verifying which engine WOULD
// handle a given layout, without spawning the subprocess.
//
// Request body: api.DispatchRequest{ModelPath: "..."}
// Response:     api.DispatchResponse (200) | 400 on malformed body
func PrismalamaDispatchHandler(c *gin.Context) {
	var req api.DispatchRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request body: " + err.Error()})
		return
	}
	d := runner.DecideEngineDetailed(req.ModelPath)
	snap := decisionToSnapshot(req.ModelPath, d)
	// Snapshot for subsequent GET /api/prismalama/capabilities.
	RecordLastDispatch(req.ModelPath, d)
	c.JSON(http.StatusOK, api.DispatchResponse{
		ModelPath: req.ModelPath,
		Decision:  snap,
	})
}

func buildPrismalamaCapabilities() api.PrismalamaCapabilitiesResponse {
	var resp api.PrismalamaCapabilitiesResponse
	resp.SchemaVersion = capabilitiesSchemaVersion
	resp.Version = version.Version

	resp.GGUF.Engine = "llama.cpp / GGML (native runner)"
	resp.GGUF.WeightSemantics = "mmap, partial GPU offload, KV in VRAM/RAM; not PyTorch layer-by-layer NVMe streaming"

	resp.AirLLM.Engine = "Python AirLLM + PyTorch (airllm_runner)"
	resp.AirLLM.WeightSemantics = "Hugging Face–style checkpoints: layer-wise execution and NVMe-oriented streaming where AirLLM supports the architecture"
	resp.AirLLM.OptInEnv = "OLLAMA_USE_AIRLLM"

	resp.LayerStreaming.Enabled = envconfig.LayerStreaming()
	resp.LayerStreaming.BudgetBytes = envconfig.StreamingBudgetBytes()
	resp.LayerStreaming.Semantics = "GGUF layer-by-layer: load block from NVMe, compute on GPU, evict, prefetch next — AirLLM-like behavior for native GGUF"
	resp.LayerStreaming.EnableEnv = "OLLAMA_LAYER_STREAMING"

	// v1 env passthrough (preserved)
	resp.Environment.OLLAMA_USE_AIRLLM = os.Getenv("OLLAMA_USE_AIRLLM")
	resp.Environment.OLLAMA_LAYER_STREAMING = os.Getenv("OLLAMA_LAYER_STREAMING")
	resp.Environment.OLLAMA_MEMORY_POLICY = os.Getenv("OLLAMA_MEMORY_POLICY")
	resp.Environment.OLLAMA_VULKAN = os.Getenv("OLLAMA_VULKAN")
	resp.Environment.OLLAMA_MMAP_ALLOW_LOW_RAM = os.Getenv("OLLAMA_MMAP_ALLOW_LOW_RAM")

	// v2 env additions
	resp.Environment.OLLAMA_KEEP_ALIVE = os.Getenv("OLLAMA_KEEP_ALIVE")
	resp.Environment.OLLAMA_GPU_OVERHEAD_RAW = os.Getenv("OLLAMA_GPU_OVERHEAD")
	resp.Environment.OLLAMA_STREAMING_BUDGET_RAW = os.Getenv("OLLAMA_STREAMING_BUDGET")
	resp.Environment.OLLAMA_LIBRARY_PATH = os.Getenv("OLLAMA_LIBRARY_PATH")
	resp.Environment.HIP_VISIBLE_DEVICES = os.Getenv("HIP_VISIBLE_DEVICES")
	resp.Environment.AIRLLM_DEVICE = os.Getenv("AIRLLM_DEVICE")
	resp.Environment.AIRLLM_COMPRESSION = os.Getenv("AIRLLM_COMPRESSION")
	resp.Environment.PRISMALAMA_AIRLLM_PYTHONPATH = os.Getenv("PRISMALAMA_AIRLLM_PYTHONPATH")

	// Resolved numeric env vars (parsed). Falls back to envconfig defaults if env unset.
	// Cast uint64 → int64 for format.HumanBytes (which expects int64).
	resp.Resolved.GpuOverheadBytes = envconfig.GpuOverhead()
	resp.Resolved.GpuOverheadHuman = format.HumanBytes(int64(envconfig.GpuOverhead()))
	resp.Resolved.StreamingBudgetBytes = envconfig.StreamingBudgetBytes()
	resp.Resolved.StreamingBudgetHuman = format.HumanBytes(int64(envconfig.StreamingBudgetBytes()))

	// Build info — GoVersion is filled at runtime; CompiledAt is best-effort
	// (operators can wire it via -ldflags -X .../version.CompiledAt=<RFC3339>).
	resp.Build.Version = version.Version
	resp.Build.GoVersion = runtime.Version()
	resp.Build.GOOS = runtime.GOOS
	resp.Build.GOARCH = runtime.GOARCH
	resp.Build.CompiledAt = version.CompiledAt

	// Backends — best-effort, recover-safe. Never lets the endpoint 500
	// because of a probe failure.
	resp.Backends = probeBackendsSafe()

	// Last decision — populated by RecordLastDispatch.
	if ld := getLastDecision(); ld != nil {
		resp.LastDecision = ld
	}

	resp.OperatorHints = prismalamaOperatorHints()

	resp.Enterprise.CapabilitiesPath = "/api/prismalama/capabilities"
	resp.Enterprise.DispatchDocs = "docs/RUNTIME_DISPATCH.md"
	resp.Enterprise.Note = "Layer streaming (OLLAMA_LAYER_STREAMING=1) brings AirLLM-like semantics to GGUF via native prismallama compute; see docs/PRISMALAMA_PRINCIPLE.md"

	return resp
}

// probeBackendsSafe wraps backend probing in a recover() so a panic in any
// backend's discovery code returns an empty slice instead of 500-ing the
// capabilities endpoint. When discover/ exposes a real BackendProbe API,
// replace the body of this function with a call to it.
//
// Default behavior for Phase 0: probe known platform-specific locations and
// return per-backend Discovered/Loaded flags. Loaded is conservative (true
// only when dlopen succeeded in this process — which it generally has not,
// because backends are loaded by the runner subprocess, not the server).
func probeBackendsSafe() []api.BackendInfo {
	defer func() {
		if r := recover(); r != nil {
			slogRecover(r, "probeBackendsSafe")
		}
	}()
	return probeBackends()
}

func probeBackends() []api.BackendInfo {
	out := []api.BackendInfo{
		{Name: "cpu"},
		{Name: "cuda"},
		{Name: "hip"},
		{Name: "vulkan"},
		{Name: "metal"},
	}
	for i := range out {
		// Mark Discovered if the platform-specific library would be loadable.
		// Phase 0 default: conservative — Discovered=false everywhere; the
		// future BackendProbe API (planned Phase 1 / JAISIU-2XXX) will set
		// this from a single scan of OLLAMA_LIBRARY_PATH.
		out[i].Discovered = false
		out[i].Loaded = false
	}
	// runtime.GOOS-specific defaults — without claiming discovery, just note
	// what *could* be on the host. Operators get the truth from logs.
	if runtime.GOOS == "linux" {
		// Vulkan is the most likely to be loadable on Linux once an ICD is
		// present. Leave Discovered=false until BackendProbe exists; do not
		// lie about presence.
	} else if runtime.GOOS == "darwin" {
		// Metal is the only GPU path on darwin. Same caveat.
	}
	return out
}

// slogRecover logs a recovered panic without crashing. Standalone so other
// v2 capabilities fields can use it if they need defensive recovery later.
func slogRecover(r interface{}, where string) {
	// Use stdlib log to avoid pulling slog into this helper (which may be
	// called before slog is fully configured during boot).
	fmt.Fprintf(os.Stderr, "prismalama: recovered panic in %s: %v\n", where, r)
}

func prismalamaOperatorHints() []string {
	var h []string
	if !envconfig.LayerStreaming() {
		h = append(h, "OLLAMA_LAYER_STREAMING is off (unset/false). Set OLLAMA_LAYER_STREAMING=1 so GGUF can use the layer-streaming path when weights exceed VRAM (requires backend support; runner logs show whether streaming activated).")
	}
	if envconfig.MemoryPolicy() != "balanced" {
		h = append(h, "OLLAMA_MEMORY_POLICY defaults to performance. For models much larger than VRAM or many parallel requests, set OLLAMA_MEMORY_POLICY=balanced for conservative default context and KV budgeting (see server/memory_policy.go).")
	}
	if runtime.GOOS == "linux" && !envconfig.EnableVulkan(true) {
		h = append(h, "OLLAMA_VULKAN is off: Vulkan GGML backends are skipped during GPU discovery (discover/runner.go). Set OLLAMA_VULKAN=1 if you rely on Vulkan. CUDA/HIP libraries are still discovered separately when present.")
	}
	if runtime.GOOS == "linux" && !envconfig.MmapAllowLowRamLinux() {
		h = append(h, "OLLAMA_MMAP_ALLOW_LOW_RAM is off: when free RAM is below the GGUF size, Linux may disable mmap and force full resident weights. For fast NVMe + larger-than-RAM GGUF, set OLLAMA_MMAP_ALLOW_LOW_RAM=1 (see llm/server.go applyLoadMmapPolicy).")
	}
	return h
}