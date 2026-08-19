package api

// PrismalamaCapabilitiesResponse is returned by GET /api/prismalama/capabilities for operators
// and automation. It documents engine split (GGML vs AirLLM); it is not a performance guarantee.
//
// Schema versioning: see SchemaVersion. v1 fields are preserved unchanged; v2 adds Build,
// Resolved, Backends, LastDecision, expanded Environment, and SchemaVersion itself.
type PrismalamaCapabilitiesResponse struct {
	// SchemaVersion — "2" for the v2 schema (additive on v1).
	// Clients MUST treat absence as v1.
	SchemaVersion string `json:"schema_version,omitempty"`

	Version string `json:"version"`

	GGUF struct {
		Engine          string `json:"engine"`
		WeightSemantics string `json:"weight_semantics"`
	} `json:"gguf_ggml"`

	AirLLM struct {
		Engine          string `json:"engine"`
		WeightSemantics string `json:"weight_semantics"`
		OptInEnv        string `json:"opt_in_environment_variable"`
	} `json:"airllm"`

	LayerStreaming struct {
		Enabled     bool   `json:"enabled"`
		BudgetBytes uint64 `json:"budget_bytes"`
		Semantics   string `json:"semantics"`
		EnableEnv   string `json:"enable_environment_variable"`
	} `json:"layer_streaming"`

	Environment struct {
		// v1 fields (preserved)
		OLLAMA_USE_AIRLLM         string `json:"ollama_use_airllm"`
		OLLAMA_LAYER_STREAMING    string `json:"ollama_layer_streaming"`
		OLLAMA_MEMORY_POLICY      string `json:"ollama_memory_policy,omitempty"`
		OLLAMA_VULKAN             string `json:"ollama_vulkan,omitempty"`
		OLLAMA_MMAP_ALLOW_LOW_RAM string `json:"ollama_mmap_allow_low_ram,omitempty"`

		// v2 additions — additive, all optional
		OLLAMA_KEEP_ALIVE            string `json:"ollama_keep_alive,omitempty"`
		OLLAMA_GPU_OVERHEAD_RAW      string `json:"ollama_gpu_overhead_raw,omitempty"`
		OLLAMA_STREAMING_BUDGET_RAW  string `json:"ollama_streaming_budget_raw,omitempty"`
		OLLAMA_LIBRARY_PATH          string `json:"ollama_library_path,omitempty"`
		HIP_VISIBLE_DEVICES          string `json:"hip_visible_devices,omitempty"`
		AIRLLM_DEVICE                string `json:"airllm_device,omitempty"`
		AIRLLM_COMPRESSION           string `json:"airllm_compression,omitempty"`
		PRISMALAMA_AIRLLM_PYTHONPATH string `json:"prismalama_airllm_pythonpath,omitempty"`
	} `json:"environment"`

	// Resolved — numeric env vars parsed to bytes (or durations), with human-readable form.
	// Additive; v1 clients ignore this block.
	Resolved struct {
		GpuOverheadBytes     uint64 `json:"gpu_overhead_bytes,omitempty"`
		GpuOverheadHuman     string `json:"gpu_overhead_human,omitempty"`
		StreamingBudgetBytes uint64 `json:"streaming_budget_bytes,omitempty"`
		StreamingBudgetHuman string `json:"streaming_budget_human,omitempty"`
	} `json:"resolved"`

	// Build — machine-readable build info (v2; additive).
	Build struct {
		Version    string `json:"version,omitempty"`
		GoVersion  string `json:"go_version,omitempty"`
		GOOS       string `json:"goos,omitempty"`
		GOARCH     string `json:"goarch,omitempty"`
		CompiledAt string `json:"compiled_at,omitempty"`
	} `json:"build"`

	// Backends — best-effort probe of which GGML backends were discovered at startup.
	// Wrapped in recover() server-side; absence means "not probed on this platform".
	Backends []BackendInfo `json:"backends,omitempty"`

	// LastDecision — most recent dispatch decision trace. nil until first runner start.
	LastDecision *DispatchDecisionSnapshot `json:"last_decision,omitempty"`

	// OperatorHints are actionable configuration reminders derived from current env (not diagnostics).
	OperatorHints []string `json:"operator_hints,omitempty"`

	Enterprise struct {
		CapabilitiesPath string `json:"capabilities_http_path"`
		DispatchDocs     string `json:"dispatch_documentation_path"`
		Note             string `json:"note"`
	} `json:"enterprise"`
}

// BackendInfo describes a single GGML backend discovered (or not) at startup.
// "discovered" means the .so was found on disk; "loaded" means dlopen succeeded
// and the probe call did not panic. Either can be false on hosts without the
// matching toolchain (HIP, CUDA, Vulkan ICD, Metal, CPU).
type BackendInfo struct {
	Name        string `json:"name"`                   // "hip" | "cuda" | "vulkan" | "cpu" | "metal"
	Discovered  bool   `json:"discovered"`             // .so / framework present
	Loaded      bool   `json:"loaded"`                 // dlopen / framework probe ok
	LibraryPath string `json:"library_path,omitempty"` // absolute path if discovered
	Version     string `json:"version,omitempty"`      // if reported by backend
	Error       string `json:"error,omitempty"`        // diagnostic if Loaded=false
}

// DispatchDecisionSnapshot is the JSON-serializable view of an EngineDecision.
// Field names mirror runner.EngineDecision; populated by capabilities after the
// first dispatch (kept thread-safe in server/prismalama_capabilities.go).
type DispatchDecisionSnapshot struct {
	Kind        string   `json:"kind"`        // "ggml" | "airllm"
	Selected    string   `json:"selected"`    // Reason.String() — stable identifier
	SelectedID  int      `json:"selected_id"` // int(Reason) for machine clients
	ReasonTrace []string `json:"reason_trace,omitempty"`
}

// DispatchRequest is the body of POST /api/prismalama/dispatch.
// model_path is the only required field; env_override is optional (testing only).
type DispatchRequest struct {
	ModelPath   string            `json:"model_path"`
	EnvOverride map[string]string `json:"env_override,omitempty"` // reserved for tests; ignored in production
}

// DispatchResponse is the JSON returned by POST /api/prismalama/dispatch.
// model_path echoes the request; if non-empty, the response also includes
// the resolved EngineDecision trace so operators can see the rule sequence.
type DispatchResponse struct {
	ModelPath string                   `json:"model_path"`
	Decision  DispatchDecisionSnapshot `json:"decision"`
}
