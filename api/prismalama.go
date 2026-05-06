package api

// PrismalamaCapabilitiesResponse is returned by GET /api/prismalama/capabilities for operators
// and automation. It documents engine split (GGML vs AirLLM); it is not a performance guarantee.
type PrismalamaCapabilitiesResponse struct {
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
		Enabled         bool   `json:"enabled"`
		BudgetBytes     uint64 `json:"budget_bytes"`
		Semantics       string `json:"semantics"`
		EnableEnv       string `json:"enable_environment_variable"`
	} `json:"layer_streaming"`

	Environment struct {
		OLLAMA_USE_AIRLLM           string `json:"ollama_use_airllm"`
		OLLAMA_LAYER_STREAMING      string `json:"ollama_layer_streaming"`
		OLLAMA_MEMORY_POLICY        string `json:"ollama_memory_policy,omitempty"`
		OLLAMA_VULKAN               string `json:"ollama_vulkan,omitempty"`
		OLLAMA_MMAP_ALLOW_LOW_RAM   string `json:"ollama_mmap_allow_low_ram,omitempty"`
	} `json:"environment"`

	// OperatorHints are actionable configuration reminders derived from current env (not diagnostics).
	OperatorHints []string `json:"operator_hints,omitempty"`

	Enterprise struct {
		CapabilitiesPath string `json:"capabilities_http_path"`
		DispatchDocs     string `json:"dispatch_documentation_path"`
		Note             string `json:"note"`
	} `json:"enterprise"`
}
