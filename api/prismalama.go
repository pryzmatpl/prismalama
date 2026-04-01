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

	Environment struct {
		OLLAMA_USE_AIRLLM string `json:"ollama_use_airllm"`
	} `json:"environment"`

	Enterprise struct {
		CapabilitiesPath string `json:"capabilities_http_path"`
		DispatchDocs     string `json:"dispatch_documentation_path"`
		Note             string `json:"note"`
	} `json:"enterprise"`
}
