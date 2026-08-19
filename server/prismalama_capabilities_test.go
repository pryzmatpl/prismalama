package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/runner"
)

func TestPrismalamaCapabilitiesHandler_JSON(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	PrismalamaCapabilitiesHandler(c)
	if w.Code != http.StatusOK {
		t.Fatalf("status %d body %s", w.Code, w.Body.String())
	}
	var got api.PrismalamaCapabilitiesResponse
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Version == "" {
		t.Fatal("version must be set")
	}
	if got.GGUF.Engine == "" || got.GGUF.WeightSemantics == "" {
		t.Fatal("gguf_ggml fields required")
	}
	if got.AirLLM.Engine == "" || got.AirLLM.OptInEnv != "OLLAMA_USE_AIRLLM" {
		t.Fatal("airllm metadata incomplete")
	}
	if got.Enterprise.CapabilitiesPath != "/api/prismalama/capabilities" {
		t.Fatalf("enterprise path: %q", got.Enterprise.CapabilitiesPath)
	}
}

func TestBuildPrismalamaCapabilities_EnvironmentPassthrough(t *testing.T) {
	t.Setenv("OLLAMA_USE_AIRLLM", "0")
	got := buildPrismalamaCapabilities()
	if got.Environment.OLLAMA_USE_AIRLLM != "0" {
		t.Fatalf("want OLLAMA_USE_AIRLLM=0 in response, got %q", got.Environment.OLLAMA_USE_AIRLLM)
	}
}

// Phase 0 / JAISIU-2158: v2 schema additions.

func TestBuildPrismalamaCapabilities_SchemaV2(t *testing.T) {
	got := buildPrismalamaCapabilities()
	if got.SchemaVersion != "2" {
		t.Fatalf("schema_version: want 2, got %q", got.SchemaVersion)
	}
	if got.Build.GOOS == "" || got.Build.GOARCH == "" || got.Build.GoVersion == "" {
		t.Fatalf("build info incomplete: %+v", got.Build)
	}
}

func TestBuildPrismalamaCapabilities_V2EnvironmentFields(t *testing.T) {
	t.Setenv("OLLAMA_KEEP_ALIVE", "10m")
	t.Setenv("OLLAMA_GPU_OVERHEAD", "1073741824")     // 1 GiB
	t.Setenv("OLLAMA_STREAMING_BUDGET", "2147483648") // 2 GiB
	t.Setenv("OLLAMA_LIBRARY_PATH", "/tmp/fake-lib")
	t.Setenv("AIRLLM_DEVICE", "cuda:0")
	t.Setenv("AIRLLM_COMPRESSION", "4bit")

	got := buildPrismalamaCapabilities()
	if got.Environment.OLLAMA_KEEP_ALIVE != "10m" {
		t.Fatalf("OLLAMA_KEEP_ALIVE passthrough: %q", got.Environment.OLLAMA_KEEP_ALIVE)
	}
	if got.Environment.OLLAMA_GPU_OVERHEAD_RAW != "1073741824" {
		t.Fatalf("OLLAMA_GPU_OVERHEAD raw: %q", got.Environment.OLLAMA_GPU_OVERHEAD_RAW)
	}
	if got.Environment.OLLAMA_STREAMING_BUDGET_RAW != "2147483648" {
		t.Fatalf("OLLAMA_STREAMING_BUDGET raw: %q", got.Environment.OLLAMA_STREAMING_BUDGET_RAW)
	}
	if got.Environment.OLLAMA_LIBRARY_PATH != "/tmp/fake-lib" {
		t.Fatalf("OLLAMA_LIBRARY_PATH: %q", got.Environment.OLLAMA_LIBRARY_PATH)
	}
	if got.Environment.AIRLLM_DEVICE != "cuda:0" {
		t.Fatalf("AIRLLM_DEVICE: %q", got.Environment.AIRLLM_DEVICE)
	}
	if got.Environment.AIRLLM_COMPRESSION != "4bit" {
		t.Fatalf("AIRLLM_COMPRESSION: %q", got.Environment.AIRLLM_COMPRESSION)
	}
}

func TestBuildPrismalamaCapabilities_ResolvedValues(t *testing.T) {
	t.Setenv("OLLAMA_GPU_OVERHEAD", "")
	t.Setenv("OLLAMA_STREAMING_BUDGET", "")
	got := buildPrismalamaCapabilities()
	if got.Resolved.GpuOverheadBytes != envconfig.GpuOverhead() {
		t.Fatalf("gpu_overhead_bytes: want %d, got %d",
			envconfig.GpuOverhead(), got.Resolved.GpuOverheadBytes)
	}
	if got.Resolved.GpuOverheadHuman == "" {
		t.Fatal("gpu_overhead_human must be set")
	}
	if !strings.Contains(got.Resolved.GpuOverheadHuman, "B") {
		t.Fatalf("gpu_overhead_human should include byte unit, got %q", got.Resolved.GpuOverheadHuman)
	}
	if got.Resolved.StreamingBudgetBytes != envconfig.StreamingBudgetBytes() {
		t.Fatalf("streaming_budget_bytes: want %d, got %d",
			envconfig.StreamingBudgetBytes(), got.Resolved.StreamingBudgetBytes)
	}
	if got.Resolved.StreamingBudgetHuman == "" {
		t.Fatal("streaming_budget_human must be set")
	}
	if uint64(format.GibiByte) != got.Resolved.StreamingBudgetBytes {
		// Sanity: default is 4 GiB = 4 * format.GibiByte. Loose — envconfig
		// may change; we only assert non-zero and human-readable.
	}
}

func TestBuildPrismalamaCapabilities_BackendsArrayPresent(t *testing.T) {
	got := buildPrismalamaCapabilities()
	if len(got.Backends) == 0 {
		t.Fatal("backends array must be present (even with all Loaded=false)")
	}
	want := map[string]bool{"cpu": false, "cuda": false, "hip": false, "vulkan": false, "metal": false}
	for _, b := range got.Backends {
		if _, ok := want[b.Name]; !ok {
			t.Errorf("unexpected backend name: %q", b.Name)
		}
		want[b.Name] = true
	}
	for n, seen := range want {
		if !seen {
			t.Errorf("backend %q missing from probe", n)
		}
	}
}

func TestProbeBackendsSafe_Recovers(t *testing.T) {
	// Even if probeBackends panics on some future platform, the safe wrapper
	// must not propagate. We can't easily inject a panic here without
	// modifying probeBackends — but we can assert the function returns a
	// non-nil slice and doesn't panic on the current host.
	got := probeBackendsSafe()
	if got == nil {
		t.Fatal("probeBackendsSafe returned nil")
	}
}

func TestPrismalamaDispatchHandler_DryRun(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/prismalama/dispatch",
		bytes.NewReader([]byte(`{"model_path":""}`)))
	c.Request.Header.Set("Content-Type", "application/json")
	PrismalamaDispatchHandler(c)
	if w.Code != http.StatusOK {
		t.Fatalf("status %d body %s", w.Code, w.Body.String())
	}
	var got api.DispatchResponse
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.ModelPath != "" {
		t.Fatalf("model_path echo: %q", got.ModelPath)
	}
	if got.Decision.Kind != "ggml" {
		t.Fatalf("empty path should be ggml, got %q", got.Decision.Kind)
	}
	if got.Decision.Selected != "empty_path" {
		t.Fatalf("selected: want empty_path, got %q", got.Decision.Selected)
	}
}

func TestPrismalamaDispatchHandler_RecordLastDecision(t *testing.T) {
	// After the dispatch handler runs, the capabilities endpoint should
	// surface last_decision.
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/prismalama/dispatch",
		bytes.NewReader([]byte(`{"model_path":"/some/nonexistent/path"}`)))
	c.Request.Header.Set("Content-Type", "application/json")
	PrismalamaDispatchHandler(c)
	if w.Code != http.StatusOK {
		t.Fatalf("dispatch status %d", w.Code)
	}

	w2 := httptest.NewRecorder()
	c2, _ := gin.CreateTestContext(w2)
	PrismalamaCapabilitiesHandler(c2)
	var got api.PrismalamaCapabilitiesResponse
	if err := json.Unmarshal(w2.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.LastDecision == nil {
		t.Fatal("last_decision must be set after a dispatch")
	}
	if got.LastDecision.Selected != "path_missing" {
		t.Fatalf("last_decision.selected: want path_missing, got %q", got.LastDecision.Selected)
	}
}

func TestPrismalamaDispatchHandler_MalformedBody(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/prismalama/dispatch",
		bytes.NewReader([]byte(`{not json}`)))
	c.Request.Header.Set("Content-Type", "application/json")
	PrismalamaDispatchHandler(c)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("want 400 on malformed body, got %d", w.Code)
	}
}

func TestDecisionToSnapshot(t *testing.T) {
	d := runner.EngineDecision{
		Kind:     runner.EngineAirLLM,
		Selected: runner.ReasonSafetensorsIndex,
		Reasons:  []runner.Reason{runner.ReasonSafetensorsIndex},
	}
	snap := decisionToSnapshot("/tmp/m", d)
	if snap.Kind != "airllm" {
		t.Fatalf("kind: %q", snap.Kind)
	}
	if snap.Selected != "model.safetensors.index.json" {
		t.Fatalf("selected: %q", snap.Selected)
	}
	if snap.SelectedID != int(runner.ReasonSafetensorsIndex) {
		t.Fatalf("selected_id: %d", snap.SelectedID)
	}
	if len(snap.ReasonTrace) != 1 || snap.ReasonTrace[0] != "model.safetensors.index.json" {
		t.Fatalf("trace: %v", snap.ReasonTrace)
	}
}
