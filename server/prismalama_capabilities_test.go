package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
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
