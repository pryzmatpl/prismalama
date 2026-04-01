//go:build integration

package integration

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

// TestPrismalamaCapabilitiesEndpointLive hits GET /api/prismalama/capabilities on a running server.
// Requires OLLAMA_TEST_EXISTING=1 and OLLAMA_HOST pointing at that server (see integration/utils_test.go).
func TestPrismalamaCapabilitiesEndpointLive(t *testing.T) {
	if os.Getenv("OLLAMA_TEST_EXISTING") == "" {
		t.Skip("set OLLAMA_TEST_EXISTING=1 and OLLAMA_HOST to a running server")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	_, testEndpoint, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://"+testEndpoint+"/api/prismalama/capabilities", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d", resp.StatusCode)
	}
	var got api.PrismalamaCapabilitiesResponse
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if got.Version == "" || got.GGUF.Engine == "" || got.AirLLM.Engine == "" {
		t.Fatalf("incomplete response: %+v", got)
	}
}
