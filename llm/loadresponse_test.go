package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestLoadResponseJSONRoundTrip(t *testing.T) {
	orig := LoadResponse{
		Success: true,
		Error:   "",
	}
	data, err := json.Marshal(orig)
	if err != nil {
		t.Fatal(err)
	}
	var got LoadResponse
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if !got.Success || got.Error != "" {
		t.Fatalf("round trip: %+v", got)
	}
}

func TestLoadResponseUnmarshalRunnerError(t *testing.T) {
	const airllmStyle = `{"success":false,"error":"CUDA OOM in Python"}`
	var r LoadResponse
	if err := json.Unmarshal([]byte(airllmStyle), &r); err != nil {
		t.Fatal(err)
	}
	if r.Success || r.Error != "CUDA OOM in Python" {
		t.Fatalf("got %+v", r)
	}
	err := errLoadCommitFailed(&r, "failed to allocate memory for model")
	if err == nil || !strings.Contains(err.Error(), "CUDA OOM") {
		t.Fatalf("expected propagated runner error, got %v", err)
	}
}

func TestLoadResponseUnmarshalLegacyPascalCase(t *testing.T) {
	// Older runners may still emit PascalCase keys
	const legacy = `{"Success":true,"Memory":{}}`
	var r LoadResponse
	if err := json.Unmarshal([]byte(legacy), &r); err != nil {
		t.Fatal(err)
	}
	if !r.Success {
		t.Fatal("expected Success true from legacy JSON")
	}
}

func TestErrLoadCommitFailedOOMFallback(t *testing.T) {
	r := &LoadResponse{Success: false}
	err := errLoadCommitFailed(r, "failed to allocate memory for model")
	if err == nil || err.Error() != "failed to allocate memory for model" {
		t.Fatalf("got %v", err)
	}
}
