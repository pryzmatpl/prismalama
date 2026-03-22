package runner

import (
	"os"
	"path/filepath"
	"testing"
)

func TestIsAirLLMModel(t *testing.T) {
	tests := []struct {
		name     string
		setup    func(string)
		expected bool
	}{
		{
			name: "empty path",
			setup: func(_ string) {
			},
			expected: false,
		},
		{
			name: "non-existent path",
			setup: func(_ string) {
			},
			expected: false,
		},
		{
			name: "safetensors index file exists",
			setup: func(dir string) {
				os.MkdirAll(dir, 0755)
				os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), []byte("{}"), 0644)
			},
			expected: true,
		},
		{
			name: "safetensors file exists",
			setup: func(dir string) {
				os.MkdirAll(dir, 0755)
				os.WriteFile(filepath.Join(dir, "model-00001-of-00004.safetensors"), []byte("fake"), 0644)
			},
			expected: true,
		},
		{
			name: "config.json with transformers",
			setup: func(dir string) {
				os.MkdirAll(dir, 0755)
				config := `{"architectures": ["LlamaForCausalLM"], "torch_dtype": "float16"}`
				os.WriteFile(filepath.Join(dir, "config.json"), []byte(config), 0644)
			},
			expected: true,
		},
		{
			name: "config.json with safetensors reference",
			setup: func(dir string) {
				os.MkdirAll(dir, 0755)
				config := `{"model_type": "llama", "_format": "safetensors"}`
				os.WriteFile(filepath.Join(dir, "config.json"), []byte(config), 0644)
			},
			expected: true,
		},
		{
			name: "GGUF file only",
			setup: func(dir string) {
				os.MkdirAll(dir, 0755)
				os.WriteFile(filepath.Join(dir, "model.gguf"), []byte("GGUF"), 0644)
			},
			expected: false,
		},
		{
			name: "multipart GGUF first shard",
			setup: func(dir string) {
				os.MkdirAll(dir, 0755)
				os.WriteFile(filepath.Join(dir, "weights-00001-of-00004.gguf"), []byte("GGUF"), 0644)
			},
			expected: true,
		},
		{
			name: "empty directory",
			setup: func(dir string) {
				os.MkdirAll(dir, 0755)
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := filepath.Join(t.TempDir(), "model")

			if tt.name != "empty path" && tt.name != "non-existent path" {
				tt.setup(dir)
			} else if tt.name == "non-existent path" {
				dir = "/non/existent/path/12345"
			} else {
				dir = ""
			}

			result := isAirLLMModel(dir)

			if result != tt.expected {
				t.Errorf("isAirLLMModel(%s) = %v, want %v", dir, result, tt.expected)
			}
		})
	}
}

func TestIsAirLLMModelWithEnv(t *testing.T) {
	dir := t.TempDir()
	os.MkdirAll(dir, 0755)
	os.WriteFile(filepath.Join(dir, "model.gguf"), []byte("GGUF"), 0644)

	originalEnv := os.Getenv("OLLAMA_USE_AIRLLM")
	defer os.Setenv("OLLAMA_USE_AIRLLM", originalEnv)

	os.Setenv("OLLAMA_USE_AIRLLM", "1")
	if !isAirLLMModel(dir) {
		t.Error("Expected isAirLLMModel to return true when OLLAMA_USE_AIRLLM=1")
	}

	os.Setenv("OLLAMA_USE_AIRLLM", "true")
	if !isAirLLMModel(dir) {
		t.Error("Expected isAirLLMModel to return true when OLLAMA_USE_AIRLLM=true")
	}

	os.Setenv("OLLAMA_USE_AIRLLM", "0")
	if isAirLLMModel(dir) {
		t.Error("Expected isAirLLMModel to return false for GGUF without env forcing")
	}
}

func TestGetModelPath(t *testing.T) {
	tests := []struct {
		args     []string
		expected string
	}{
		{[]string{"--model", "/path/to/model"}, "/path/to/model"},
		{[]string{"--model=/another/path"}, "/another/path"},
		{[]string{"other", "args", "--model", "/model/path"}, "/model/path"},
		{[]string{"--model"}, ""},
		{[]string{"--port", "8080"}, ""},
		{[]string{}, ""},
	}

	for _, tt := range tests {
		result := getModelPath(tt.args)
		if result != tt.expected {
			t.Errorf("getModelPath(%v) = %q, want %q", tt.args, result, tt.expected)
		}
	}
}
