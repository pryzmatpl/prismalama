package integration

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestBC4ShaderCompilation(t *testing.T) {
	glslc, err := exec.LookPath("glslc")
	if err != nil {
		t.Skip("glslc not available, skipping shader compilation test")
	}

	shaderPath := filepath.Join(os.Getenv("OLLAMA_TEST_SHADER_PATH"),
		"llama/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/bc4_decompress.comp")
	if shaderPath == "" || os.Getenv("OLLAMA_TEST_SHADER_PATH") == "" {
		shaderPath = "/home/prizm/prismalama/llama/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/bc4_decompress.comp"
	}
	if _, err := os.Stat(shaderPath); os.IsNotExist(err) {
		t.Fatalf("BC4 decompress shader not found at %s", shaderPath)
	}

	cmd := exec.Command(glslc, "--target-env=vulkan1.2", "-o", "/dev/null", shaderPath)
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Errorf("glslc compilation failed: %s\nOutput: %s", err, string(output))
		return
	}
	t.Logf("BC4 decompress shader compiled successfully")
}

func TestBC4ShaderExists(t *testing.T) {
	paths := []string{
		"/home/prizm/prismalama/llama/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/bc4_decompress.comp",
		filepath.Join(os.Getenv("PWD"), "llama/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/bc4_decompress.comp"),
	}

	var shaderPath string
	for _, p := range paths {
		if f, err := os.Stat(p); err == nil && !f.IsDir() {
			shaderPath = p
			break
		}
	}

	if shaderPath == "" {
		t.Fatal("BC4 decompress shader not found")
	}

	data, err := os.ReadFile(shaderPath)
	if err != nil {
		t.Fatalf("Failed to read shader: %v", err)
	}

	content := string(data)

	expected := []string{
		"#version 450",
		"local_size_x = 64",
		"BC4Block",
		"OutputF32",
		"BlockInfo",
		"num_blocks",
		"output_stride",
		"p0_norm",
		"p1_norm",
		"val_norm",
	}

	for _, e := range expected {
		if !contains(content, e) {
			t.Errorf("Shader missing expected element: %s", e)
		}
	}

	if !contains(content, "gl_GlobalInvocationID.x") {
		t.Errorf("Shader missing global invocation ID usage")
	}

	t.Logf("BC4 shader structure validated (%d bytes)", len(data))
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsHelper(s, substr))
}

func containsHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
