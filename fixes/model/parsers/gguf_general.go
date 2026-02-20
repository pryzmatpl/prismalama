package parsers

import (
	"github.com/ollama/ollama/api"
)

// GGUFGeneralParser is a parser for general GGUF models that supports tool calling
// Many GGUF models (like Qwen, Llama3, etc.) support tools but the Ollama manifest
// doesn't always detect this correctly
type GGUFGeneralParser struct {
	hasToolSupport bool
}

// NewGGUFGeneralParser creates a new GGUF parser with tool support
func NewGGUFGeneralParser() *GGUFGeneralParser {
	return &GGUFGeneralParser{
		hasToolSupport: true,
	}
}

// Init initializes the parser with tools
func (p *GGUFGeneralParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	return tools
}

// Add processes streamed content
func (p *GGUFGeneralParser) Add(s string, done bool) (content, thinking string, calls []api.ToolCall, err error) {
	return s, "", nil, nil
}

// HasToolSupport returns true for GGUF models that support tool calling
func (p *GGUFGeneralParser) HasToolSupport() bool {
	return p.hasToolSupport
}

// HasThinkingSupport returns false by default
func (p *GGUFGeneralParser) HasThinkingSupport() bool {
	return false
}

// EnableToolSupport enables tool support
func (p *GGUFGeneralParser) EnableToolSupport() {
	p.hasToolSupport = true
}

// DisableToolSupport disables tool support
func (p *GGUFGeneralParser) DisableToolSupport() {
	p.hasToolSupport = false
}
