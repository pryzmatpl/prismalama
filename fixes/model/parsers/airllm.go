package parsers

import (
	"github.com/ollama/ollama/api"
)

// AirLLMParser is a parser for AirLLM models that supports tool calling
type AirLLMParser struct {
	hasToolSupport bool
}

// NewAirLLMParser creates a new AirLLM parser with tool support enabled
func NewAirLLMParser() *AirLLMParser {
	return &AirLLMParser{
		hasToolSupport: true,
	}
}

// Init initializes the parser with tools
func (p *AirLLMParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	// AirLLM models can handle tools, return them as-is
	return tools
}

// Add processes streamed content and returns parsed content
func (p *AirLLMParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	// For AirLLM models, we pass through the content
	// Tool calls should be parsed from the model output
	return s, "", nil, nil
}

// HasToolSupport returns true as AirLLM models support tool calling
func (p *AirLLMParser) HasToolSupport() bool {
	return p.hasToolSupport
}

// HasThinkingSupport returns false as AirLLM models don't support thinking mode yet
func (p *AirLLMParser) HasThinkingSupport() bool {
	return false
}

// EnableToolSupport enables tool support for this parser
func (p *AirLLMParser) EnableToolSupport() {
	p.hasToolSupport = true
}

// DisableToolSupport disables tool support for this parser
func (p *AirLLMParser) DisableToolSupport() {
	p.hasToolSupport = false
}
