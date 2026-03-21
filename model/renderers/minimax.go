package renderers

import (
	"strings"

	"github.com/ollama/ollama/api"
)

type MiniMaxRenderer struct{}

func (r *MiniMaxRenderer) Render(messages []api.Message, tools []api.Tool, thinkValue *api.ThinkValue) (string, error) {
	var sb strings.Builder

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			sb.WriteString("]~b]system\n")
			sb.WriteString(msg.Content)
			sb.WriteString("\n]~b]ai\n\n")
		case "user":
			sb.WriteString("]~b]user\n")
			sb.WriteString(msg.Content)
			sb.WriteString("\n]~b]ai\n\n")
		case "assistant":
			sb.WriteString("]~b]ai\n")
			sb.WriteString(msg.Content)
			sb.WriteString("\n")
		case "tool":
			sb.WriteString("]~b]tool\n")
			sb.WriteString(msg.Content)
			sb.WriteString("\n")
		}
	}

	return sb.String(), nil
}
