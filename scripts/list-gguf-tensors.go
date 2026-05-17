//go:build ignore

package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

func main() {
	path := "/home/models/blobs/sha256-f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d"
	if len(os.Args) > 1 {
		path = os.Args[1]
	}
	f, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	m, err := ggml.Decode(f, 0)
	if err != nil {
		panic(err)
	}
	fmt.Println("arch:", m.KV().Architecture())
	for _, t := range m.Tensors().Items() {
		name := t.Name
		if !strings.HasPrefix(name, "blk.") {
			continue
		}
		// sample a few layers
		if !strings.HasPrefix(name, "blk.0.") && !strings.HasPrefix(name, "blk.1.") &&
			!strings.HasPrefix(name, "blk.10.") && !strings.HasPrefix(name, "blk.20.") {
			continue
		}
		if strings.Contains(name, "ssm") || strings.Contains(name, "attn_q") ||
			strings.Contains(name, "attn_k") || strings.Contains(name, "attn_gate") ||
			strings.Contains(name, "attn_qkv") || strings.Contains(name, "attn_output") {
			fmt.Println(name)
		}
	}
}
