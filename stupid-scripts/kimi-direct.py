#!/usr/bin/env python3
"""
Direct Kimi K2.5 inference using llama-cpp-python
No Ollama, no AirLLM - just raw llama.cpp with ROCm
"""

from llama_cpp import Llama
import sys
import os

# Model configuration
MODEL_PATH = "/nvme3/AI Models/Kimi/Kimi-K2.5-Q4_K_M-00001-of-00013.gguf"

print("=" * 60)
print("Kimi K2.5 - Direct llama.cpp Inference")
print("=" * 60)
print(f"Model: {MODEL_PATH}")
print(f"Size: 579GB (Q4_K_M quantized, 13 shards)")
print("=" * 60)
print()

# Get prompt
if len(sys.argv) > 1:
    prompt = sys.argv[1]
else:
    prompt = "Hello, how are you?"

print(f"Loading model...")
print("(This may take 1-2 minutes for first shard)")
print()

try:
    # Load model with ROCm GPU acceleration
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,  # Use all GPU layers
        n_ctx=8192,       # Context length
        verbose=True,
        # Multi-file GGUF support
        model_kwargs={
            "split_mode": 1,  # Layer split mode
        }
    )
    
    print("✓ Model loaded successfully!")
    print()
    
    # Format for Kimi
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    print(f"Prompt: {prompt}")
    print("-" * 60)
    print("Generating...")
    print()
    
    # Generate
    output = llm(
        formatted_prompt,
        max_tokens=512,
        temperature=0.6,
        top_p=0.95,
        stop=["<|im_end|>", "<|im_start|>"],
        echo=False
    )
    
    print("Response:")
    print(output['choices'][0]['text'])
    print()
    
    # Print stats
    print("-" * 60)
    print(f"Tokens generated: {output['usage']['completion_tokens']}")
    print(f"Total tokens: {output['usage']['total_tokens']}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
