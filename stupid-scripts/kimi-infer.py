#!/usr/bin/env python3
"""
Direct AirLLM inference for Kimi K2.5
Bypasses Ollama's storage to handle 579GB model
"""

import os
import sys
import argparse

# Setup paths
sys.path.insert(0, '/usr/share/ollama/airllm')
os.environ['AIRLLM_COMPRESSION'] = '4bit'
os.environ['AIRLLM_DEVICE'] = 'cuda:0'

def main():
    parser = argparse.ArgumentParser(description='Run Kimi K2.5 with AirLLM')
    parser.add_argument('prompt', nargs='?', default='Hello, how are you?', help='Input prompt')
    parser.add_argument('--max-tokens', type=int, default=512, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.6, help='Temperature')
    parser.add_argument('--top-p', type=float, default=0.95, help='Top-p sampling')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Kimi K2.5 - AirLLM Inference")
    print("=" * 60)
    print(f"Model: /nvme3/AI Models/Kimi")
    print(f"Size: 579GB (Q4_K_M quantized)")
    print(f"Device: {os.environ.get('AIRLLM_DEVICE', 'cuda:0')}")
    print(f"Compression: {os.environ.get('AIRLLM_COMPRESSION', '4bit')}")
    print("=" * 60)
    print()
    
    try:
        print("Loading AirLLM...")
        from airllm import AutoModel
        
        print("Initializing Kimi model (this may take 1-2 minutes)...")
        model = AutoModel.from_pretrained(
            "/nvme3/AI Models/Kimi",
            compression="4bit",
            device="cuda:0",
            layer_offload=True,
            profiling_mode=False
        )
        
        print("✓ Model loaded successfully!")
        print()
        
        # Format prompt for Kimi
        prompt = f"<|im_start|>user\n{args.prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        print(f"Prompt: {args.prompt}")
        print("-" * 60)
        print("Generating...")
        print()
        
        # Generate
        output = model.generate(
            prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            use_cache=True
        )
        
        print("Response:")
        print(output)
        print()
        
    except ImportError as e:
        print(f"✗ Error: AirLLM not found: {e}")
        print("Please install AirLLM: pip install airllm")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
