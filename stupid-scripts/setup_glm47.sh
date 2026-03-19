#!/bin/bash
# Setup script for GLM-4.7-Flash model with Ollama and AirLLM

set -e

MODEL_PATH="/run/media/piotro/CACHE/GLM-4.7-Flash-4bit"
AIRLLM_PATH="/run/media/piotro/CACHE/prismalama/airllm/air_llm"
OLLAMA_MODELS="/run/media/piotro/CACHE/airllm"

echo "============================================================"
echo "GLM-4.7-Flash Setup for Ollama + AirLLM"
echo "============================================================"

# Step 1: Create AirLLM models directory
echo ""
echo "Step 1: Creating AirLLM models directory..."
mkdir -p "$OLLAMA_MODELS"
echo "✓ Created: $OLLAMA_MODELS"

# Step 2: Create symlink to GLM-4.7 model
echo ""
echo "Step 2: Creating symlink to GLM-4.7 model..."
ln -sf "$MODEL_PATH" "$OLLAMA_MODELS/GLM-4.7-Flash-4bit"
echo "✓ Symlink: $OLLAMA_MODELS/GLM-4.7-Flash-4bit -> $MODEL_PATH"

# Step 3: Copy Modelfile
echo ""
echo "Step 3: Installing Modelfile..."
cp /run/media/piotro/CACHE/Modelfile.glm47 "$OLLAMA_MODELS/Modelfile.glm47"
echo "✓ Installed: $OLLAMA_MODELS/Modelfile.glm47"

# Step 4: Set up Python environment for AirLLM
echo ""
echo "Step 4: Setting up Python environment..."
cat > "$OLLAMA_MODELS/airllm_env.sh" << 'PYTHONENV'
#!/bin/bash
# Environment setup for AirLLM with GLM-4.7

export AIRLLM_MODEL_PATH="/run/media/piotro/CACHE/GLM-4.7-Flash-4bit"
export PYTHONPATH="/run/media/piotro/CACHE/prismalama/airllm/air_llm:$PYTHONPATH"
export OLLAMA_MODELS="/run/media/piotro/CACHE/airllm"

# GPU memory optimization
export OLLAMA_NUM_PARALLEL="1"
export OLLAMA_MAX_LOADED_MODELS="1"
export OLLAMA_CONTEXT_LENGTH="4096"
export OLLAMA_KEEP_ALIVE="5m"

# AirLLM specific
export AIRLLM_COMPRESSION="4bit"
export AIRLLM_PREFETCHING="true"

echo "AirLLM environment configured for GLM-4.7-Flash-4bit"
echo "Model path: $AIRLLM_MODEL_PATH"
echo "Ollama models: $OLLAMA_MODELS"
PYTHONENV

chmod +x "$OLLAMA_MODELS/airllm_env.sh"
echo "✓ Created: $OLLAMA_MODELS/airllm_env.sh"

# Step 5: Create test script
echo ""
echo "Step 5: Creating test script..."
cat > "$OLLAMA_MODELS/test_glm47.py" << 'PYTHONSCRIPT'
#!/usr/bin/env python3
"""
Test script for GLM-4.7-Flash with AirLLM
"""

import sys
import os

# Add AirLLM to path
sys.path.insert(0, "/run/media/piotro/CACHE/prismalama/airllm/air_llm")

from airllm import AutoModel
import torch

print("=" * 60)
print("Testing GLM-4.7-Flash with AirLLM")
print("=" * 60)

# Check CUDA availability
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    free_mem = torch.cuda.memory_allocated(device) / (1024**3)
    print(f"✓ CUDA available")
    print(f"  Device: {device}")
    print(f"  Total memory: {total_mem:.2f} GB")
    print(f"  Free memory: {total_mem - free_mem:.2f} GB")
else:
    print("✗ CUDA not available, using CPU")
    print("  Note: AirLLM requires CUDA for GPU acceleration")

print("\nLoading model (this may take a while)...")
print("Using low-memory layer-by-layer loading...")

try:
    model = AutoModel.from_pretrained(
        "/run/media/piotro/CACHE/GLM-4.7-Flash-4bit",
        compression='4bit',  # Already 4-bit, but enable AirLLM compression
        profiling_mode=True  # Show timing info
    )
    print("✓ Model loaded successfully!")
    
    # Test inference
    print("\n" + "=" * 60)
    print("Testing inference...")
    print("=" * 60)
    
    input_text = ["What is the capital of Poland?"]
    MAX_LENGTH = 128
    
    input_tokens = model.tokenizer(
        input_text,
        return_tensors="pt",
        return_attention_mask=False,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )
    
    print(f"Input: {input_text[0]}")
    print(f"Tokens: {input_tokens['input_ids'].shape}")
    
    generation_output = model.generate(
        input_tokens['input_ids'].cuda(),
        max_new_tokens=50,
        use_cache=True,
        return_dict_in_generate=True
    )
    
    output = model.tokenizer.decode(generation_output.sequences[0])
    print(f"\nOutput: {output}")
    print("\n✓ GLM-4.7-Flash is working with AirLLM!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Check if model path exists")
    print("2. Ensure sufficient GPU memory (needs ~8-12GB VRAM)")
    print("3. Verify CUDA is installed and working")
    print("4. Try reducing MAX_LENGTH if memory error occurs")
    sys.exit(1)
PYTHONSCRIPT

chmod +x "$OLLAMA_MODELS/test_glm47.py"
echo "✓ Created: $OLLAMA_MODELS/test_glm47.py"

# Step 6: Create OpenCode integration script
echo ""
echo "Step 6: Creating OpenCode integration..."
cat > "$OLLAMA_MODELS/opencode_integration.py" << 'OPENCODE'
#!/usr/bin/env python3
"""
OpenCode integration for GLM-4.7-Flash with AirLLM and Ollama
This script allows OpenCode to pick up and use the GLM-4.7 model
"""

import json
import sys
import os
from pathlib import Path

# Configuration
CONFIG = {
    "model": {
        "name": "GLM-4.7-Flash-4bit",
        "path": "/run/media/piotro/CACHE/GLM-4.7-Flash-4bit",
        "architecture": "Glm4MoeLiteForCausalLM",
        "quantization": "4bit",
        "size_gb": 15.7,
        "context_length": 4096,
        "supported_features": [
            "text_generation",
            "chat",
            "multilingual"
        ]
    },
    "inference": {
        "backend": "airllm",
        "airllm_path": "/run/media/piotro/CACHE/prismalama/airllm/air_llm",
        "ollama_models": "/run/media/piotro/CACHE/airllm",
        "compression": "4bit",
        "layer_loading": True,
        "prefetching": True
    },
    "ollama": {
        "host": "127.0.0.1:11434",
        "model_name": "glm47",
        "modelfile": "/run/media/piotro/CACHE/airllm/Modelfile.glm47"
    },
    "performance": {
        "num_parallel": 1,
        "max_loaded_models": 1,
        "gpu_memory_optimization": True,
        "context_length": 4096,
        "batch_size": 8
    }
}

def main():
    """Write configuration for OpenCode"""
    
    config_path = Path("/run/media/piotro/CACHE/airllm/opencode_config.json")
    
    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    print("=" * 60)
    print("OpenCode Configuration Created")
    print("=" * 60)
    print(f"Config file: {config_path}")
    print("\nModel Information:")
    print(f"  Name: {CONFIG['model']['name']}")
    print(f"  Path: {CONFIG['model']['path']}")
    print(f"  Size: {CONFIG['model']['size_gb']} GB")
    print(f"  Context: {CONFIG['model']['context_length']}")
    print("\nInference Backend:")
    print(f"  Backend: {CONFIG['inference']['backend']}")
    print(f"  AirLLM: {CONFIG['inference']['airllm_path']}")
    print(f"  Ollama: {CONFIG['ollama']['host']}")
    print("\n" + "=" * 60)
    print("OpenCode can now use GLM-4.7-Flash model!")
    print("=" * 60)
    print("\nUsage in OpenCode:")
    print("  1. Load config: load_config('/run/media/piotro/CACHE/airllm/opencode_config.json')")
    print("  2. Initialize model: init_model()")
    print("  3. Generate text: generate(prompt)")

if __name__ == "__main__":
    main()
OPENCODE

chmod +x "$OLLAMA_MODELS/opencode_integration.py"
echo "✓ Created: $OLLAMA_MODELS/opencode_integration.py"

# Step 7: Set permissions
echo ""
echo "Step 7: Setting permissions..."
chmod -R 755 "$OLLAMA_MODELS"
echo "✓ Permissions set"

# Final summary
echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Test the setup:"
echo "   cd /run/media/piotro/CACHE/airllm"
echo "   source ./airllm_env.sh"
echo "   ./test_glm47.py"
echo ""
echo "2. Create OpenCode config:"
echo "   cd /run/media/piotro/CACHE/airllm"
echo "   ./opencode_integration.py"
echo ""
echo "3. If Ollama is installed, create model:"
echo "   ollama create glm47 -f Modelfile.glm47"
echo ""
echo "4. Start Ollama service:"
echo "   systemctl start ollama"
echo ""
echo "5. Run model with Ollama:"
echo "   ollama run glm47"
echo ""
echo "============================================================"
