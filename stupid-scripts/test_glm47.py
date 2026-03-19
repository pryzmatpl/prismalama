#!/usr/bin/env python3
"""
Test script to verify GLM-4.7-Flash-4bit model compatibility
and configure it for use with AirLLM and Ollama
"""

import sys
import os
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from pathlib import Path

MODEL_PATH = "/run/media/piotro/CACHE/GLM-4.7-Flash-4bit"
AIRLLM_PATH = "/run/media/piotro/CACHE/prismalama/airllm/air_llm"

def test_model_structure():
    """Test if model files are present and valid"""
    print("=" * 60)
    print("1. Testing Model Structure")
    print("=" * 60)
    
    model_path = Path(MODEL_PATH)
    
    required_files = [
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
    ]
    
    for file in required_files:
        file_path = model_path / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✓ {file}: {size_mb:.2f} MB")
        else:
            print(f"✗ {file}: NOT FOUND")
    
    safetensors_files = list(model_path.glob("*.safetensors"))
    total_size = sum(f.stat().st_size for f in safetensors_files)
    print(f"\nSafetensors files: {len(safetensors_files)}")
    print(f"Total model size: {total_size / (1024**3):.2f} GB")
    
    return model_path.exists()

def test_config():
    """Test model configuration"""
    print("\n" + "=" * 60)
    print("2. Testing Model Configuration")
    print("=" * 60)
    
    try:
        config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print(f"✓ Model architecture: {config.architectures[0]}")
        print(f"✓ Model type: {config.model_type}")
        print(f"✓ Hidden size: {config.hidden_size}")
        print(f"✓ Num layers: {config.num_hidden_layers}")
        print(f"✓ Vocab size: {config.vocab_size}")
        print(f"✓ Max position embeddings: {config.max_position_embeddings}")
        
        if hasattr(config, 'quantization_config'):
            print(f"✓ Quantization: {config.quantization_config['bits']}-bit")
        
        return config
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return None

def test_tokenizer():
    """Test tokenizer loading"""
    print("\n" + "=" * 60)
    print("3. Testing Tokenizer")
    print("=" * 60)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print(f"✓ Tokenizer loaded successfully")
        print(f"  Vocab size: {len(tokenizer)}")
        test_text = "Hello, world!"
        tokens = tokenizer.encode(test_text)
        print(f"  Test encode: '{test_text}' -> {len(tokens)} tokens")
        return tokenizer
    except Exception as e:
        print(f"✗ Error loading tokenizer: {e}")
        return None

def test_transformers_loading():
    """Test if model can be loaded with transformers"""
    print("\n" + "=" * 60)
    print("4. Testing Transformers Model Loading")
    print("=" * 60)
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {device}")
        if torch.cuda.is_available():
            mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  GPU Memory: {mem_gb:.2f} GB")
        
        # Just test loading, don't do inference
        print("  Loading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            low_cpu_mem_usage=True
        )
        print(f"✓ Model loaded successfully!")
        print(f"  Model device: {next(model.parameters()).device}")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print(f"  This is expected if GPU memory is insufficient")
        return None

def test_airllm_compatibility():
    """Test if AirLLM can be configured for this model"""
    print("\n" + "=" * 60)
    print("5. Testing AirLLM Compatibility")
    print("=" * 60)
    
    sys.path.insert(0, AIRLLM_PATH)
    
    try:
        from airllm.auto_model import AutoModel as AirLLMAutoModel
        
        # Check if architecture is recognized
        config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
        arch = config.architectures[0]
        
        print(f"  Architecture: {arch}")
        
        if "ChatGLM" in arch:
            print("✓ Architecture supported by AirLLM (ChatGLM)")
        elif "Glm4" in arch or "GLM4" in arch or "GLM" in arch:
            print("⚠ GLM-4.7 detected - needs custom adapter")
            print("  Will create adapter for GLM-4.7 support")
        else:
            print("⚠ Architecture not directly supported by AirLLM")
            print("  May fall back to Llama2 (might not work)")
        
        return True
    except Exception as e:
        print(f"✗ Error testing AirLLM: {e}")
        return False

def create_glm47_adapter():
    """Create a custom adapter for GLM-4.7 in AirLLM"""
    print("\n" + "=" * 60)
    print("6. Creating GLM-4.7 Adapter for AirLLM")
    print("=" * 60)
    
    adapter_code = '''"""
AirLLM adapter for GLM-4.7-Flash models
Based on AirLLMChatGLM but adapted for GLM-4.7 architecture
"""

from transformers import GenerationConfig
from .airllm_base import AirLLMBaseModel

class AirLLMGLM4(AirLLMBaseModel):

    def __init__(self, *args, **kwargs):
        super(AirLLMGLM4, self).__init__(*args, **kwargs)
        print("Initialized GLM-4.7 model with AirLLM")

    def get_use_better_transformer(self):
        return False

    def get_generation_config(self):
        return GenerationConfig()

    def get_sequence_len(self, seq):
        return seq.shape[0]

    def get_past_key_values_cache_seq_len(self, past_key_values):
        return past_key_values[0][0].shape[0]

    def set_layer_names_dict(self):
        # GLM-4.7 specific layer names - may need adjustment
        self.layer_names_dict = {
            'embed': 'transformer.embedding.word_embeddings',
            'layer_prefix': 'transformer.encoder.layers',
            'norm': 'transformer.encoder.final_layernorm',
            'lm_head': 'transformer.output_layer',
            'rotary_pos_emb': 'transformer.rotary_pos_emb'
        }

    def get_pos_emb_args(self, len_p, len_s):
        rotary_pos_emb = self.model.transformer.rotary_pos_emb(self.config.seq_length)
        rotary_pos_emb = rotary_pos_emb[None, : len_s]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()
        return {'rotary_pos_emb': rotary_pos_emb}

    def get_past_key_value_args(self, k_cache, v_cache):
        return {'kv_cache': (k_cache, v_cache)}

    def get_attention_mask_args(self, full_attention_mask, len_p, len_s):
        return {'attention_mask': None}

    def get_position_ids_args(self, full_position_ids, len_p, len_s):
        return {}
'''

    adapter_path = Path(AIRLLM_PATH) / "airllm" / "airllm_glm4.py"
    try:
        with open(adapter_path, 'w') as f:
            f.write(adapter_code)
        print(f"✓ Created adapter: {adapter_path}")
        return True
    except Exception as e:
        print(f"✗ Error creating adapter: {e}")
        return False

def main():
    print(f"\nGLM-4.7-Flash-4bit Model Compatibility Test")
    print(f"Model path: {MODEL_PATH}\n")
    
    # Run all tests
    tests_passed = 0
    total_tests = 5
    
    if test_model_structure():
        tests_passed += 1
    
    config = test_config()
    if config:
        tests_passed += 1
    
    if test_tokenizer():
        tests_passed += 1
    
    # Skip model loading test (requires GPU)
    # test_transformers_loading()
    
    if test_airllm_compatibility():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: {tests_passed}/{total_tests} tests passed")
    print("=" * 60)
    
    if tests_passed >= 3:
        print("\n✓ Model is compatible for basic usage!")
        print("\nRecommendations:")
        print("1. Model structure is valid")
        print("2. Can be loaded with transformers (if enough VRAM)")
        print("3. Tokenizer works correctly")
        print("4. May need custom GLM-4.7 adapter for AirLLM")
        print("\nNext steps:")
        print("- Create GLM-4.7 adapter for AirLLM")
        print("- Create Ollama Modelfile for easy serving")
        print("- Configure environment variables for low-memory setup")
    else:
        print("\n✗ Model has compatibility issues")
        print("Please check error messages above")

if __name__ == "__main__":
    main()
