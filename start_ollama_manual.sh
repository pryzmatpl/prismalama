#!/bin/bash

# Export ROCM environment variables
export HSA_ENABLE_SDMA=0
export HIP_VISIBLE_DEVICES=2
unset HSA_OVERRIDE_GFX_VERSION

# Start Ollama manually with proper environment
ollama serve