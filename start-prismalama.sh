#!/bin/bash
cd /home/prizm/prismalama
exec env OLLAMA_MODELS=/home/models OLLAMA_HOST=0.0.0.0:11434 ./prismalama-ollama serve
