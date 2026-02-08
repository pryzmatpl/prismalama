# Ollama with AirLLM Integration - Arch Linux Package

## Overview
This package provides an Arch Linux compatible build of Ollama integrated with AirLLM, allowing you to run large language models on systems with limited GPU memory.

## Package Details
- **Package Name**: ollama-airllm
- **Version**: v0.4.1.r5052.e102207f-1
- **Architecture**: x86_64
- **Package File**: `ollama-airllm-v0.4.1.r5052.e102207f-1-x86_64.pkg.tar.zst`
- **Size**: 53M

## Key Features
1. **AirLLM Integration**: Full AirLLM Python package included at `/usr/share/ollama/airllm`
2. **Custom Models Path**: Configured to use `/run/media/piotro/CACHE/airllm` for models storage
3. **Systemd Service**: Includes properly configured systemd service
4. **Conflict Resolution**: Replaces any existing ollama installation

## Installation

### Step 1: Stop existing ollama service (if installed)
```bash
sudo systemctl stop ollama
sudo systemctl disable ollama
```

### Step 2: Remove existing ollama (if installed)
```bash
sudo pacman -Rns ollama
```

### Step 3: Install the new package
```bash
cd /run/media/piotro/CACHE/prismalama
sudo pacman -U ollama-airllm-v0.4.1.r5052.e102207f-1-x86_64.pkg.tar.zst
```

### Step 4: Verify installation
```bash
ollama --version
ls -la /usr/share/ollama/airllm/
```

### Step 5: Start the service
```bash
sudo systemctl start ollama
sudo systemctl enable ollama
```

### Step 6: Verify service is running
```bash
systemctl status ollama
```

## Configuration

### Environment Variables
The main configuration is located at `/etc/default/ollama`:

```bash
# Models directory (automatically set to AirLLM path)
export OLLAMA_MODELS="/run/media/piotro/CACHE/airllm"

# Optional: Configure host
# export OLLAMA_HOST="127.0.0.1:11434"

# Optional: GPU configuration
# export CUDA_VISIBLE_DEVICES="0"
# export HIP_VISIBLE_DEVICES="0"
# export OLLAMA_VULKAN="1"

# Optional: Performance tuning
# export OLLAMA_NUM_PARALLEL="4"
# export OLLAMA_MAX_LOADED_MODELS="3"
# export OLLAMA_MAX_QUEUE="512"
# export OLLAMA_KEEP_ALIVE="5m"
```

### Service Configuration
Systemd service file: `/usr/lib/systemd/system/ollama.service`
- **User**: ollama (created automatically)
- **Models Path**: `/run/media/piotro/CACHE/airllm`
- **ReadWritePaths**: Includes `/run/media/piotro/CACHE/airllm` and `/var/lib/ollama`

## Directory Structure
After installation, the following structure is created:

```
/usr/bin/ollama                          - Main binary
/usr/lib/systemd/system/ollama.service    - Systemd service
/usr/lib/sysusers.d/ollama.conf          - User configuration
/etc/default/ollama                      - Environment variables
/usr/share/ollama/airllm/                - AirLLM Python package
/run/media/piotro/CACHE/airllm           - Models directory
```

## AirLLM Usage

The AirLLM Python package is installed at `/usr/share/ollama/airllm`. To use it:

```bash
# Add AirLLM to Python path
export PYTHONPATH="/usr/share/ollama/airllm:$PYTHONPATH"

# Import in your Python scripts
from airllm import AutoModel

# Example usage
model = AutoModel.from_pretrained("/path/to/model")
```

## Rebuilding the Package

If you need to rebuild the package:

```bash
cd /run/media/piotro/CACHE/prismalama
./build.sh
```

This will create a new package file with the current git commit.

## Troubleshooting

### Service won't start
```bash
# Check logs
sudo journalctl -u ollama -n 50

# Check if models directory has correct permissions
ls -la /run/media/piotro/CACHE/airllm
sudo chown -R ollama:ollama /run/media/piotro/CACHE/airllm
```

### AirLLM not accessible
```bash
# Verify AirLLM installation
ls -la /usr/share/ollama/airllm/

# Test Python import
cd /usr/share/ollama/airllm
python3 -c "from airllm import AutoModel; print('OK')"
```

### Port conflicts
Edit `/etc/default/ollama` and set:
```bash
export OLLAMA_HOST="127.0.0.1:11434"
```

Then restart:
```bash
sudo systemctl restart ollama
```

## Building from Source (Alternative PKGBUILD)

A standard Arch Linux PKGBUILD is also provided:

```bash
cd /run/media/piotro/CACHE/prismalama
makepkg -si
```

Note: This may take longer as it builds all components from scratch.

## Testing the Installation

After installation, test with:

```bash
# Test Ollama server
curl http://127.0.0.1:11434/api/version

# Test with a simple model download (if you have models)
ollama list
```

## Overwriting Existing Installation

This package is designed to replace any existing ollama installation:
- Conflicts: `ollama`
- Provides: `ollama`

When you install this package, it will:
1. Replace the ollama binary
2. Update the systemd service configuration
3. Point to the AirLLM models directory
4. Include the full AirLLM Python package

## Support

For issues related to:
- **Ollama**: https://github.com/ollama/ollama
- **AirLLM**: https://github.com/lyogavin/airllm
- **This Package**: /run/media/piotro/CACHE/prismalama

## License

This package includes:
- Ollama: MIT License
- AirLLM: MIT License

See individual license files in `/usr/share/licenses/` for details.
