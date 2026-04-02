#!/bin/bash
# Verify prismalama + Ollama + ROCm + AirLLM configuration for Kimi
# The Ultimate Potato Hardware Setup Check 🥔✨

echo "======================================================================"
echo "  PRISMALAMA + OLLAMA + ROCM + AIRLLM VERIFICATION"
echo "  Kimi on Potato Hardware - Humanity Saving Configuration"
echo "======================================================================"
echo ""

ERRORS=0

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ((ERRORS++))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

echo "[1/8] Checking Ollama Installation..."
if command -v ollama &> /dev/null; then
    check_pass "Ollama installed: $(ollama --version 2>&1 | head -1)"
else
    check_fail "Ollama not found in PATH"
fi

echo ""
echo "[2/8] Checking ROCm/GPU Support..."
if command -v rocminfo &> /dev/null; then
    GPU=$(rocminfo 2>/dev/null | grep -m1 "Name:" | awk '{print $2}')
    if [ -n "$GPU" ]; then
        check_pass "ROCm GPU detected: $GPU"
    else
        check_warn "ROCm installed but GPU not detected"
    fi
else
    check_warn "rocminfo not found"
fi

# Check if Ollama detects GPU
if systemctl is-active --quiet ollama; then
    GPU_LOG=$(journalctl -u ollama --since "1 minute ago" 2>/dev/null | grep -i "ROCm\|AMD\|gpu" | tail -3)
    if [ -n "$GPU_LOG" ]; then
        check_pass "Ollama detects GPU"
    else
        check_warn "Ollama GPU detection unclear"
    fi
fi

echo ""
echo "[3/8] Checking Model Storage Locations..."
if [ -d "/nvme3/ollama-models" ]; then
    SIZE=$(du -sh /nvme3/ollama-models 2>/dev/null | cut -f1)
    check_pass "Ollama models on /nvme3: $SIZE"
else
    check_fail "/nvme3/ollama-models not found"
fi

if [ -d "/nvme3/models" ]; then
    SIZE=$(du -sh /nvme3/models 2>/dev/null | cut -f1)
    check_pass "Raw models on /nvme3: $SIZE"
else
    check_warn "/nvme3/models not found"
fi

if [ -d "/nvme3/AI Models/Kimi" ]; then
    SIZE=$(du -sh "/nvme3/AI Models/Kimi" 2>/dev/null | cut -f1)
    check_pass "Kimi model found: $SIZE"
else
    check_fail "Kimi model not found on /nvme3"
fi

echo ""
echo "[4/8] Checking Systemd Configuration..."
if [ -f "/etc/systemd/system/ollama.service.d/override.conf" ]; then
    if grep -q "/nvme3" /etc/systemd/system/ollama.service.d/override.conf; then
        check_pass "Systemd configured for /nvme3"
    else
        check_fail "Systemd not configured for /nvme3"
    fi
else
    check_fail "Systemd override not found"
fi

if [ -f "/etc/default/ollama" ]; then
    if grep -q "/nvme3" /etc/default/ollama; then
        check_pass "Environment file configured for /nvme3"
    else
        check_fail "Environment file not configured for /nvme3"
    fi
else
    check_fail "/etc/default/ollama not found"
fi

echo ""
echo "[5/8] Checking AirLLM Integration..."
if [ -d "/usr/share/ollama/airllm" ]; then
    check_pass "AirLLM installed at /usr/share/ollama/airllm"
    
    if [ -f "/usr/share/ollama/airllm/airllm_runner.py" ]; then
        check_pass "AirLLM runner present"
    else
        check_warn "AirLLM runner not found"
    fi
else
    check_warn "AirLLM not installed (optional)"
fi

# Check environment variables
if [ -f "/etc/default/ollama" ]; then
    if grep -q "AIRLLM_COMPRESSION" /etc/default/ollama; then
        check_pass "AirLLM compression configured"
    fi
fi

echo ""
echo "[6/8] Checking prismalama Build Artifacts..."
cd /sda2/prismalama

if [ -f "build/pkg/etc/default/ollama" ]; then
    if grep -q "/nvme3" build/pkg/etc/default/ollama; then
        check_pass "prismalama build configured for /nvme3"
    else
        check_warn "prismalama build still has old paths"
    fi
fi

if [ -f "PKGBUILD" ]; then
    check_pass "PKGBUILD present for Arch Linux"
else
    check_warn "PKGBUILD not found"
fi

echo ""
echo "[7/8] Checking Ollama Service Status..."
if systemctl is-active --quiet ollama; then
    check_pass "Ollama service is running"
    
    # Check if it can see models
    MODEL_COUNT=$(ollama list 2>/dev/null | grep -c ":" || echo "0")
    if [ "$MODEL_COUNT" -gt 0 ]; then
        check_pass "Ollama sees $MODEL_COUNT models"
    else
        check_warn "No models registered yet"
    fi
else
    check_fail "Ollama service is not running"
    echo ""
    echo "  To start: sudo systemctl start ollama"
fi

echo ""
echo "[8/8] Checking Available Space..."
NVME_FREE=$(df -h /nvme3 2>/dev/null | tail -1 | awk '{print $4}')
if [ -n "$NVME_FREE" ]; then
    check_pass "/nvme3 available space: $NVME_FREE"
else
    check_fail "Cannot determine /nvme3 space"
fi

echo ""
echo "======================================================================"
echo "  VERIFICATION SUMMARY"
echo "======================================================================"
echo ""

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "Your Potato Hardware Setup is ready to save humanity! 🥔🚀"
    echo ""
    echo "Quick Start:"
    echo "  1. Register Kimi: ollama create kimi -f /sda2/Modelfile.kimi"
    echo "  2. Run Kimi: ollama run kimi 'Hello, how can you help humanity?'"
    echo "  3. Use with opencode: opencode run -m ollama/kimi"
    echo ""
    echo "Architecture:"
    echo "  Ollama (API) → prismalama (ROCm runner) → llama.cpp → Kimi (579GB)"
    echo "  With AirLLM layer offloading for potato hardware!"
else
    echo -e "${RED}✗ $ERRORS check(s) failed${NC}"
    echo ""
    echo "Run the fix script:"
    echo "  sudo bash /sda2/update-prismalama-nvme.sh"
    echo ""
    echo "Then re-verify:"
    echo "  sudo bash /sda2/verify-prismalama-setup.sh"
fi

echo ""
echo "======================================================================"
