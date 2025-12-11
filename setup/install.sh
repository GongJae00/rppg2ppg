#!/bin/bash
# Adaptive installation script for rppg2ppg
# Usage: bash setup/install.sh [--cpu]
#
# Note: Activate your virtual environment first!
#   pyenv activate rppg-310  (or your env name)
#   conda activate myenv
#   source venv/bin/activate

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "  rppg2ppg Installer"
echo "=============================================="
echo "Project: $PROJECT_DIR"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo "[!] Python 3.10+ required. Found: $PYTHON_VERSION"
    exit 1
fi
echo "[OK] Python $PYTHON_VERSION"

# Check for --cpu flag
CPU_ONLY=false
if [ "$1" = "--cpu" ]; then
    CPU_ONLY=true
    echo "[!] CPU-only mode"
fi

# Detect CUDA
CUDA_VERSION=""
if [ "$CPU_ONLY" = false ]; then
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[\d.]+')
        echo "[OK] CUDA (nvcc): $CUDA_VERSION"
    elif command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[\d.]+')
        echo "[OK] CUDA (driver): $CUDA_VERSION"
    else
        echo "[!] CUDA not detected, using CPU"
    fi
fi

# Run Python installer
echo ""
echo "Running Python installer..."
echo ""

if [ "$CPU_ONLY" = true ]; then
    python3 "$SCRIPT_DIR/install.py" --cpu
else
    python3 "$SCRIPT_DIR/install.py"
fi
