# Adaptive installation script for rppg2ppg (Windows PowerShell)
# Usage: .\setup\install.ps1 [-Cpu] [-Check]
#
# Note: Activate your virtual environment first!
#   conda activate myenv
#   .\venv\Scripts\Activate.ps1

param(
    [switch]$Cpu,
    [switch]$Check,
    [switch]$SkipPytorch,
    [switch]$SkipMamba
)

$ErrorActionPreference = "Stop"

Write-Host "=============================================="
Write-Host "  rppg2ppg Installer (Windows PowerShell)"
Write-Host "=============================================="
Write-Host ""

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[OK] $pythonVersion"
} catch {
    Write-Host "[!] Python not found. Please install Python 3.10+"
    exit 1
}

# Check CUDA (nvidia-smi)
$cudaVersion = $null
try {
    $nvidiaSmi = nvidia-smi 2>&1
    if ($nvidiaSmi -match "CUDA Version:\s*([\d.]+)") {
        $cudaVersion = $Matches[1]
        Write-Host "[OK] CUDA: $cudaVersion"
    }
} catch {
    Write-Host "[!] CUDA not detected (CPU mode)"
}

# Check GPU
try {
    $gpuInfo = nvidia-smi --query-gpu=name --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] GPU: $gpuInfo"
    }
} catch {
    # No GPU
}

Write-Host ""
Write-Host "Running Python installer..."
Write-Host ""

# Build arguments
$args = @()
if ($Cpu) { $args += "--cpu" }
if ($Check) { $args += "--check" }
if ($SkipPytorch) { $args += "--skip-pytorch" }
if ($SkipMamba) { $args += "--skip-mamba" }

# Get script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$installPy = Join-Path $scriptDir "install.py"

# Run Python installer
if ($args.Count -gt 0) {
    python $installPy @args
} else {
    python $installPy
}

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[!] Installation failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[OK] Installation complete!" -ForegroundColor Green
