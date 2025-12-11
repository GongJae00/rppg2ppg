#!/usr/bin/env python3
"""
Environment detection utility for rppg2ppg project.
Cross-platform support: Windows, macOS, Linux

Detects CUDA version, GPU info, and recommends PyTorch installation.
"""

import subprocess
import sys
import os
import re
import platform
from typing import Optional, Tuple


def get_platform() -> str:
    """Get current platform: 'windows', 'macos', 'linux'."""
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    return system


def get_python_version() -> Tuple[int, int, int]:
    """Get current Python version."""
    return sys.version_info[:3]


def run_command(cmd: str, shell: bool = True) -> Optional[str]:
    """Run shell command and return output (cross-platform)."""
    try:
        # Windows에서는 shell=True가 필요
        result = subprocess.run(
            cmd,
            shell=shell,
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "LANG": "en_US.UTF-8"}  # 영어 출력 강제
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def get_nvidia_smi_path() -> str:
    """Get nvidia-smi path based on platform."""
    plat = get_platform()

    if plat == "windows":
        # Windows 기본 경로들
        possible_paths = [
            "nvidia-smi",  # PATH에 있는 경우
            r"C:\Windows\System32\nvidia-smi.exe",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        ]
        for path in possible_paths:
            if run_command(f'"{path}" --version') is not None:
                return f'"{path}"'
        return "nvidia-smi"

    return "nvidia-smi"  # Linux/macOS


def get_cuda_version_nvcc() -> Optional[str]:
    """Get CUDA version from nvcc."""
    output = run_command("nvcc --version")
    if output:
        match = re.search(r"release (\d+\.\d+)", output)
        if match:
            return match.group(1)
    return None


def get_cuda_version_nvidia_smi() -> Optional[str]:
    """Get CUDA version from nvidia-smi (driver's max supported CUDA)."""
    nvidia_smi = get_nvidia_smi_path()
    output = run_command(nvidia_smi)
    if output:
        match = re.search(r"CUDA Version:\s*(\d+\.\d+)", output)
        if match:
            return match.group(1)
    return None


def get_gpu_info() -> list:
    """Get list of available GPUs."""
    plat = get_platform()

    # NVIDIA GPU
    nvidia_smi = get_nvidia_smi_path()
    output = run_command(f"{nvidia_smi} --query-gpu=name,memory.total --format=csv,noheader")
    if output:
        gpus = []
        for line in output.strip().split("\n"):
            if line.strip():
                gpus.append(f"[NVIDIA] {line.strip()}")
        return gpus

    # macOS: Apple Silicon (MPS)
    if plat == "macos":
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # Apple Silicon 모델 확인
                chip_info = run_command("sysctl -n machdep.cpu.brand_string")
                return [f"[Apple MPS] {chip_info or 'Apple Silicon'}"]
        except ImportError:
            pass

        # system_profiler로 GPU 정보 확인
        output = run_command("system_profiler SPDisplaysDataType 2>/dev/null | grep 'Chipset Model'")
        if output:
            return [f"[Apple] {output.replace('Chipset Model:', '').strip()}"]

    return []


def check_torch_installed() -> Optional[str]:
    """Check if PyTorch is installed and return version."""
    try:
        import torch
        return torch.__version__
    except ImportError:
        return None


def check_torch_cuda() -> Tuple[bool, Optional[str]]:
    """Check if PyTorch has CUDA support."""
    try:
        import torch
        if torch.cuda.is_available():
            return True, torch.version.cuda
        return False, None
    except ImportError:
        return False, None


def check_torch_mps() -> bool:
    """Check if PyTorch has MPS (Apple Silicon) support."""
    try:
        import torch
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except ImportError:
        return False


def check_mamba_installed() -> Optional[str]:
    """Check if mamba-ssm is installed."""
    try:
        import mamba_ssm
        return getattr(mamba_ssm, "__version__", "installed")
    except ImportError:
        return None


def get_pytorch_install_command(cuda_version: Optional[str], plat: str) -> str:
    """Get recommended PyTorch installation command based on CUDA version and platform."""

    # macOS: MPS 지원 버전
    if plat == "macos":
        return "pip install torch torchvision torchaudio"

    # CPU only
    if cuda_version is None:
        return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"

    cuda_major_minor = cuda_version.split(".")
    cuda_major = int(cuda_major_minor[0])
    cuda_minor = int(cuda_major_minor[1]) if len(cuda_major_minor) > 1 else 0

    # PyTorch CUDA version mapping (2024-2025)
    if cuda_major >= 12 and cuda_minor >= 4:
        return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
    elif cuda_major >= 12 and cuda_minor >= 1:
        return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    elif cuda_major >= 11 and cuda_minor >= 8:
        return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    elif cuda_major >= 11:
        return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else:
        return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"


def detect_environment() -> dict:
    """Detect full environment and return info dict."""
    plat = get_platform()
    python_ver = get_python_version()
    cuda_nvcc = get_cuda_version_nvcc()
    cuda_smi = get_cuda_version_nvidia_smi()
    cuda_version = cuda_nvcc or cuda_smi
    gpus = get_gpu_info()
    torch_ver = check_torch_installed()
    torch_cuda_available, torch_cuda_ver = check_torch_cuda()
    torch_mps_available = check_torch_mps()
    mamba_ver = check_mamba_installed()

    return {
        "platform": plat,
        "python_version": f"{python_ver[0]}.{python_ver[1]}.{python_ver[2]}",
        "python_ok": python_ver >= (3, 10, 0),
        "cuda_nvcc": cuda_nvcc,
        "cuda_nvidia_smi": cuda_smi,
        "cuda_version": cuda_version,
        "gpus": gpus,
        "torch_version": torch_ver,
        "torch_cuda_available": torch_cuda_available,
        "torch_cuda_version": torch_cuda_ver,
        "torch_mps_available": torch_mps_available,
        "mamba_version": mamba_ver,
        "pytorch_install_cmd": get_pytorch_install_command(cuda_version, plat),
    }


def print_environment_report(env: dict):
    """Print formatted environment report."""
    print("\n" + "=" * 60)
    print("  rppg2ppg Environment Detection Report")
    print("=" * 60)

    # Platform
    plat_names = {"windows": "Windows", "macos": "macOS", "linux": "Linux"}
    print(f"\n[Platform]")
    print(f"  OS: {plat_names.get(env['platform'], env['platform'])}")

    # Python
    py_status = "[OK]" if env["python_ok"] else "[!] (3.10+ required)"
    print(f"\n[Python]")
    print(f"  Version: {env['python_version']} {py_status}")

    # CUDA
    print(f"\n[CUDA]")
    if env["cuda_version"]:
        print(f"  NVCC Version: {env['cuda_nvcc'] or 'not found'}")
        print(f"  Driver CUDA: {env['cuda_nvidia_smi'] or 'not found'}")
        print(f"  Detected: {env['cuda_version']}")
    elif env["platform"] == "macos":
        print("  CUDA: Not available on macOS")
        print("  Alternative: Apple MPS (Metal Performance Shaders)")
    else:
        print("  CUDA: Not detected (CPU mode)")

    # GPUs
    print(f"\n[GPU]")
    if env["gpus"]:
        for i, gpu in enumerate(env["gpus"]):
            print(f"  [{i}] {gpu}")
    else:
        print("  No GPU detected")

    # PyTorch
    print(f"\n[PyTorch]")
    if env["torch_version"]:
        if env["torch_cuda_available"]:
            accel = f"CUDA {env['torch_cuda_version']}"
        elif env["torch_mps_available"]:
            accel = "MPS (Apple Silicon)"
        else:
            accel = "CPU only"
        print(f"  Version: {env['torch_version']} ({accel})")
    else:
        print("  Not installed")
        print(f"  Recommended: {env['pytorch_install_cmd']}")

    # mamba-ssm (Linux only)
    print(f"\n[mamba-ssm]")
    if env["mamba_version"]:
        print(f"  Version: {env['mamba_version']}")
    else:
        print("  Not installed")
        if env["platform"] == "linux" and env["cuda_version"]:
            print("  Install: pip install mamba-ssm causal-conv1d")
        elif env["platform"] != "linux":
            plat_name = "macOS" if env["platform"] == "macos" else "Windows"
            print(f"  [!] mamba-ssm is Linux only (not available on {plat_name})")
            print("      Use other models: lstm, bilstm, rnn, transformer, physdiff")
        else:
            print("  [!] Requires CUDA for GPU acceleration")

    print("\n" + "=" * 60)

    # Warnings
    warnings = []
    if not env["python_ok"]:
        warnings.append("Python 3.10+ required")
    if env["torch_version"] and not env["torch_cuda_available"] and env["cuda_version"]:
        warnings.append("PyTorch installed without CUDA support, but CUDA is available")
    if not env["mamba_version"] and env["torch_cuda_available"]:
        warnings.append("mamba-ssm not installed (required for Mamba models)")
    if env["platform"] == "macos" and not env["torch_mps_available"] and env["torch_version"]:
        warnings.append("MPS not available - update PyTorch for Apple Silicon acceleration")

    if warnings:
        print("\n[!] Warnings:")
        for w in warnings:
            print(f"    - {w}")
        print()


if __name__ == "__main__":
    env = detect_environment()
    print_environment_report(env)

    # Exit code: 0 if ready, 1 if missing dependencies
    if env["python_ok"] and env["torch_version"]:
        sys.exit(0)
    sys.exit(1)
