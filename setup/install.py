#!/usr/bin/env python3
"""
Adaptive installation script for rppg2ppg project.
Cross-platform support: Windows, macOS, Linux

Automatically detects CUDA/MPS and installs appropriate dependencies.

Usage:
    python setup/install.py           # Full install
    python setup/install.py --cpu     # Force CPU-only
    python setup/install.py --check   # Check environment only
"""

import subprocess
import sys
import argparse
import os
import platform

# Add setup directory to path for detect_env
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_env import detect_environment, print_environment_report, get_platform


def run_pip(args: list, check: bool = True) -> bool:
    """Run pip command."""
    cmd = [sys.executable, "-m", "pip"] + args
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if check and result.returncode != 0:
        print(f"[!] Command failed with code {result.returncode}")
        return False
    return result.returncode == 0


def install_pytorch(cuda_version: str = None, force_cpu: bool = False, plat: str = "linux"):
    """Install PyTorch with appropriate CUDA/MPS support."""
    print("\n[1/3] Installing PyTorch...")

    # macOS: MPS 지원 (Apple Silicon)
    if plat == "macos":
        print("  -> Installing for macOS (MPS support)")
        return run_pip(["install", "torch", "torchvision", "torchaudio"])

    # CPU only
    if force_cpu or cuda_version is None:
        print("  -> Installing CPU version")
        return run_pip([
            "install", "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ])

    # Parse CUDA version
    cuda_parts = cuda_version.split(".")
    cuda_major = int(cuda_parts[0])
    cuda_minor = int(cuda_parts[1]) if len(cuda_parts) > 1 else 0

    # Select appropriate PyTorch CUDA version
    if cuda_major >= 12 and cuda_minor >= 4:
        cuda_tag = "cu124"
    elif cuda_major >= 12 and cuda_minor >= 1:
        cuda_tag = "cu121"
    elif cuda_major >= 11 and cuda_minor >= 8:
        cuda_tag = "cu118"
    else:
        cuda_tag = "cu118"  # Fallback to oldest supported

    print(f"  -> Installing for CUDA {cuda_version} (using {cuda_tag})")
    return run_pip([
        "install", "torch", "torchvision", "torchaudio",
        "--index-url", f"https://download.pytorch.org/whl/{cuda_tag}"
    ])


def install_base_requirements():
    """Install base requirements (non-CUDA dependent)."""
    print("\n[2/3] Installing base requirements...")

    packages = [
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "tqdm",
    ]

    return run_pip(["install"] + packages)


def install_mamba(has_cuda: bool, plat: str = "linux"):
    """Install mamba-ssm and causal-conv1d."""
    print("\n[3/3] Installing mamba-ssm...")

    # Linux에서만 mamba-ssm 사용 가능
    if plat != "linux":
        if plat == "macos":
            print("  [!] mamba-ssm is not available on macOS (requires CUDA)")
        else:  # windows
            print("  [!] mamba-ssm is not available on Windows (Linux only)")
        print("  [!] Available models: lstm, bilstm, rnn, transformer, physdiff")
        return True

    if not has_cuda:
        print("  [!] Skipping mamba-ssm (requires CUDA)")
        print("  [!] Mamba models will not be available")
        return True

    # causal-conv1d is a dependency of mamba-ssm
    success = run_pip(["install", "causal-conv1d"], check=False)
    if not success:
        print("  [!] causal-conv1d installation failed")
        print("  [!] You may need to install it manually:")
        print("      pip install causal-conv1d --no-build-isolation")

    success = run_pip(["install", "mamba-ssm"], check=False)
    if not success:
        print("\n  [!] mamba-ssm installation failed")
        print("  [!] Common fixes:")
        print("      1. Ensure CUDA toolkit is installed (nvcc available)")
        print("      2. Try: pip install mamba-ssm --no-build-isolation")
        print("      3. Check PyTorch CUDA version matches system CUDA")
        return False

    return True


def verify_installation(plat: str = "linux"):
    """Verify installation by importing key modules."""
    print("\n[Verification]")

    modules = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "Matplotlib"),
    ]

    all_ok = True
    for module, name in modules:
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError as e:
            print(f"  [X] {name}: {e}")
            all_ok = False

    # Check mamba separately (Linux only)
    if plat == "linux":
        try:
            import mamba_ssm
            print(f"  [OK] mamba-ssm")
        except ImportError:
            print(f"  [!] mamba-ssm (optional, requires CUDA)")

    # Check PyTorch acceleration
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  [OK] PyTorch CUDA: {torch.version.cuda}")
            print(f"  [OK] GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print(f"  [OK] PyTorch MPS: Apple Silicon")
        else:
            print(f"  [!] PyTorch: CPU only")
    except Exception as e:
        print(f"  [X] PyTorch check failed: {e}")

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive installation for rppg2ppg project (Windows/macOS/Linux)"
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU-only installation"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Check environment only, don't install"
    )
    parser.add_argument(
        "--skip-pytorch", action="store_true",
        help="Skip PyTorch installation (use existing)"
    )
    parser.add_argument(
        "--skip-mamba", action="store_true",
        help="Skip mamba-ssm installation"
    )
    args = parser.parse_args()

    # Get platform
    plat = get_platform()

    print("=" * 60)
    print("  rppg2ppg Adaptive Installer")
    print("=" * 60)

    # Detect environment
    env = detect_environment()
    print_environment_report(env)

    if args.check:
        print("\n[Check mode] No packages installed.")
        sys.exit(0 if env["python_ok"] else 1)

    # Check Python version
    if not env["python_ok"]:
        print("\n[!] Python 3.10+ required. Please upgrade Python.")
        sys.exit(1)

    # Upgrade pip first
    print("\n[0/3] Upgrading pip...")
    run_pip(["install", "--upgrade", "pip"], check=False)

    # Determine CUDA version to use
    cuda_version = None if args.cpu else env["cuda_version"]

    # Install PyTorch
    if args.skip_pytorch and env["torch_version"]:
        print("\n[1/3] Skipping PyTorch (already installed)")
    else:
        if not install_pytorch(cuda_version, force_cpu=args.cpu, plat=plat):
            print("\n[!] PyTorch installation failed")
            sys.exit(1)

    # Install base requirements
    if not install_base_requirements():
        print("\n[!] Base requirements installation failed")
        sys.exit(1)

    # Install mamba-ssm (Linux only)
    if args.skip_mamba or plat != "linux":
        if plat == "macos":
            print("\n[3/3] Skipping mamba-ssm (not available on macOS)")
        elif plat == "windows":
            print("\n[3/3] Skipping mamba-ssm (not available on Windows)")
        else:
            print("\n[3/3] Skipping mamba-ssm")
    else:
        # Re-check CUDA after PyTorch install
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False

        install_mamba(has_cuda and not args.cpu, plat=plat)

    # Verify
    print("\n" + "=" * 60)
    if verify_installation(plat=plat):
        print("\n[OK] Installation complete!")

        # auto_profile.py 실행하여 환경변수 출력
        print("\n" + "=" * 60)
        print("  Environment Profile (auto_profile.py)")
        print("=" * 60)
        auto_profile_path = os.path.join(os.path.dirname(__file__), "auto_profile.py")
        if os.path.exists(auto_profile_path):
            subprocess.run([sys.executable, auto_profile_path], check=False)
            print("\n[Tip] Apply these environment variables before training:")
            if plat == "windows":
                print('  PowerShell: . { python setup/auto_profile.py | ForEach { $_.Split("=") | Set-Variable } }')
            else:
                print('  eval "$(python setup/auto_profile.py)"')
        else:
            print("  [!] auto_profile.py not found")

        print("\nNext steps:")
        print("  1. cd /path/to/rppg2ppg")
        if plat == "linux":
            print('  2. eval "$(python setup/auto_profile.py)"  # Set environment')
            print("  3. python tools/tool_A_baseline.py --config configs/mamba.toon")
        else:
            print("  2. python tools/tool_A_baseline.py --config configs/lstm.toon")
            print(f"     (Note: Mamba models not available on {plat})")
        print("  4. Or run all: python run_all.py")
    else:
        print("\n[!] Some packages failed to install")
        print("    Please check the errors above and install manually")
        sys.exit(1)


if __name__ == "__main__":
    main()
