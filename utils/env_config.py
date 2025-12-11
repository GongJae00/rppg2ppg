"""
Environment Configuration Utility

auto_profile.py에서 설정한 환경변수를 읽어서 사용합니다.
DataLoader, AMP, 디바이스 설정 등을 환경에 맞게 자동 구성합니다.

Usage:
    from utils.env_config import get_env_config, get_device, get_dataloader_kwargs

    config = get_env_config()
    device = get_device()
    loader_kwargs = get_dataloader_kwargs()
"""
from __future__ import annotations

import os
from typing import Optional
from dataclasses import dataclass

import torch


@dataclass
class EnvConfig:
    """환경 설정 데이터 클래스"""
    device: str
    num_workers: int
    prefetch_factor: int
    persistent_workers: bool
    pin_memory: bool
    use_amp: bool
    amp_dtype: str
    threads_per_proc: int


def _get_env_int(key: str, default: int) -> int:
    """환경변수에서 정수값 읽기"""
    val = os.environ.get(key, None)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_env_bool(key: str, default: bool) -> bool:
    """환경변수에서 불리언값 읽기"""
    val = os.environ.get(key, None)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def _get_env_str(key: str, default: str) -> str:
    """환경변수에서 문자열 읽기"""
    return os.environ.get(key, default)


def get_env_config() -> EnvConfig:
    """환경변수에서 전체 설정 로드"""

    # Device 감지 (환경변수 우선, 없으면 자동 감지)
    env_device = _get_env_str("DEVICE", "")
    if env_device:
        device = env_device
    elif torch.cuda.is_available():
        device = "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # DataLoader 설정
    num_workers = _get_env_int("NUM_WORKERS", 0)
    prefetch_factor = _get_env_int("PREFETCH_FACTOR", 2 if num_workers > 0 else 0)
    persistent_workers = _get_env_bool("PERSISTENT_WORKERS", num_workers > 0)
    pin_memory = _get_env_bool("PIN_MEMORY", device.startswith("cuda"))

    # AMP 설정
    use_amp = _get_env_bool("USE_AMP", False)
    amp_dtype = _get_env_str("AMP_DTYPE", "fp16")

    # 스레드 설정
    threads_per_proc = _get_env_int("PYTORCH_NUM_THREADS", 1)

    return EnvConfig(
        device=device,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        threads_per_proc=threads_per_proc,
    )


def get_device() -> torch.device:
    """최적 디바이스 반환"""
    config = get_env_config()
    return torch.device(config.device)


def get_dataloader_kwargs(override_workers: Optional[int] = None) -> dict:
    """
    DataLoader용 kwargs 반환

    Args:
        override_workers: num_workers 오버라이드 (None이면 환경변수 사용)

    Returns:
        DataLoader에 전달할 kwargs dict
    """
    config = get_env_config()

    num_workers = override_workers if override_workers is not None else config.num_workers

    kwargs = {
        "num_workers": num_workers,
        "pin_memory": config.pin_memory,
    }

    # num_workers > 0일 때만 추가 옵션 적용
    if num_workers > 0:
        kwargs["persistent_workers"] = config.persistent_workers
        kwargs["prefetch_factor"] = config.prefetch_factor

    return kwargs


def get_amp_context():
    """
    AMP autocast 컨텍스트 반환

    Returns:
        torch.amp.autocast context manager (또는 nullcontext)
    """
    config = get_env_config()

    if not config.use_amp:
        from contextlib import nullcontext
        return nullcontext()

    # dtype 결정
    if config.amp_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    # device_type 결정
    if config.device.startswith("cuda"):
        device_type = "cuda"
    elif config.device == "mps":
        device_type = "cpu"  # MPS는 autocast 미지원, CPU로 fallback
        return nullcontext() if config.device == "mps" else torch.amp.autocast(device_type, dtype=dtype)
    else:
        device_type = "cpu"
        return nullcontext()  # CPU는 AMP 비활성화

    return torch.amp.autocast(device_type, dtype=dtype)


def get_grad_scaler():
    """
    GradScaler 반환 (AMP 사용 시)

    Returns:
        GradScaler 또는 None
    """
    config = get_env_config()

    if not config.use_amp or not config.device.startswith("cuda"):
        return None

    return torch.amp.GradScaler("cuda")


def print_env_summary():
    """현재 환경 설정 요약 출력"""
    config = get_env_config()

    print(f"[EnvConfig] Device: {config.device}")
    print(f"  DataLoader: workers={config.num_workers}, pin_memory={config.pin_memory}")
    print(f"  AMP: {config.use_amp} ({config.amp_dtype})")
    print(f"  Threads: {config.threads_per_proc}")


# 모듈 로드 시 PyTorch 스레드 수 설정
def _apply_thread_settings():
    """PyTorch 내부 스레드 수 적용"""
    threads = _get_env_int("PYTORCH_NUM_THREADS", 0)
    if threads > 0:
        torch.set_num_threads(threads)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(max(1, threads // 2))


_apply_thread_settings()
