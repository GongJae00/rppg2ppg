"""
데이터 증강 유틸리티

MixUp, CutMix 등 배치 단위 증강 함수를 제공합니다.
"""
from __future__ import annotations

from typing import Tuple

import torch


def apply_mixup(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MixUp 증강

    두 샘플을 베타 분포 기반 가중치로 선형 결합합니다.

    Args:
        x: 입력 텐서 (B, L, F)
        y: 레이블 텐서 (B, L)
        alpha: 베타 분포 파라미터 (작을수록 원본에 가까움)

    Returns:
        혼합된 x, y
    """
    # torch 기반 Beta 샘플링 (bfloat16 파라미터는 Beta에서 미지원일 수 있어 float32로 샘플 후 캐스트)
    alpha_t = torch.tensor(alpha, device=x.device, dtype=torch.float32)
    lam = torch.distributions.Beta(alpha_t, alpha_t).sample().to(dtype=x.dtype)
    batch_size = x.size(0)

    perm = torch.randperm(batch_size, device=x.device)
    x2, y2 = x[perm], y[perm]

    x_mixed = lam * x + (1.0 - lam) * x2
    y_mixed = lam * y + (1.0 - lam) * y2

    return x_mixed, y_mixed


def apply_cutmix(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CutMix 증강 (메모리 최적화)

    시퀀스의 일부 구간을 다른 샘플로 교체합니다.
    clone() 대신 empty_like() + 직접 할당으로 메모리 효율화.

    Args:
        x: 입력 텐서 (B, L, F)
        y: 레이블 텐서 (B, L)
        alpha: 베타 분포 파라미터 (교체 비율 결정)

    Returns:
        교체된 x, y
    """
    batch_size, seq_len, _ = x.size()
    # cut_len 계산은 정수 연산이므로 CPU에서 샘플링
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())

    perm = torch.randperm(batch_size, device=x.device)

    # 교체할 구간 계산
    cut_len = int(seq_len * lam)
    if cut_len < seq_len:
        start = int(torch.randint(0, seq_len - cut_len + 1, (1,)).item())
    else:
        start = 0
    end = start + cut_len

    # 메모리 최적화: empty_like + 직접 할당 (clone 대신)
    x_new = torch.empty_like(x)
    y_new = torch.empty_like(y)

    # 원본 구간 복사
    if start > 0:
        x_new[:, :start, :] = x[:, :start, :]
        y_new[:, :start] = y[:, :start]

    # 교체 구간 (permuted sample에서)
    x_new[:, start:end, :] = x[perm, start:end, :]
    y_new[:, start:end] = y[perm, start:end]

    # 나머지 구간 복사
    if end < seq_len:
        x_new[:, end:, :] = x[:, end:, :]
        y_new[:, end:] = y[:, end:]

    return x_new, y_new
