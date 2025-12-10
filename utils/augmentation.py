"""
데이터 증강 유틸리티

MixUp, CutMix 등 배치 단위 증강 함수를 제공합니다.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
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
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)

    perm = torch.randperm(batch_size, device=x.device)
    x2, y2 = x[perm], y[perm]

    x_mixed = lam * x + (1 - lam) * x2
    y_mixed = lam * y + (1 - lam) * y2

    return x_mixed, y_mixed


def apply_cutmix(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CutMix 증강

    시퀀스의 일부 구간을 다른 샘플로 교체합니다.

    Args:
        x: 입력 텐서 (B, L, F)
        y: 레이블 텐서 (B, L)
        alpha: 베타 분포 파라미터 (교체 비율 결정)

    Returns:
        교체된 x, y
    """
    batch_size, seq_len, _ = x.size()
    lam = np.random.beta(alpha, alpha)

    perm = torch.randperm(batch_size, device=x.device)
    x2, y2 = x[perm], y[perm]

    # 교체할 구간 계산
    cut_len = int(seq_len * lam)
    start = np.random.randint(0, seq_len - cut_len + 1) if cut_len < seq_len else 0
    end = start + cut_len

    # 구간 교체
    x_new = x.clone()
    y_new = y.clone()
    x_new[:, start:end, :] = x2[:, start:end, :]
    y_new[:, start:end] = y2[:, start:end]

    return x_new, y_new
