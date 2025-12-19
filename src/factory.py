"""
factory.py - 컴포넌트 팩토리

학습에 필요한 컴포넌트들(손실함수, 옵티마이저, 스케줄러, 증강)을
문자열 키로 생성하는 팩토리 패턴 구현.

사용법:
    from src.factory import build_criterion, build_optimizer, build_scheduler, build_augment

    criterion = build_criterion("smoothl1")
    optimizer = build_optimizer("adamw", model.parameters(), lr=3e-4)
    scheduler = build_scheduler("cosine", optimizer, T_0=10)
    augment_fn = build_augment("cutmix", alpha=0.2)
"""
from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

from src.registry import StrategyConfig
from utils.augmentation import apply_mixup, apply_cutmix
from utils.losses import PhysMambaSSSDLoss


# ===========================================================================
# Loss Factory
# ===========================================================================

LOSS_FACTORY: Dict[str, type] = {
    "mse": nn.MSELoss,
    "smoothl1": nn.SmoothL1Loss,
    "huber": nn.HuberLoss,
    "l1": nn.L1Loss,
    "mae": nn.L1Loss,  # alias
    "physmamba_sssd": PhysMambaSSSDLoss,
}


def build_criterion(name: str, **kwargs) -> nn.Module:
    """
    손실함수 생성

    Args:
        name: 손실함수 이름 (mse, smoothl1, huber, l1)
        **kwargs: 추가 인자

    Returns:
        손실함수 인스턴스
    """
    name = name.lower()
    if name not in LOSS_FACTORY:
        available = ", ".join(sorted(LOSS_FACTORY.keys()))
        raise ValueError(f"Unknown loss: '{name}'. Available: {available}")
    return LOSS_FACTORY[name](**kwargs)


# ===========================================================================
# Optimizer Factory
# ===========================================================================

OPTIMIZER_FACTORY: Dict[str, type] = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
}


def build_optimizer(
    name: str,
    params: Iterator[nn.Parameter],
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    **kwargs,
) -> optim.Optimizer:
    """
    옵티마이저 생성

    Args:
        name: 옵티마이저 이름 (adam, adamw, sgd, rmsprop)
        params: 모델 파라미터
        lr: 학습률
        weight_decay: L2 정규화 계수
        **kwargs: 추가 인자

    Returns:
        옵티마이저 인스턴스
    """
    name = name.lower()
    if name not in OPTIMIZER_FACTORY:
        available = ", ".join(sorted(OPTIMIZER_FACTORY.keys()))
        raise ValueError(f"Unknown optimizer: '{name}'. Available: {available}")

    opt_class = OPTIMIZER_FACTORY[name]
    return opt_class(params, lr=lr, weight_decay=weight_decay, **kwargs)


# ===========================================================================
# Scheduler Factory
# ===========================================================================

SCHEDULER_FACTORY: Dict[str, type] = {
    "cosine": optim.lr_scheduler.CosineAnnealingWarmRestarts,
    "step": optim.lr_scheduler.StepLR,
    "multistep": optim.lr_scheduler.MultiStepLR,
    "exponential": optim.lr_scheduler.ExponentialLR,
    "plateau": optim.lr_scheduler.ReduceLROnPlateau,
}


def build_scheduler(
    name: Optional[str],
    optimizer: optim.Optimizer,
    T_0: int = 10,
    T_mult: int = 2,
    **kwargs,
) -> Optional[_LRScheduler]:
    """
    스케줄러 생성

    Args:
        name: 스케줄러 이름 (cosine, step, multistep, exponential, plateau, None)
        optimizer: 옵티마이저
        T_0: Cosine 스케줄러의 첫 번째 재시작 주기
        T_mult: Cosine 스케줄러의 주기 배수
        **kwargs: 추가 인자

    Returns:
        스케줄러 인스턴스 또는 None
    """
    if name is None:
        return None

    name = name.lower()
    if name not in SCHEDULER_FACTORY:
        available = ", ".join(sorted(SCHEDULER_FACTORY.keys()))
        raise ValueError(f"Unknown scheduler: '{name}'. Available: {available}")

    if name == "cosine":
        return SCHEDULER_FACTORY[name](optimizer, T_0=T_0, T_mult=T_mult, **kwargs)
    elif name == "step":
        step_size = kwargs.pop("step_size", 30)
        gamma = kwargs.pop("gamma", 0.1)
        return SCHEDULER_FACTORY[name](optimizer, step_size=step_size, gamma=gamma, **kwargs)
    elif name == "multistep":
        milestones = kwargs.pop("milestones", [30, 60, 90])
        gamma = kwargs.pop("gamma", 0.1)
        return SCHEDULER_FACTORY[name](optimizer, milestones=milestones, gamma=gamma, **kwargs)
    elif name == "exponential":
        gamma = kwargs.pop("gamma", 0.95)
        return SCHEDULER_FACTORY[name](optimizer, gamma=gamma, **kwargs)
    elif name == "plateau":
        patience = kwargs.pop("patience", 10)
        factor = kwargs.pop("factor", 0.1)
        return SCHEDULER_FACTORY[name](optimizer, patience=patience, factor=factor, **kwargs)

    return SCHEDULER_FACTORY[name](optimizer, **kwargs)


# ===========================================================================
# Augmentation Factory
# ===========================================================================


def build_augment(
    name: Optional[str],
    alpha: float = 0.2,
) -> Union[None, Callable, List[Callable]]:
    """
    증강 함수 생성

    Args:
        name: 증강 이름 (None, "cutmix", "mixup", "all")
        alpha: Beta 분포 파라미터

    Returns:
        - None: 증강 없음
        - Callable: 단일 증강 함수
        - List[Callable]: 복수 증강 함수 (aug_all 전략)
    """
    if name is None:
        return None

    name = name.lower()

    if name == "cutmix":
        return partial(apply_cutmix, alpha=alpha)
    elif name == "mixup":
        return partial(apply_mixup, alpha=alpha)
    elif name == "all":
        # 복수 증강: [mixup, cutmix]
        return [
            partial(apply_mixup, alpha=alpha),
            partial(apply_cutmix, alpha=alpha),
        ]
    else:
        raise ValueError(f"Unknown augmentation: '{name}'. Available: None, cutmix, mixup, all")


# ===========================================================================
# Strategy-based Builder
# ===========================================================================


def build_from_strategy(
    strategy: StrategyConfig,
    model: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    """
    전략 설정에서 모든 컴포넌트 빌드

    Args:
        strategy: StrategyConfig 인스턴스
        model: 모델 (파라미터 추출용)
        device: 디바이스

    Returns:
        {
            "criterion": 손실함수,
            "optimizer": 옵티마이저,
            "scheduler": 스케줄러 또는 None,
            "augment": 증강 함수 또는 None,
            "augment_fns": 복수 증강 함수 리스트 또는 None,
        }
    """
    criterion = build_criterion(strategy.loss)
    optimizer = build_optimizer(
        strategy.optimizer,
        model.parameters(),
        lr=strategy.lr,
        weight_decay=strategy.weight_decay,
    )
    scheduler = build_scheduler(
        strategy.scheduler,
        optimizer,
        T_0=strategy.scheduler_T0,
        T_mult=strategy.scheduler_Tmult,
    )
    augment_result = build_augment(strategy.augment, alpha=strategy.augment_alpha)

    # 단일 증강 vs 복수 증강 분리
    if isinstance(augment_result, list):
        augment_fn = None
        augment_fns = augment_result
    else:
        augment_fn = augment_result
        augment_fns = None

    return {
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "augment_fn": augment_fn,
        "augment_fns": augment_fns,
    }


# ===========================================================================
# Export
# ===========================================================================

__all__ = [
    # 팩토리
    "LOSS_FACTORY",
    "OPTIMIZER_FACTORY",
    "SCHEDULER_FACTORY",
    # 빌더 함수
    "build_criterion",
    "build_optimizer",
    "build_scheduler",
    "build_augment",
    "build_from_strategy",
]
