"""
strategy.py - 전략 정의 및 레지스트리

모든 학습 전략을 데코레이터 기반으로 자동 등록합니다.

새 전략 추가 방법:
    @register_strategy(name="customstrategy", code="F")
    class CustomStrategy(StrategyConfig):
        augment = "cutmix"
        loss = "smoothl1"
        optimizer = "adamw"
        ...

사용법:
    from src.strategy import get_strategy, list_strategies

    strategy = get_strategy("baseline")
    print(strategy.loss)  # "mse"
"""
from __future__ import annotations

from src.registry import (
    StrategyConfig,
    register_strategy,
    get_strategy,
    list_strategies,
    STRATEGY_REGISTRY,
)


# ===========================================================================
# 기본 전략 정의 (A ~ E)
# ===========================================================================


@register_strategy(name="baseline", code="A")
class BaselineStrategy(StrategyConfig):
    """
    Strategy A: 기본 전략

    - 증강 없음
    - MSE 손실
    - Adam 옵티마이저
    - 스케줄러 없음
    """

    augment = None
    augment_alpha = 0.2
    loss = "mse"
    optimizer = "adam"
    lr = 1e-4
    weight_decay = 0.0
    scheduler = None
    clip_grad = None
    epochs = 100
    patience = 10
    overlap = 0.0
    fill_tail = True


@register_strategy(name="improved", code="B")
class ImprovedStrategy(StrategyConfig):
    """
    Strategy B: 개선된 전략

    - 증강 없음
    - SmoothL1 손실
    - AdamW 옵티마이저
    - Cosine 스케줄러
    - 그래디언트 클리핑
    """

    augment = None
    augment_alpha = 0.2
    loss = "smoothl1"
    optimizer = "adamw"
    lr = 3e-4
    weight_decay = 1e-5
    scheduler = "cosine"
    scheduler_T0 = 10
    scheduler_Tmult = 2
    clip_grad = 1.0
    epochs = 65
    patience = 10
    overlap = 0.5
    fill_tail = True


@register_strategy(name="cutmix", code="C")
class CutMixStrategy(StrategyConfig):
    """
    Strategy C: CutMix 증강 전략

    - CutMix 증강
    - SmoothL1 손실
    - AdamW 옵티마이저
    - Cosine 스케줄러
    """

    augment = "cutmix"
    augment_alpha = 0.2
    loss = "smoothl1"
    optimizer = "adamw"
    lr = 3e-4
    weight_decay = 1e-5
    scheduler = "cosine"
    scheduler_T0 = 10
    scheduler_Tmult = 2
    clip_grad = 1.0
    epochs = 65
    patience = 8
    overlap = 0.5
    fill_tail = True


@register_strategy(name="mixup", code="D")
class MixUpStrategy(StrategyConfig):
    """
    Strategy D: MixUp 증강 전략

    - MixUp 증강
    - SmoothL1 손실
    - AdamW 옵티마이저
    - Cosine 스케줄러
    """

    augment = "mixup"
    augment_alpha = 0.2
    loss = "smoothl1"
    optimizer = "adamw"
    lr = 3e-4
    weight_decay = 1e-5
    scheduler = "cosine"
    scheduler_T0 = 10
    scheduler_Tmult = 2
    clip_grad = 1.0
    epochs = 65
    patience = 8
    overlap = 0.5
    fill_tail = True


@register_strategy(name="aug_all", code="E")
class AugAllStrategy(StrategyConfig):
    """
    Strategy E: 복합 증강 전략

    - 원본 + MixUp + CutMix (3배 배치)
    - SmoothL1 손실
    - AdamW 옵티마이저
    - Cosine 스케줄러
    """

    augment = "all"
    augment_alpha = 0.2
    loss = "smoothl1"
    optimizer = "adamw"
    lr = 3e-4
    weight_decay = 1e-5
    scheduler = "cosine"
    scheduler_T0 = 10
    scheduler_Tmult = 2
    clip_grad = 1.0
    epochs = 65
    patience = 12
    overlap = 0.5
    fill_tail = True


# ===========================================================================
# 전략 유틸리티
# ===========================================================================


def get_strategy_by_code(code: str) -> StrategyConfig:
    """전략 코드(A, B, C, ...)로 전략 찾기"""
    code = code.upper()
    for name, strategy in STRATEGY_REGISTRY.items():
        if strategy.code == code:
            return strategy
    raise ValueError(f"Unknown strategy code: '{code}'. Available: A, B, C, D, E")


def list_strategy_codes() -> list:
    """등록된 전략 코드 목록"""
    return [(s.code, s.name) for s in STRATEGY_REGISTRY.values()]


# ===========================================================================
# Export
# ===========================================================================


__all__ = [
    # 전략 클래스
    "StrategyConfig",
    "BaselineStrategy",
    "ImprovedStrategy",
    "CutMixStrategy",
    "MixUpStrategy",
    "AugAllStrategy",
    # 함수
    "register_strategy",
    "get_strategy",
    "get_strategy_by_code",
    "list_strategies",
    "list_strategy_codes",
    # 레지스트리
    "STRATEGY_REGISTRY",
]
