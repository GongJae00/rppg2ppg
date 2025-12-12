"""
train.py - 학습 루프 유틸리티

Trainer(전략 독립 엔진) 위에서 동작하는 공통 학습 루프를 제공합니다.
strategies/의 각 전략 스크립트가 중복 없이 재사용할 수 있도록 설계했습니다.

역할 분리:
- trainer.py (Low-level): 배치 처리, AMP, 그래디언트 클리핑
- train.py (Mid-level): 에폭 루프, Early Stopping, 로깅
- strategies/ (High-level): 전략 설정, 실행 진입점
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol, Sequence, Union

from torch.utils.data import DataLoader

from .trainer import Trainer
from utils.early_stopping import EarlyStopping


# ===========================================================================
# TrainResult 데이터클래스
# ===========================================================================


@dataclass
class TrainResult:
    """학습 결과를 담는 데이터클래스"""

    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    best_epoch: int = 0
    last_epoch: int = 0
    stopped_early: bool = False

    @property
    def best_train_loss(self) -> float:
        """최소 학습 손실"""
        return min(self.train_losses) if self.train_losses else float("inf")

    @property
    def best_val_loss(self) -> float:
        """최소 검증 손실"""
        return min(self.val_losses) if self.val_losses else float("inf")

    @property
    def final_train_loss(self) -> float:
        """최종 학습 손실"""
        return self.train_losses[-1] if self.train_losses else float("inf")

    @property
    def final_val_loss(self) -> float:
        """최종 검증 손실"""
        return self.val_losses[-1] if self.val_losses else float("inf")


# ===========================================================================
# Callback Protocol
# ===========================================================================


class TrainCallback(Protocol):
    """학습 콜백 프로토콜"""

    def on_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        trainer: Trainer,
    ) -> bool:
        """
        에폭 종료 시 호출.

        Returns:
            True이면 학습 중단
        """
        ...


# ===========================================================================
# fit() 함수
# ===========================================================================


def fit(
    trainer: Trainer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    augment_fn: Optional[Callable] = None,
    augment_fns: Optional[Sequence[Callable]] = None,
    early_stopping: Optional[EarlyStopping] = None,
    callbacks: Optional[List[TrainCallback]] = None,
    verbose: bool = True,
) -> TrainResult:
    """
    공통 학습 루프 실행.

    모든 전략(strategy_A~E)이 이 함수를 재사용하여 코드 중복을 제거합니다.

    Args:
        trainer: Trainer 인스턴스 (모델, 옵티마이저, 손실함수 포함)
        train_loader: 학습 DataLoader
        val_loader: 검증 DataLoader
        epochs: 최대 에폭 수
        augment_fn: 단일 증강 함수 (train_epoch 사용)
        augment_fns: 복수 증강 함수 (train_epoch_multi_aug 사용)
        early_stopping: EarlyStopping 콜백 (선택)
        callbacks: 추가 콜백 리스트 (선택)
        verbose: 진행 상황 출력 여부

    Returns:
        TrainResult: 학습 결과 (손실 리스트, 에폭 정보 등)

    Example:
        >>> trainer = Trainer(model, criterion, optimizer, device)
        >>> result = fit(trainer, train_loader, val_loader, epochs=100)
        >>> print(f"Best val loss: {result.best_val_loss:.4f} at epoch {result.best_epoch}")
    """
    result = TrainResult()
    callbacks = callbacks or []
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # 학습
        if augment_fns is not None:
            train_loss = trainer.train_epoch_multi_aug(train_loader, list(augment_fns))
        else:
            train_loss = trainer.train_epoch(train_loader, augment_fn=augment_fn)

        # 검증
        val_loss = trainer.validate(val_loader)

        # 결과 기록
        result.train_losses.append(train_loss)
        result.val_losses.append(val_loss)
        result.last_epoch = epoch

        # 최고 성능 갱신
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            result.best_epoch = epoch

        # 출력
        if verbose:
            print(f"  Epoch {epoch:3d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        # Early Stopping 체크
        if early_stopping is not None and early_stopping(val_loss, trainer.model, epoch):
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            result.stopped_early = True
            break

        # 커스텀 콜백 실행
        for callback in callbacks:
            if callback.on_epoch_end(epoch, train_loss, val_loss, trainer):
                result.stopped_early = True
                break

        if result.stopped_early:
            break

    return result


# ===========================================================================
# 하위 호환성을 위한 래퍼 (기존 반환 형식)
# ===========================================================================


def fit_legacy(
    trainer: Trainer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    augment_fn: Optional[Callable] = None,
    augment_fns: Optional[Sequence[Callable]] = None,
    early_stopping: Optional[EarlyStopping] = None,
    verbose: bool = True,
) -> tuple:
    """
    하위 호환성을 위한 fit() 래퍼.

    Returns:
        (train_losses, val_losses, last_epoch) 튜플
    """
    result = fit(
        trainer=trainer,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        augment_fn=augment_fn,
        augment_fns=augment_fns,
        early_stopping=early_stopping,
        verbose=verbose,
    )
    return result.train_losses, result.val_losses, result.last_epoch
