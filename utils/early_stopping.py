"""
Early Stopping 유틸리티

학습 중 검증 손실이 개선되지 않으면 조기 종료합니다.
모델의 최적 상태를 자동으로 저장하고 복원할 수 있습니다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn


class EarlyStopping:
    """
    Early Stopping 콜백

    검증 손실이 patience 에폭 동안 개선되지 않으면 학습을 조기 종료합니다.

    Args:
        patience: 개선 없이 대기할 최대 에폭 수
        min_delta: 개선으로 인정할 최소 변화량
        mode: 'min' (손실 최소화) 또는 'max' (정확도 최대화)
        save_path: 최적 모델 저장 경로 (None이면 저장 안 함)
        verbose: 상태 출력 여부

    Example:
        early_stopping = EarlyStopping(patience=10, save_path="best.pth")

        for epoch in range(epochs):
            train_loss = train_one_epoch(...)
            val_loss = validate(...)

            if early_stopping(val_loss, model):
                print(f"Early stopping at epoch {epoch}")
                break

        # 최적 모델 복원
        early_stopping.load_best(model)
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        save_path: Optional[Path] = None,
        verbose: bool = False,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.save_path = Path(save_path) if save_path else None
        self.verbose = verbose

        self.counter = 0
        self.best_score: Optional[float] = None
        self.best_state: Optional[dict] = None
        self.early_stop = False
        self.stopped_epoch = 0

        # mode에 따른 비교 함수 설정
        if mode == "min":
            self.is_better = lambda new, best: new < best - min_delta
        elif mode == "max":
            self.is_better = lambda new, best: new > best + min_delta
        else:
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

    def __call__(self, score: float, model: nn.Module, epoch: int = 0) -> bool:
        """
        검증 점수를 확인하고 early stopping 여부 결정

        Args:
            score: 현재 에폭의 검증 점수 (손실 또는 정확도)
            model: 모델 인스턴스 (최적 상태 저장용)
            epoch: 현재 에폭 번호 (로깅용)

        Returns:
            True면 학습 중단, False면 계속
        """
        if self.best_score is None:
            # 첫 에폭
            self.best_score = score
            self._save_checkpoint(model)
        elif self.is_better(score, self.best_score):
            # 개선됨
            if self.verbose:
                improvement = self.best_score - score if self.mode == "min" else score - self.best_score
                print(f"  [EarlyStopping] Improved by {improvement:.6f}")
            self.best_score = score
            self.counter = 0
            self._save_checkpoint(model)
        else:
            # 개선 안 됨
            self.counter += 1
            if self.verbose:
                print(f"  [EarlyStopping] No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True
                self.stopped_epoch = epoch
                return True

        return False

    def _save_checkpoint(self, model: nn.Module):
        """최적 모델 상태 저장"""
        self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if self.save_path:
            torch.save(self.best_state, self.save_path)

    def load_best(self, model: nn.Module) -> bool:
        """
        저장된 최적 모델 상태 복원

        Args:
            model: 상태를 복원할 모델

        Returns:
            성공 여부
        """
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            return True
        elif self.save_path and self.save_path.exists():
            model.load_state_dict(torch.load(self.save_path, weights_only=True))
            return True
        return False

    def reset(self):
        """상태 초기화 (재사용 시)"""
        self.counter = 0
        self.best_score = None
        self.best_state = None
        self.early_stop = False
        self.stopped_epoch = 0

    @property
    def should_stop(self) -> bool:
        """학습 중단 여부"""
        return self.early_stop

    def state_dict(self) -> dict:
        """체크포인트 저장용 상태"""
        return {
            "counter": self.counter,
            "best_score": self.best_score,
            "early_stop": self.early_stop,
            "stopped_epoch": self.stopped_epoch,
        }

    def load_state_dict(self, state: dict):
        """체크포인트에서 상태 복원"""
        self.counter = state.get("counter", 0)
        self.best_score = state.get("best_score")
        self.early_stop = state.get("early_stop", False)
        self.stopped_epoch = state.get("stopped_epoch", 0)
