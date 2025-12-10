"""
시각화 유틸리티

학습 곡선, 예측 결과 시각화 등을 담당합니다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def plot_losses(
    path: Path,
    train_losses: Iterable[float],
    val_losses: Iterable[float],
    ylabel: str = "Loss",
):
    """
    학습/검증 손실 곡선 저장

    Args:
        path: 저장 경로
        train_losses: 학습 손실 리스트
        val_losses: 검증 손실 리스트
        ylabel: y축 라벨
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train", linewidth=2)
    plt.plot(val_losses, label="Validation", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_prediction(
    path: Path,
    y_true: Iterable[float],
    y_pred: Iterable[float],
    title: str = "Prediction vs Ground Truth",
):
    """
    예측 결과 시각화

    Args:
        path: 저장 경로
        y_true: 실제값
        y_pred: 예측값
        title: 그래프 제목
    """
    plt.figure(figsize=(12, 4))
    plt.plot(y_true, label="Ground Truth", alpha=0.7)
    plt.plot(y_pred, label="Prediction", alpha=0.7)
    plt.xlabel("Sample")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
