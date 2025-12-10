"""
메트릭 계산 유틸리티

신호 품질 평가를 위한 메트릭 함수를 제공합니다.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import pearsonr


def psnr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR) 계산

    Args:
        y_true: 실제값
        y_pred: 예측값

    Returns:
        PSNR (dB)
    """
    mse = ((y_true - y_pred) ** 2).mean()
    if mse == 0:
        return float("inf")
    max_val = np.abs(y_true).max()
    return 20 * math.log10(max_val / math.sqrt(mse))


def compute_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Pearson 상관계수 계산

    Args:
        y_true: 실제값
        y_pred: 예측값

    Returns:
        Pearson 상관계수
    """
    corr, _ = pearsonr(y_true, y_pred)
    return corr


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    return np.abs(y_true - y_pred).mean()


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Error"""
    return np.sqrt(((y_true - y_pred) ** 2).mean())


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error"""
    return (np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))).mean() * 100
