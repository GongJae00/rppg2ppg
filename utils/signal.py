"""
신호 처리 유틸리티

필터링, 전처리 등 신호 처리 관련 함수를 제공합니다.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def lowpass_filter(
    signal: np.ndarray,
    fs: float = 30.0,
    cutoff: float = 3.0,
    order: int = 4,
) -> np.ndarray:
    """
    Butterworth 저역통과 필터

    Args:
        signal: 입력 신호
        fs: 샘플링 레이트 (Hz)
        cutoff: 차단 주파수 (Hz)
        order: 필터 차수

    Returns:
        필터링된 신호
    """
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    b, a = butter(order, normalized_cutoff, btype="low")
    return filtfilt(b, a, signal)


def bandpass_filter(
    signal: np.ndarray,
    fs: float = 30.0,
    low: float = 0.75,
    high: float = 3.0,
    order: int = 4,
) -> np.ndarray:
    """
    Butterworth 대역통과 필터

    Args:
        signal: 입력 신호
        fs: 샘플링 레이트 (Hz)
        low: 하한 주파수 (Hz)
        high: 상한 주파수 (Hz)
        order: 필터 차수

    Returns:
        필터링된 신호
    """
    nyquist = 0.5 * fs
    low_norm = low / nyquist
    high_norm = high / nyquist
    b, a = butter(order, [low_norm, high_norm], btype="band")
    return filtfilt(b, a, signal)


def detrend(signal: np.ndarray, lambda_val: float = 100.0) -> np.ndarray:
    """
    신호 트렌드 제거 (Tarvainen et al., 2002)

    Args:
        signal: 입력 신호
        lambda_val: 정규화 파라미터 (클수록 더 스무스)

    Returns:
        트렌드가 제거된 신호
    """
    T = len(signal)
    I = np.eye(T)
    D2 = np.zeros((T - 2, T))
    for i in range(T - 2):
        D2[i, i] = 1
        D2[i, i + 1] = -2
        D2[i, i + 2] = 1

    trend = np.linalg.solve(I + lambda_val ** 2 * D2.T @ D2, signal)
    return signal - trend
