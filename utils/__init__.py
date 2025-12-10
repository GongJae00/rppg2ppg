"""
utils/ - rPPG to PPG 유틸리티 모듈

순수 유틸리티 함수들을 기능별로 분리하여 제공합니다.

Modules:
    data: 데이터 로드, Dataset, DataLoader
    augmentation: 데이터 증강 (MixUp, CutMix)
    signal: 신호 처리 (필터링, 전처리)
    metrics: 평가 지표 (PSNR, Pearson, MAE, RMSE)
    plot: 시각화 (학습 곡선, 예측 결과)
    toon: TOON 설정 파일 파서
"""
from .data import (
    RPPGDataset,
    load_and_scale,
    split_train_test,
    make_loaders,
)
from .augmentation import apply_mixup, apply_cutmix
from .signal import lowpass_filter, bandpass_filter, detrend
from .metrics import psnr, compute_pearson, mae, rmse, mape
from .plot import plot_losses, plot_prediction
from .toon import load_toon

__all__ = [
    # data
    "RPPGDataset",
    "load_and_scale",
    "split_train_test",
    "make_loaders",
    # augmentation
    "apply_mixup",
    "apply_cutmix",
    # signal
    "lowpass_filter",
    "bandpass_filter",
    "detrend",
    # metrics
    "psnr",
    "compute_pearson",
    "mae",
    "rmse",
    "mape",
    # plot
    "plot_losses",
    "plot_prediction",
    # toon
    "load_toon",
]
