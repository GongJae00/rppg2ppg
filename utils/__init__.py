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
    env_config: 환경변수 기반 설정 (auto_profile.py 연동)
    early_stopping: Early Stopping 콜백
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
from .env_config import (
    get_env_config,
    get_device,
    get_dataloader_kwargs,
    get_amp_context,
    print_env_summary,
)
from .early_stopping import EarlyStopping
from .output_paths import build_output_paths
from .metrics_biosignal import (
    BiosignalMetrics,
    CompositeResult,
    compute_biosignal_metrics,
    compute_subject_metrics,
    composite_v2_from_metrics,
    compute_bwmd_for_chunk,
    compute_sre,
    compute_wcr,
    compute_edd,
    compute_hrv_metrics,
)
from .plot_biosignal import (
    plot_subject_signals_chunks,
    plot_multi_subject_signals_chunks,
)
from .dataset_meta import DatasetMeta, get_dataset_meta, list_datasets, canonicalize_dataset

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
    # env_config
    "get_env_config",
    "get_device",
    "get_dataloader_kwargs",
    "get_amp_context",
    "print_env_summary",
    # early_stopping
    "EarlyStopping",
    # output paths
    "build_output_paths",
    # biosignal metrics
    "BiosignalMetrics",
    "CompositeResult",
    "compute_biosignal_metrics",
    "compute_subject_metrics",
    "composite_v2_from_metrics",
    "compute_bwmd_for_chunk",
    "compute_sre",
    "compute_wcr",
    "compute_edd",
    "compute_hrv_metrics",
    # biosignal plots
    "plot_subject_signals_chunks",
    "plot_multi_subject_signals_chunks",
    # dataset meta
    "DatasetMeta",
    "get_dataset_meta",
    "list_datasets",
    "canonicalize_dataset",
]
