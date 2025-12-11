"""
데이터 로드 및 처리 유틸리티

데이터 로드, 정규화, Dataset/DataLoader 생성을 담당합니다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import warnings
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# PyTorch Dataloader worker 과다 경고 무시 (num_workers를 명시적으로 설정할 때)
warnings.filterwarnings(
    "ignore",
    message="This DataLoader will create .* worker processes",
    category=UserWarning,
)


class RPPGDataset(Dataset):
    """
    rPPG/PPG 시퀀스 데이터셋

    시퀀스 길이만큼 슬라이딩하며 샘플을 생성합니다.
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray, seq_len: int = 300):
        self.x = torch.from_numpy(data).float().unsqueeze(-1)  # (T, 1)
        self.y = torch.from_numpy(labels).float()  # (T,)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.x) - self.seq_len)

    def __getitem__(self, idx: int):
        return (
            self.x[idx : idx + self.seq_len],
            self.y[idx : idx + self.seq_len],
        )


def load_and_scale(
    pred_path: Path,
    label_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """
    데이터 로드 및 MinMax 정규화

    Args:
        pred_path: rPPG 예측값 경로 (.npy)
        label_path: PPG 레이블 경로 (.npy)

    Returns:
        raw_pred: 원본 예측값
        raw_label: 원본 레이블
        x_scaled: 정규화된 예측값
        y_scaled: 정규화된 레이블
        scaler_x: x 스케일러
        scaler_y: y 스케일러
    """
    raw_pred = np.load(pred_path)
    raw_label = np.load(label_path)

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    x_scaled = scaler_x.fit_transform(raw_pred.reshape(-1, 1)).flatten()
    y_scaled = scaler_y.fit_transform(raw_label.reshape(-1, 1)).flatten()

    return raw_pred, raw_label, x_scaled, y_scaled, scaler_x, scaler_y


def split_train_test(
    x: np.ndarray,
    y: np.ndarray,
    split: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    시계열 데이터 분할 (순차적 분할)

    Args:
        x: 입력 데이터
        y: 레이블 데이터
        split: 학습 데이터 비율

    Returns:
        train_x, test_x, train_y, test_y
    """
    n = len(x)
    cut = int(n * split)
    return x[:cut], x[cut:], y[:cut], y[cut:]


def make_loaders(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    seq_len: int = 300,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """
    학습/검증 DataLoader 생성

    Args:
        train_x, train_y: 학습 데이터
        test_x, test_y: 검증 데이터
        seq_len: 시퀀스 길이
        batch_size: 배치 크기
        shuffle: 학습 데이터 셔플 여부
        num_workers: 데이터 로더 워커 수
        pin_memory: CUDA 핀 메모리 사용
        persistent_workers: 워커 프로세스 유지 (num_workers > 0일 때만)
        prefetch_factor: 프리페치 배치 수 (num_workers > 0일 때만)

    Returns:
        train_loader, test_loader
    """
    train_ds = RPPGDataset(train_x, train_y, seq_len)
    test_ds = RPPGDataset(test_x, test_y, seq_len)

    # num_workers > 0일 때만 persistent_workers, prefetch_factor 적용
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_ds,
        shuffle=shuffle,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, test_loader
