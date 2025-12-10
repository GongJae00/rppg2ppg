"""
Predictor - 추론 및 후처리 로직

학습된 모델로 전체 시퀀스에 대한 예측을 수행합니다.
슬라이딩 윈도우 방식으로 중첩 예측 후 평균화합니다.

Example:
    from src import Predictor

    predictor = Predictor(model, device)
    preds = predictor.predict(x_all, seq_len=300, overlap=0.5)
    preds_final = predictor.postprocess(preds, scaler_y, fs=30, cutoff=3.0)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from utils.signal import lowpass_filter


class Predictor:
    """
    추론 엔진

    Attributes:
        model: 학습된 모델
        device: 추론 디바이스
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def predict(
        self,
        x_all: np.ndarray,
        seq_len: int = 300,
        overlap: float = 0.5,
        fill_tail: bool = True,
    ) -> np.ndarray:
        """
        슬라이딩 윈도우 예측

        Args:
            x_all: 전체 입력 시퀀스 (정규화된 상태)
            seq_len: 윈도우 크기
            overlap: 중첩 비율 (0.0 ~ 1.0)
            fill_tail: 마지막 부분 채우기 여부

        Returns:
            예측값 (정규화된 상태)
        """
        step = max(1, int(seq_len * (1 - overlap)))
        L = len(x_all)

        preds_accum = np.zeros(L, dtype=np.float32)
        weight_accum = np.zeros(L, dtype=np.float32)

        self.model.eval()
        with torch.no_grad():
            for i in range(0, L - seq_len + 1, step):
                chunk = torch.tensor(x_all[i : i + seq_len]).float()
                chunk = chunk.unsqueeze(0).unsqueeze(-1).to(self.device)

                out = self.model(chunk).cpu().numpy().flatten()
                preds_accum[i : i + seq_len] += out
                weight_accum[i : i + seq_len] += 1

            # 마지막 부분 처리
            if fill_tail:
                last_start = L - seq_len
                if last_start >= 0 and weight_accum[last_start:].min() == 0:
                    chunk = torch.tensor(x_all[last_start:]).float()
                    chunk = chunk.unsqueeze(0).unsqueeze(-1).to(self.device)
                    out = self.model(chunk).cpu().numpy().flatten()
                    preds_accum[last_start:] += out
                    weight_accum[last_start:] += 1

        return preds_accum / np.maximum(weight_accum, 1e-8)

    def postprocess(
        self,
        preds_norm: np.ndarray,
        scaler_y: MinMaxScaler,
        raw_label_len: int,
        fs: float = 30.0,
        cutoff: float = 3.0,
    ) -> np.ndarray:
        """
        예측값 후처리 (역정규화 + 저역통과 필터)

        Args:
            preds_norm: 정규화된 예측값
            scaler_y: y값 스케일러
            raw_label_len: 원본 레이블 길이
            fs: 샘플링 레이트
            cutoff: 저역통과 필터 차단 주파수

        Returns:
            후처리된 예측값
        """
        # 역정규화
        preds_scaled = scaler_y.inverse_transform(
            preds_norm[:raw_label_len].reshape(-1, 1)
        ).flatten()

        # 저역통과 필터
        preds_filtered = lowpass_filter(preds_scaled, fs=fs, cutoff=cutoff)

        return preds_filtered
