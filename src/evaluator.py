"""
Evaluator - 생체신호 기반 평가 엔진

main_eval(origin)의 평가 기능을 현행 rppg2ppg 구조(data/, results/)에 맞게
재구성한 모듈입니다.

기능:
  - 라벨/예측 신호 로드 및 길이 정렬
  - 기본 지표(Pearson, PSNR, MAE, RMSE, MAPE) 계산
  - 고급 biosignal 지표(BWMD, SRE, WCR, EDD, HRV, Composite v2) 계산
  - subject 단위 평가 지원(subject_counts 필요)

Example:
    from src.evaluator import Evaluator

    ev = Evaluator(dataset="PURE", eval_seconds=10)
    preds = {"mamba_A_baseline": np.load(...), "lstm_A_baseline": np.load(...)}
    results = ev.evaluate_many(preds, per_subject=True)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from utils.metrics import psnr, compute_pearson, mae, rmse, mape
from utils.metrics_biosignal import (
    BiosignalMetrics,
    CompositeResult,
    compute_biosignal_metrics,
    compute_subject_metrics,
    composite_v2_from_metrics,
)
from utils.dataset_meta import DatasetMeta, get_dataset_meta


@dataclass
class EvaluationRecord:
    """단일 예측에 대한 평가 결과."""

    name: str
    basic: Dict[str, float]
    biosignal_mean: Dict[str, float]
    composite_v2: CompositeResult
    biosignal_per_subject: Optional[List[BiosignalMetrics]] = None


class Evaluator:
    """
    데이터셋 단위 평가 클래스.
    """

    def __init__(
        self,
        dataset: str,
        label_path: Optional[Path] = None,
        subject_counts: Optional[Sequence[int]] = None,
        fs: Optional[float] = None,
        eval_seconds: int = 10,
    ):
        self.meta: DatasetMeta = get_dataset_meta(dataset)
        self.label_path = label_path or self.meta.label_path
        self.subject_counts = list(subject_counts) if subject_counts is not None else self.meta.subject_counts
        self.fs = float(fs) if fs is not None else self.meta.fs
        self.eval_seconds = eval_seconds

        self.labels: np.ndarray = self._load_npy(self.label_path)

    @staticmethod
    def _load_npy(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"npy not found: {path}")
        return np.load(path)

    def _align(self, preds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """label과 prediction 길이를 맞춰 반환."""
        p = np.asarray(preds).ravel()
        l = np.asarray(self.labels).ravel()
        L = min(p.size, l.size)
        return l[:L], p[:L]

    def evaluate_basic(self, preds: np.ndarray) -> Dict[str, float]:
        """
        기본 지표 계산.
        """
        y_true, y_pred = self._align(preds)
        return {
            "Pearson": float(compute_pearson(y_true, y_pred)),
            "PSNR(dB)": float(psnr(y_true, y_pred)),
            "MAE": float(mae(y_true, y_pred)),
            "RMSE": float(rmse(y_true, y_pred)),
            "MAPE(%)": float(mape(y_true, y_pred)),
        }

    def evaluate_one(
        self,
        preds: np.ndarray,
        name: str = "prediction",
        per_subject: bool = False,
        biosignal_kwargs: Optional[Dict] = None,
    ) -> EvaluationRecord:
        """
        단일 prediction 평가.
        """
        y_true, y_pred = self._align(preds)
        basic = self.evaluate_basic(y_pred)
        biosignal_kwargs = biosignal_kwargs or {}

        if per_subject and self.subject_counts:
            per_subj = compute_subject_metrics(
                preds=y_pred,
                labels=y_true,
                subject_counts=self.subject_counts,
                fs=self.fs,
                eval_seconds=self.eval_seconds,
                **biosignal_kwargs,
            )
            mean_bwmd = float(np.nanmean([m.bwmd for m in per_subj])) if per_subj else 0.0
            mean_sre = float(np.nanmean([m.sre for m in per_subj])) if per_subj else 0.0
            mean_wcr = float(np.nanmean([m.wcr for m in per_subj])) if per_subj else 0.0
            mean_edd = float(np.nanmean([m.edd for m in per_subj])) if per_subj else 0.0
        else:
            per_subj = None
            m = compute_biosignal_metrics(
                y_true=y_true,
                y_pred=y_pred,
                fs=self.fs,
                eval_seconds=self.eval_seconds,
                **biosignal_kwargs,
            )
            mean_bwmd, mean_sre, mean_wcr, mean_edd = m.bwmd, m.sre, m.wcr, m.edd

        # composite 계산을 위해 임시 reference를 자신만 사용(단일 평가 시 의미 없음)
        comp = composite_v2_from_metrics(
            mean_bwmd,
            mean_sre,
            mean_wcr,
            mean_edd,
            ref_bwmd=[mean_bwmd],
            ref_sre=[mean_sre],
            ref_wcr=[mean_wcr],
            ref_edd=[mean_edd],
        )

        return EvaluationRecord(
            name=name,
            basic=basic,
            biosignal_mean={
                "BWMD": mean_bwmd,
                "SRE": mean_sre,
                "WCR": mean_wcr,
                "EDD": mean_edd,
            },
            composite_v2=comp,
            biosignal_per_subject=per_subj,
        )

    def evaluate_many(
        self,
        preds_dict: Dict[str, np.ndarray],
        per_subject: bool = True,
        biosignal_kwargs: Optional[Dict] = None,
    ) -> Dict[str, EvaluationRecord]:
        """
        여러 prediction을 한 번에 평가하고 Composite v2까지 산출.
        """
        biosignal_kwargs = biosignal_kwargs or {}

        records: Dict[str, EvaluationRecord] = {}
        bwmd_means: List[float] = []
        sre_means: List[float] = []
        wcr_means: List[float] = []
        edd_means: List[float] = []

        # 1) 개별 평가 + mean 지표 수집
        for name, preds in preds_dict.items():
            rec = self.evaluate_one(
                preds=preds,
                name=name,
                per_subject=per_subject,
                biosignal_kwargs=biosignal_kwargs,
            )
            records[name] = rec
            bwmd_means.append(rec.biosignal_mean["BWMD"])
            sre_means.append(rec.biosignal_mean["SRE"])
            wcr_means.append(rec.biosignal_mean["WCR"])
            edd_means.append(rec.biosignal_mean["EDD"])

        # 2) reference 분포 기반 composite 재계산
        for name, rec in records.items():
            comp = composite_v2_from_metrics(
                rec.biosignal_mean["BWMD"],
                rec.biosignal_mean["SRE"],
                rec.biosignal_mean["WCR"],
                rec.biosignal_mean["EDD"],
                ref_bwmd=bwmd_means,
                ref_sre=sre_means,
                ref_wcr=wcr_means,
                ref_edd=edd_means,
            )
            rec.composite_v2 = comp

        return records

