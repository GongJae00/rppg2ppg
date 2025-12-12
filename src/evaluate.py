#!/usr/bin/env python3
"""
evaluate.py - 평가 실행 스크립트

Evaluator를 사용해 데이터셋/예측 파일들을 일괄 평가합니다.

기본 동작:
  1) data/{DATASET}_prediction/*.npy 로부터 baseline/legacy 예측을 로드
  2) (옵션) results/**/predictions/*.npy 로부터 모델 예측을 추가 로드
  3) basic + biosignal 메트릭 계산 후 콘솔 출력

Usage:
    python -m src.evaluate --dataset PURE
    python -m src.evaluate --dataset PURE --results-root results
    python -m src.evaluate --dataset UBFC --pred-dir data/UBFC_prediction --per-subject
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np

from .evaluator import Evaluator
from utils.dataset_meta import get_dataset_meta


def _scan_dir(dir_path: Path) -> Dict[str, np.ndarray]:
    preds: Dict[str, np.ndarray] = {}
    if not dir_path.exists():
        return preds
    for fp in sorted(dir_path.glob("*.npy")):
        preds[fp.stem] = np.load(fp)
    return preds


def _scan_results(results_root: Path) -> Dict[str, np.ndarray]:
    preds: Dict[str, np.ndarray] = {}
    if not results_root.exists():
        return preds
    for fp in sorted(results_root.glob("*/predictions/*.npy")):
        preds[fp.stem] = np.load(fp)
    return preds


def main():
    parser = argparse.ArgumentParser(description="Biosignal evaluation runner")
    parser.add_argument("--dataset", required=True, help="dataset key (PURE/UBFC/cohface)")
    parser.add_argument("--pred-dir", type=Path, default=None, help="baseline prediction dir override")
    parser.add_argument("--results-root", type=Path, default=None, help="results root to include model predictions")
    parser.add_argument("--per-subject", action="store_true", help="compute subject-level metrics")
    parser.add_argument("--eval-seconds", type=int, default=10, help="chunk length (sec) for BWMD etc.")
    parser.add_argument("--out-json", type=Path, default=None, help="optional json output path")
    args = parser.parse_args()

    meta = get_dataset_meta(args.dataset)
    pred_dir = args.pred_dir or meta.prediction_dir

    preds_dict = _scan_dir(pred_dir)
    if args.results_root is not None:
        preds_dict.update(_scan_results(args.results_root))

    if not preds_dict:
        raise SystemExit(f"No predictions found in {pred_dir} (and results_root={args.results_root})")

    evaluator = Evaluator(dataset=args.dataset, eval_seconds=args.eval_seconds)
    records = evaluator.evaluate_many(preds_dict, per_subject=args.per_subject)

    print(f"\n=== Evaluation: {args.dataset} ({len(records)} preds) ===")
    for name, rec in records.items():
        b = rec.basic
        m = rec.biosignal_mean
        print(
            f"\n>> {name}\n"
            f"   Pearson : {b['Pearson']:.4f} | PSNR(dB): {b['PSNR(dB)']:.2f} | "
            f"MAE: {b['MAE']:.4f} | RMSE: {b['RMSE']:.4f} | MAPE(%): {b['MAPE(%)']:.2f}\n"
            f"   BWMD: {m['BWMD']:.5f} | SRE: {m['SRE']:.5f} | WCR: {m['WCR']:.5f} | "
            f"EDD: {m['EDD']:.5f} | Composite_v2: {rec.composite_v2.score:.2f}"
        )

    if args.out_json is not None:
        payload = {k: asdict(v) for k, v in records.items()}
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"\nSaved json: {args.out_json}")


if __name__ == "__main__":
    main()

