#!/usr/bin/env python3
"""
Strategy D: MixUp 전략

MixUp 데이터 증강을 적용한 학습 전략입니다.
- 증강: MixUp (샘플 간 선형 보간)
- 손실함수: SmoothL1 (Huber Loss)
- 옵티마이저: AdamW
- 스케줄러: Cosine Annealing Warm Restarts
- Early Stopping: patience=8

Usage:
    python strategies/strategy_D_mixup.py --config configs/mamba.toon
    python strategies/strategy_D_mixup.py --config configs/lstm.toon --model lstm
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runner import run_strategy


def main():
    parser = argparse.ArgumentParser(description="Run MixUp strategy (Strategy D)")
    parser.add_argument("--config", type=Path, default=Path("configs/mamba.toon"))
    parser.add_argument("--model", type=str, default=None, help="모델 키 (config 오버라이드)")
    args = parser.parse_args()

    run_strategy(
        strategy_name="mixup",
        model_key=args.model,
        config_path=args.config,
        verbose=True,
    )


if __name__ == "__main__":
    main()
