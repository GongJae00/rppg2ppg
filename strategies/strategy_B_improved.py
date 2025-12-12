#!/usr/bin/env python3
"""
Strategy B: Improved 전략

최적화된 학습 전략입니다.
- 증강: 없음
- 손실함수: SmoothL1 (Huber Loss)
- 옵티마이저: AdamW (가중치 감쇠 포함)
- 스케줄러: Cosine Annealing Warm Restarts
- 그래디언트 클리핑: 1.0
- 윈도우 중첩: 50%
- Early Stopping: patience=10

Usage:
    python strategies/strategy_B_improved.py --config configs/mamba.toon
    python strategies/strategy_B_improved.py --config configs/lstm.toon --model lstm
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
    parser = argparse.ArgumentParser(description="Run improved strategy (Strategy B)")
    parser.add_argument("--config", type=Path, default=Path("configs/mamba.toon"))
    parser.add_argument("--model", type=str, default=None, help="모델 키 (config 오버라이드)")
    args = parser.parse_args()

    run_strategy(
        strategy_name="improved",
        model_key=args.model,
        config_path=args.config,
        verbose=True,
    )


if __name__ == "__main__":
    main()
