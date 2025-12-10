#!/usr/bin/env python3
"""
run_all.py - 전체 모델 및 전략 일괄 실행 스크립트

모든 모델에 대해 A~E 전략을 순차적으로 실행합니다.
각 모델별로 5개 전략 결과가 results/{model}/ 디렉토리에 저장됩니다.

Usage:
    # 전체 실행 (8 모델 × 5 전략 = 40 실험)
    python run_all.py

    # 특정 모델만 실행
    python run_all.py --models mamba lstm

    # 특정 전략만 실행
    python run_all.py --strategies A B C

    # 특정 모델과 전략 조합
    python run_all.py --models mamba --strategies A B

    # config 파일 지정
    python run_all.py --config configs/custom.toon
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


# 사용 가능한 모델 목록
AVAILABLE_MODELS = [
    "mamba",
    "lstm",
    "bilstm",
    "rnn",
    "physmamba",
    "rhythmmamba",
    "physdiff",
    "transformer",
]

# 전략 매핑: A~E → 실행 스크립트
STRATEGY_SCRIPTS = {
    "A": "tools/tool_A_baseline.py",
    "B": "tools/tool_B_improved.py",
    "C": "tools/tool_C_cutmix.py",
    "D": "tools/tool_D_mixup.py",
    "E": "tools/tool_E_aug_all.py",
}

STRATEGY_NAMES = {
    "A": "baseline",
    "B": "improved",
    "C": "cutmix",
    "D": "mixup",
    "E": "aug_all",
}


def run_experiment(model: str, strategy: str, config: Path) -> bool:
    """단일 실험 실행"""
    script = STRATEGY_SCRIPTS[strategy]
    strategy_name = STRATEGY_NAMES[strategy]

    cmd = [
        sys.executable,
        script,
        "--config", str(config),
        "--model", model,
    ]

    print(f"\n{'='*70}")
    print(f"  [{model}] Strategy {strategy} ({strategy_name})")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] {model}:{strategy_name} failed with code {e.returncode}")
        return False
    except Exception as e:
        print(f"  [ERROR] {model}:{strategy_name} failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="전체 모델 및 전략 일괄 실행",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_all.py                          # 전체 실행
    python run_all.py --models mamba lstm      # 특정 모델만
    python run_all.py --strategies A B C       # 특정 전략만
    python run_all.py --models mamba --strategies A B
        """
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        default=AVAILABLE_MODELS,
        choices=AVAILABLE_MODELS,
        help=f"실행할 모델 목록 (기본: 전체)",
    )
    parser.add_argument(
        "--strategies", "-s",
        nargs="+",
        default=list(STRATEGY_SCRIPTS.keys()),
        choices=list(STRATEGY_SCRIPTS.keys()),
        help="실행할 전략 (A~E, 기본: 전체)",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("configs/mamba.toon"),
        help="기본 config 파일 경로",
    )
    args = parser.parse_args()

    models = args.models
    strategies = args.strategies
    config = args.config

    total = len(models) * len(strategies)

    print("\n" + "=" * 70)
    print("  rPPG → PPG Regression: 일괄 학습 실행")
    print("=" * 70)
    print(f"  시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Config: {config}")
    print(f"  모델 ({len(models)}): {', '.join(models)}")
    print(f"  전략 ({len(strategies)}): {', '.join(strategies)}")
    print(f"  총 실험 수: {total}")
    print("=" * 70)

    results = []
    success_count = 0

    for model in models:
        for strategy in strategies:
            success = run_experiment(model, strategy, config)
            results.append((model, strategy, success))
            if success:
                success_count += 1

    # 결과 요약
    print("\n" + "=" * 70)
    print("  실행 결과 요약")
    print("=" * 70)
    print(f"  완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  성공: {success_count}/{total}")
    print()

    # 성공/실패 테이블
    header = "    Model         | " + " | ".join(f"  {s}  " for s in strategies)
    print(header)
    print("    " + "-" * (len(header) - 4))

    for model in models:
        row = f"    {model:13} |"
        for strategy in strategies:
            for m, s, success in results:
                if m == model and s == strategy:
                    status = " ✓ " if success else " ✗ "
                    row += f"  {status}  |"
                    break
        print(row)

    print("=" * 70)

    # 결과 디렉토리 안내
    print("\n  결과 저장 위치:")
    for model in models:
        print(f"    - results/{model}/")

    if success_count < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
