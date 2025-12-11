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

    # 환경 프로파일링 건너뛰기
    python run_all.py --skip-profile
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def apply_auto_profile() -> dict[str, str]:
    """
    auto_profile.py를 실행하여 환경변수를 현재 프로세스에 적용

    Returns:
        적용된 환경변수 딕셔너리
    """
    profile_script = Path(__file__).parent / "setup" / "auto_profile.py"

    if not profile_script.exists():
        print("[run_all] auto_profile.py not found, skipping environment profiling")
        return {}

    try:
        # auto_profile.py 실행하여 출력 캡처
        result = subprocess.run(
            [sys.executable, str(profile_script)],
            capture_output=True,
            text=True,
        )

        # export KEY="VALUE" 형식 파싱
        env_vars = {}
        for line in result.stdout.strip().split("\n"):
            match = re.match(r'^export\s+(\w+)="([^"]*)"', line)
            if match:
                key, value = match.groups()
                env_vars[key] = value
                os.environ[key] = value

        # stderr에 출력된 정보 표시
        if result.stderr:
            print(result.stderr.strip())

        return env_vars

    except Exception as e:
        print(f"[run_all] Failed to run auto_profile.py: {e}")
        return {}


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


def run_experiment(model: str, strategy: str, config_dir: Path) -> bool:
    """단일 실험 실행"""
    script = STRATEGY_SCRIPTS[strategy]
    strategy_name = STRATEGY_NAMES[strategy]

    # 모델별 config 파일 자동 선택
    model_config = config_dir / f"{model}.toon"
    if not model_config.exists():
        # fallback: 기본 config 사용
        model_config = config_dir / "mamba.toon"

    cmd = [
        sys.executable,
        script,
        "--config", str(model_config),
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
        "--config-dir", "-c",
        type=Path,
        default=Path("configs"),
        help="config 디렉토리 (모델별 {model}.toon 자동 선택)",
    )
    parser.add_argument(
        "--skip-profile",
        action="store_true",
        help="auto_profile.py 실행 건너뛰기",
    )
    args = parser.parse_args()

    # 환경 프로파일링 적용
    if not args.skip_profile:
        print("\n" + "=" * 70)
        print("  환경 프로파일링 적용 중...")
        print("=" * 70)
        env_vars = apply_auto_profile()
        if env_vars:
            print(f"  적용된 환경변수: {len(env_vars)}개")
    else:
        print("\n[run_all] --skip-profile: 환경 프로파일링 건너뜀")

    models = args.models
    strategies = args.strategies
    config_dir = args.config_dir

    total = len(models) * len(strategies)

    print("\n" + "=" * 70)
    print("  rPPG → PPG Regression: 일괄 학습 실행")
    print("=" * 70)
    print(f"  시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Config 디렉토리: {config_dir}/ (모델별 자동 선택)")
    print(f"  모델 ({len(models)}): {', '.join(models)}")
    print(f"  전략 ({len(strategies)}): {', '.join(strategies)}")
    print(f"  총 실험 수: {total}")
    print("=" * 70)

    results = []
    success_count = 0
    skipped_count = 0

    for model in models:
        model_failed = False
        for strategy in strategies:
            # 모델의 첫 전략이 실패하면 나머지 전략 스킵
            if model_failed:
                print(f"\n  [SKIP] {model}:{STRATEGY_NAMES[strategy]} (모델 사용 불가)")
                results.append((model, strategy, None))  # None = skipped
                skipped_count += 1
                continue

            success = run_experiment(model, strategy, config_dir)
            results.append((model, strategy, success))
            if success:
                success_count += 1
            elif strategy == strategies[0]:
                # 첫 번째 전략에서 실패 → 모델 자체가 사용 불가능
                model_failed = True
                print(f"  [!] {model} 모델 사용 불가 - 나머지 전략 스킵")

    # 결과 요약
    print("\n" + "=" * 70)
    print("  실행 결과 요약")
    print("=" * 70)
    print(f"  완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  성공: {success_count}/{total}, 스킵: {skipped_count}, 실패: {total - success_count - skipped_count}")
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
                    if success is None:
                        status = " - "  # skipped
                    elif success:
                        status = " ✓ "
                    else:
                        status = " ✗ "
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
