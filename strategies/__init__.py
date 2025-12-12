"""
strategies/ - 학습 전략 실행 스크립트

각 strategy_X.py는 특정 학습 전략을 정의하고 실행합니다.
전략의 핵심 설정이 각 파일 상단에 명확히 드러나도록 구성되어 있습니다.

Available Strategies:
    strategy_A_baseline: 기본 전략 (증강 없음, MSE, Adam)
    strategy_B_improved: 개선 전략 (SmoothL1, AdamW, Cosine LR)
    strategy_C_cutmix: CutMix 증강 + Early Stopping
    strategy_D_mixup: MixUp 증강 + Early Stopping
    strategy_E_aug_all: 복합 증강 (baseline + mixup + cutmix)

Usage:
    python strategies/strategy_A_baseline.py --config configs/mamba.toon --model mamba
"""
