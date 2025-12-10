"""
tools/ - 학습 전략 실행 스크립트

각 tool_X.py는 특정 학습 전략을 정의하고 실행합니다.
전략의 핵심 설정이 각 파일 상단에 명확히 드러나도록 구성되어 있습니다.

Available Tools:
    tool_A_baseline: 기본 전략 (증강 없음, MSE, Adam)
    tool_B_improved: 개선 전략 (SmoothL1, AdamW, Cosine LR)
    tool_C_cutmix: CutMix 증강 + Early Stopping
    tool_D_mixup: MixUp 증강 + Early Stopping
    tool_E_aug_all: 복합 증강 (baseline + mixup + cutmix)

Usage:
    python tools/tool_A_baseline.py --config configs/mamba.toon --model mamba
"""
