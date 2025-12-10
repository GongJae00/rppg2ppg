"""
src/ - rPPG to PPG 파이프라인 핵심 모듈

이 패키지는 학습/추론 파이프라인의 핵심 로직을 담당합니다.
tools/의 각 전략 스크립트에서 이 모듈들을 import하여 사용합니다.

Modules:
    registry: 모델 레지스트리 (모든 모델의 단일 등록 소스)
    trainer: 학습 로직 (전략 독립적인 학습 엔진)
    predictor: 추론 로직 (슬라이딩 윈도우 예측, 후처리)
"""
from .registry import MODEL_REGISTRY, get_model
from .trainer import Trainer
from .predictor import Predictor

__all__ = [
    "MODEL_REGISTRY",
    "get_model",
    "Trainer",
    "Predictor",
]
