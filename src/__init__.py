"""
src/ - rPPG to PPG 파이프라인 핵심 모듈

이 패키지는 학습/추론 파이프라인의 핵심 로직을 담당합니다.
strategies/의 각 전략 스크립트에서 이 모듈들을 import하여 사용합니다.

Modules:
    registry: 통합 레지스트리 (모델, 전략, 데이터셋 자동 등록)
    strategy: 전략 정의 (A~E 전략 클래스)
    factory: 컴포넌트 팩토리 (loss, optimizer, scheduler, augment)
    runner: 통합 실행 함수
    trainer: 학습 엔진 (전략 독립적)
    train: 학습 루프 (에폭 루프, Early Stopping)
    predictor: 추론 엔진 (슬라이딩 윈도우 예측, 후처리)
    evaluator: 평가 엔진 (다중 지표 계산)

Quick Start:
    # 전략으로 학습 실행
    from src.runner import run_strategy
    run_strategy("baseline", model_key="lstm", config_path="configs/lstm.toon")

    # 모델만 가져오기
    from src.registry import get_model
    model = get_model("lstm", seq_len=300)

    # 전략 설정 가져오기
    from src.strategy import get_strategy
    strategy = get_strategy("baseline")
"""
# Registry
from .registry import (
    MODEL_REGISTRY,
    STRATEGY_REGISTRY,
    DATASET_REGISTRY,
    ModelInfo,
    StrategyConfig,
    DatasetInfo,
    register_model,
    register_strategy,
    register_dataset,
    get_model,
    get_strategy,
    get_dataset_info,
    list_models,
    list_strategies,
    list_datasets,
    is_mamba_available,
)

# Strategy (전략 정의)
from .strategy import (
    BaselineStrategy,
    ImprovedStrategy,
    CutMixStrategy,
    MixUpStrategy,
    AugAllStrategy,
    get_strategy_by_code,
    list_strategy_codes,
)

# Factory (컴포넌트 팩토리)
from .factory import (
    build_criterion,
    build_optimizer,
    build_scheduler,
    build_augment,
    build_from_strategy,
)

# Runner (통합 실행)
from .runner import run_strategy, run_strategy_by_code

# Trainer (학습 엔진)
from .trainer import Trainer

# Train (학습 루프)
from .train import fit, fit_legacy, TrainResult, TrainCallback

# Predictor (추론 엔진)
from .predictor import Predictor

# Evaluator (평가 엔진)
from .evaluator import Evaluator, EvaluationRecord


__all__ = [
    # Registry
    "MODEL_REGISTRY",
    "STRATEGY_REGISTRY",
    "DATASET_REGISTRY",
    "ModelInfo",
    "StrategyConfig",
    "DatasetInfo",
    "register_model",
    "register_strategy",
    "register_dataset",
    "get_model",
    "get_strategy",
    "get_dataset_info",
    "list_models",
    "list_strategies",
    "list_datasets",
    "is_mamba_available",
    # Strategy
    "BaselineStrategy",
    "ImprovedStrategy",
    "CutMixStrategy",
    "MixUpStrategy",
    "AugAllStrategy",
    "get_strategy_by_code",
    "list_strategy_codes",
    # Factory
    "build_criterion",
    "build_optimizer",
    "build_scheduler",
    "build_augment",
    "build_from_strategy",
    # Runner
    "run_strategy",
    "run_strategy_by_code",
    # Trainer
    "Trainer",
    # Train
    "fit",
    "fit_legacy",
    "TrainResult",
    "TrainCallback",
    # Predictor
    "Predictor",
    # Evaluator
    "Evaluator",
    "EvaluationRecord",
]
