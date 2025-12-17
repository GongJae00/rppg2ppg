"""
통합 레지스트리 - 모델, 전략, 데이터셋의 데코레이터 기반 자동 등록 시스템

새 컴포넌트 추가 방법:
1. 모델: @register_model(name="custommodel") 데코레이터 사용
2. 전략: @register_strategy(name="customstrategy", code="X") 데코레이터 사용
3. 데이터셋: @register_dataset(name="customdataset") 데코레이터 사용

Example:
    # 모델 등록
    @register_model(name="lstm", default_params={"hidden_size": 64})
    class LSTMModel(nn.Module):
        ...

    # 모델 가져오기
    model = get_model("lstm", seq_len=300)

    # 등록된 모델 목록 확인
    print(list_models())
"""
from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import torch.nn as nn

T = TypeVar("T")


# ===========================================================================
# Model Registry
# ===========================================================================


@dataclass
class ModelInfo:
    """모델 메타정보"""

    cls: Type[nn.Module]
    name: str
    requires: Optional[str] = None  # 필요한 패키지 (예: "mamba-ssm")
    default_params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def is_available(self) -> bool:
        """의존성 패키지가 설치되어 있는지 확인"""
        if self.requires is None:
            return True
        try:
            importlib.import_module(self.requires.replace("-", "_"))
            return True
        except ImportError:
            return False


MODEL_REGISTRY: Dict[str, ModelInfo] = {}

# Mamba 가용성 플래그 (하위 호환성)
_MAMBA_AVAILABLE = False


def register_model(
    name: str,
    requires: Optional[str] = None,
    default_params: Optional[Dict[str, Any]] = None,
    description: str = "",
):
    """
    모델 자동 등록 데코레이터

    Args:
        name: 모델 식별 키 (예: "lstm", "mamba")
        requires: 필요한 패키지 (예: "mamba-ssm")
        default_params: 모델 기본 파라미터
        description: 모델 설명

    Example:
        @register_model(name="lstm", default_params={"hidden_size": 64})
        class LSTMModel(nn.Module):
            ...
    """

    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        info = ModelInfo(
            cls=cls,
            name=name,
            requires=requires,
            default_params=default_params or {},
            description=description,
        )

        # 의존성 체크
        if requires is not None and not info.is_available():
            warnings.warn(
                f"[registry] Model '{name}' requires '{requires}' which is not installed.\n"
                f"  → 해결: pip install {requires}"
            )
        else:
            MODEL_REGISTRY[name] = info

        return cls

    return decorator


def get_model(model_key: str, seq_len: int = 300, **kwargs) -> nn.Module:
    """
    레지스트리에서 모델 인스턴스 생성

    Args:
        model_key: 모델 이름
        seq_len: 시퀀스 길이 (일부 모델에서 필요)
        **kwargs: 추가 모델 파라미터 (기본값 오버라이드)

    Returns:
        초기화된 모델 인스턴스

    Raises:
        ValueError: 등록되지 않은 모델 키 또는 의존성 미설치
    """
    if model_key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))

        # Mamba 계열 모델인지 확인
        mamba_models = {"mamba", "physmamba_td", "physmamba_sssd", "rhythmmamba"}
        if model_key in mamba_models:
            raise ValueError(
                f"Model '{model_key}' requires mamba-ssm which is not installed.\n"
                f"  → 해결: pip uninstall mamba-ssm causal-conv1d && pip install causal-conv1d mamba-ssm\n"
                f"  → Available models: {available}"
            )

        raise ValueError(f"Unknown model: '{model_key}'. Available: {available}")

    info = MODEL_REGISTRY[model_key]

    # 의존성 재확인
    if not info.is_available():
        raise ValueError(
            f"Model '{model_key}' requires '{info.requires}' which is not installed.\n"
            f"  → 해결: pip install {info.requires}"
        )

    # 파라미터 병합 (default + kwargs)
    params = {**info.default_params, **kwargs}

    # seq_len이 필요한 모델인지 확인 (생성자 시그니처 검사)
    import inspect

    sig = inspect.signature(info.cls.__init__)
    if "seq_len" in sig.parameters:
        params["seq_len"] = seq_len

    return info.cls(**params)


def list_models() -> List[str]:
    """등록된 모델 키 목록 반환"""
    return list(MODEL_REGISTRY.keys())


def list_available_models() -> List[str]:
    """현재 사용 가능한 모델 키 목록 반환 (의존성 설치된 모델만)"""
    return [name for name, info in MODEL_REGISTRY.items() if info.is_available()]


def get_model_info(model_key: str) -> Optional[ModelInfo]:
    """모델 메타정보 반환"""
    return MODEL_REGISTRY.get(model_key)


def is_mamba_available() -> bool:
    """mamba-ssm 사용 가능 여부 (하위 호환성)"""
    try:
        importlib.import_module("mamba_ssm")
        return True
    except ImportError:
        return False


# ===========================================================================
# Strategy Registry
# ===========================================================================


@dataclass
class StrategyConfig:
    """전략 설정 베이스 클래스"""

    name: str = ""
    code: str = ""  # A, B, C, D, E, F, ...

    # 증강
    augment: Optional[str] = None  # None, "cutmix", "mixup", "all"
    augment_alpha: float = 0.2

    # 손실함수
    loss: str = "mse"  # "mse", "smoothl1", "huber"

    # 옵티마이저
    optimizer: str = "adam"  # "adam", "adamw", "sgd"
    lr: float = 1e-4
    weight_decay: float = 0.0

    # 스케줄러
    scheduler: Optional[str] = None  # None, "cosine", "step"
    scheduler_T0: int = 10
    scheduler_Tmult: int = 2

    # 그래디언트
    clip_grad: Optional[float] = None

    # 학습
    epochs: int = 100
    patience: int = 10

    # 예측
    overlap: float = 0.0
    fill_tail: bool = True


STRATEGY_REGISTRY: Dict[str, StrategyConfig] = {}


def register_strategy(name: str, code: str):
    """
    전략 자동 등록 데코레이터

    Args:
        name: 전략 식별 키 (예: "baseline", "improved")
        code: 전략 코드 (예: "A", "B")

    Example:
        @register_strategy(name="baseline", code="A")
        class BaselineStrategy(StrategyConfig):
            loss = "mse"
            optimizer = "adam"
    """

    def decorator(cls: Type[StrategyConfig]) -> Type[StrategyConfig]:
        # 클래스 속성에서 필드 값 추출 (dataclass 기본값 대신 클래스 속성 사용)
        instance = StrategyConfig(
            name=name,
            code=code,
            augment=getattr(cls, "augment", None),
            augment_alpha=getattr(cls, "augment_alpha", 0.2),
            loss=getattr(cls, "loss", "mse"),
            optimizer=getattr(cls, "optimizer", "adam"),
            lr=getattr(cls, "lr", 1e-4),
            weight_decay=getattr(cls, "weight_decay", 0.0),
            scheduler=getattr(cls, "scheduler", None),
            scheduler_T0=getattr(cls, "scheduler_T0", 10),
            scheduler_Tmult=getattr(cls, "scheduler_Tmult", 2),
            clip_grad=getattr(cls, "clip_grad", None),
            epochs=getattr(cls, "epochs", 100),
            patience=getattr(cls, "patience", 10),
            overlap=getattr(cls, "overlap", 0.0),
            fill_tail=getattr(cls, "fill_tail", True),
        )
        STRATEGY_REGISTRY[name] = instance
        return cls

    return decorator


def get_strategy(strategy_name: str) -> StrategyConfig:
    """전략 설정 반환"""
    if strategy_name not in STRATEGY_REGISTRY:
        available = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
        raise ValueError(f"Unknown strategy: '{strategy_name}'. Available: {available}")
    return STRATEGY_REGISTRY[strategy_name]


def list_strategies() -> List[str]:
    """등록된 전략 목록 반환"""
    return list(STRATEGY_REGISTRY.keys())


# ===========================================================================
# Dataset Registry
# ===========================================================================


@dataclass
class DatasetInfo:
    """데이터셋 메타정보"""

    key: str
    display_name: str
    subject_counts: List[int]
    fs: float = 30.0  # 샘플링 주파수
    label_filename: str = ""
    prediction_dirname: str = ""

    @property
    def total_subjects(self) -> int:
        return len(self.subject_counts)

    @property
    def total_samples(self) -> int:
        return sum(self.subject_counts)

    def get_label_path(self, root: Union[str, Path] = "data") -> Path:
        return Path(root) / self.label_filename

    def get_prediction_dir(self, root: Union[str, Path] = "data") -> Path:
        return Path(root) / self.prediction_dirname


DATASET_REGISTRY: Dict[str, DatasetInfo] = {}


def register_dataset(
    name: str,
    display_name: Optional[str] = None,
    fs: float = 30.0,
    label_filename: Optional[str] = None,
    prediction_dirname: Optional[str] = None,
):
    """
    데이터셋 자동 등록 데코레이터

    클래스에 SUBJECT_COUNTS 속성이 필요합니다.

    Args:
        name: 데이터셋 식별 키 (예: "pure", "ubfc")
        display_name: 표시 이름 (예: "PURE", "UBFC")
        fs: 샘플링 주파수
        label_filename: 라벨 파일명 (기본: "{DISPLAY}_label.npy")
        prediction_dirname: 예측 디렉토리명 (기본: "{DISPLAY}_prediction")

    Example:
        @register_dataset(name="pure", display_name="PURE", fs=30.0)
        class PUREDataset:
            SUBJECT_COUNTS = [6, 6, 8, 8, ...]
    """

    def decorator(cls: Type[T]) -> Type[T]:
        disp = display_name or name.upper()
        info = DatasetInfo(
            key=name,
            display_name=disp,
            subject_counts=getattr(cls, "SUBJECT_COUNTS", []),
            fs=fs,
            label_filename=label_filename or f"{disp}_label.npy",
            prediction_dirname=prediction_dirname or f"{disp}_prediction",
        )
        DATASET_REGISTRY[name] = info
        return cls

    return decorator


def get_dataset_info(dataset_name: str) -> DatasetInfo:
    """데이터셋 메타정보 반환"""
    # 대소문자 무시
    key = dataset_name.strip().lower()
    if key not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise ValueError(f"Unknown dataset: '{dataset_name}'. Available: {available}")
    return DATASET_REGISTRY[key]


def list_datasets() -> List[str]:
    """등록된 데이터셋 목록 반환"""
    return list(DATASET_REGISTRY.keys())


# ===========================================================================
# Auto-discovery: 모델/데이터셋 모듈 자동 로드
#
# strategies/ 폴더는 실행 스크립트(엔트리포인트)만 두는 영역이라
# 레지스트리에서 자동 import 하지 않습니다. (전략 정의는 src/strategy.py)
# ===========================================================================


def _auto_import_models():
    """models/ 디렉토리의 모든 모델 모듈 자동 import"""
    import os

    models_dir = Path(__file__).parent.parent / "models"
    if not models_dir.exists():
        return

    # 디버그 모드: RPPG_DEBUG=1 환경변수로 활성화
    debug_mode = os.environ.get("RPPG_DEBUG", "0") == "1"

    # Mamba 의존성 모듈 목록 (선택적 의존성)
    mamba_modules = {"mamba", "physmamba_td", "physmamba_sssd", "rhythmmamba"}

    for f in models_dir.glob("*.py"):
        if f.stem.startswith("_"):
            continue
        try:
            importlib.import_module(f"models.{f.stem}")
        except ImportError as e:
            # Mamba 계열은 선택적 의존성 → 디버그 모드에서만 경고
            if f.stem in mamba_modules:
                if debug_mode:
                    warnings.warn(
                        f"[registry] 선택적 모델 '{f.stem}' 로드 실패 (mamba-ssm 미설치): {e}"
                    )
            else:
                # 필수 모델 로드 실패 → 항상 경고
                warnings.warn(
                    f"[registry] 모델 '{f.stem}' 임포트 실패: {e}"
                )


def _auto_import_datasets():
    """datasets/ 디렉토리의 모든 데이터셋 모듈 자동 import"""
    datasets_dir = Path(__file__).parent.parent / "datasets"
    if not datasets_dir.exists():
        return

    for f in datasets_dir.glob("*.py"):
        if f.stem.startswith("_"):
            continue
        try:
            importlib.import_module(f"datasets.{f.stem}")
        except ImportError as e:
            warnings.warn(f"Failed to import dataset module '{f.stem}': {e}")


# 모듈 로드 시 자동 discovery 실행
_auto_import_models()
_auto_import_datasets()
