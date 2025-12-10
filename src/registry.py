"""
모델 레지스트리 - 모든 모델의 단일 등록 소스

새 모델을 추가하려면:
1. models/에 모델 클래스 정의
2. models/__init__.py에 export 추가
3. 여기 MODEL_REGISTRY에 등록

Example:
    # 모델 가져오기
    model = get_model("mamba", seq_len=300)

    # 등록된 모델 목록 확인
    print(list(MODEL_REGISTRY.keys()))
"""
from __future__ import annotations

from typing import Callable, Dict, Type

import torch.nn as nn

from models import (
    MambaModel,
    LSTMModel,
    BiLSTMModel,
    RNNModel,
    PhysMambaModel,
    PhysDiffModel,
    RhythmMambaModel,
    RadiantModel,
)


# ---------------------------------------------------------------------------
# 모델 레지스트리
# ---------------------------------------------------------------------------
# 각 모델은 (seq_len: int) -> nn.Module 형태의 팩토리 함수로 등록
# seq_len이 필요 없는 모델도 일관성을 위해 인자로 받음

MODEL_REGISTRY: Dict[str, Callable[[int], nn.Module]] = {
    "mamba": lambda seq_len: MambaModel(),
    "lstm": lambda seq_len: LSTMModel(),
    "bilstm": lambda seq_len: BiLSTMModel(),
    "rnn": lambda seq_len: RNNModel(),
    "physmamba": lambda seq_len: PhysMambaModel(),
    "physdiff": lambda seq_len: PhysDiffModel(),
    "rhythmmamba": lambda seq_len: RhythmMambaModel(),
    "transformer": lambda seq_len: RadiantModel(seq_len=seq_len),
}


def get_model(model_key: str, seq_len: int = 300) -> nn.Module:
    """
    레지스트리에서 모델 인스턴스 생성

    Args:
        model_key: 모델 이름 (mamba, lstm, bilstm, rnn, physmamba, physdiff, rhythmmamba, transformer)
        seq_len: 시퀀스 길이 (transformer 모델에서 필요)

    Returns:
        초기화된 모델 인스턴스

    Raises:
        ValueError: 등록되지 않은 모델 키
    """
    if model_key not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: '{model_key}'. Available: {available}")

    return MODEL_REGISTRY[model_key](seq_len)


def list_models() -> list[str]:
    """등록된 모델 키 목록 반환"""
    return list(MODEL_REGISTRY.keys())
