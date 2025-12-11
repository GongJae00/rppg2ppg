"""
models/ - 신경망 모델 정의

mamba-ssm은 Linux + CUDA 환경에서만 작동합니다.
mamba-ssm 로드 실패 시 Mamba 기반 모델들은 비활성화됩니다.
"""
import warnings

# Mamba 기반 모델들 (조건부 로드)
_MAMBA_AVAILABLE = False
MambaModel = None
PhysMambaModel = None
RhythmMambaModel = None
neg_pearson_loss = None

try:
    from .mamba import MambaModel
    from .physmamba import PhysMambaModel, neg_pearson_loss
    from .rhythmmamba import RhythmMambaModel
    _MAMBA_AVAILABLE = True
except ImportError as e:
    warnings.warn(
        f"[models] mamba-ssm 로드 실패: {e}\n"
        "  → Mamba 기반 모델(mamba, physmamba, rhythmmamba) 비활성화됨\n"
        "  → 해결: pip uninstall mamba-ssm causal-conv1d && pip install causal-conv1d mamba-ssm"
    )

# 항상 사용 가능한 모델들
from .lstm import LSTMModel
from .bilstm import BiLSTMModel
from .rnn import RNNModel
from .physdiff import PhysDiffModel
from .transformer import RadiantModel

__all__ = [
    "MambaModel",
    "LSTMModel",
    "BiLSTMModel",
    "RNNModel",
    "PhysMambaModel",
    "neg_pearson_loss",
    "PhysDiffModel",
    "RhythmMambaModel",
    "RadiantModel",
    "_MAMBA_AVAILABLE",
]
