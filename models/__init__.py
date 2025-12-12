"""
models/ - 신경망 모델 정의

데코레이터 기반 자동 등록 시스템을 사용합니다.
각 모델 파일에서 @register_model 데코레이터로 레지스트리에 자동 등록됩니다.

mamba-ssm은 Linux + CUDA 환경에서만 작동합니다.
mamba-ssm 로드 실패 시 Mamba 기반 모델들은 비활성화됩니다.

사용법:
    from src.registry import get_model, list_models

    model = get_model("lstm", seq_len=300)
    print(list_models())
"""
import warnings

# ---------------------------------------------------------------------------
# 항상 사용 가능한 모델들 (자동 등록)
# ---------------------------------------------------------------------------
from .lstm import LSTMModel
from .bilstm import BiLSTMModel
from .rnn import RNNModel
from .physdiff import PhysDiffModel
from .transformer import RadiantModel

# ---------------------------------------------------------------------------
# Mamba 기반 모델들 (조건부 로드 + 자동 등록)
# ---------------------------------------------------------------------------
_MAMBA_AVAILABLE = False
MambaModel = None
PhysMamba_TD = None
PhysMamba_SSSD = None
RhythmMambaModel = None
neg_pearson_loss = None

try:
    from .mamba import MambaModel
    from .physmamba_td import PhysMamba_TD, neg_pearson_loss
    from .physmamba_sssd import PhysMamba_SSSD
    from .rhythmmamba import RhythmMambaModel
    _MAMBA_AVAILABLE = True
except ImportError as e:
    warnings.warn(
        f"[models] mamba-ssm 로드 실패: {e}\n"
        "  → Mamba 기반 모델(mamba, physmamba_td, physmamba_sssd, rhythmmamba) 비활성화됨\n"
        "  → 해결: pip uninstall mamba-ssm causal-conv1d && pip install causal-conv1d mamba-ssm"
    )


__all__ = [
    # 항상 사용 가능
    "LSTMModel",
    "BiLSTMModel",
    "RNNModel",
    "PhysDiffModel",
    "RadiantModel",
    # Mamba 기반 (조건부)
    "MambaModel",
    "PhysMamba_TD",
    "PhysMamba_SSSD",
    "RhythmMambaModel",
    "neg_pearson_loss",
    # 플래그
    "_MAMBA_AVAILABLE",
]
