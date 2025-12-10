# rPPG to PPG Regression Pipeline

비디오에서 추출한 원격 심박신호(rPPG)를 직접 측정한 심박신호(PPG)로 변환하는 딥러닝 파이프라인입니다.

## 프로젝트 구조

```
rppg2ppg/
├── src/                    # 핵심 파이프라인 모듈
│   ├── registry.py         # 모델 레지스트리 (단일 등록 소스)
│   ├── trainer.py          # 학습 엔진
│   └── predictor.py        # 추론 엔진
│
├── models/                 # 신경망 모델 정의
│   ├── mamba.py            # MambaModel (SSM 기반)
│   ├── lstm.py             # LSTMModel
│   ├── bilstm.py           # BiLSTMModel (양방향)
│   ├── rnn.py              # RNNModel
│   ├── physmamba.py        # PhysMambaModel (BiMamba + SE)
│   ├── rhythmmamba.py      # RhythmMambaModel
│   ├── physdiff.py         # PhysDiffModel (Transformer)
│   └── transformer.py      # RadiantModel (ViT 기반)
│
├── tools/                  # 학습 전략 실행 스크립트
│   ├── tool_A_baseline.py  # 기본 전략 (MSE, Adam)
│   ├── tool_B_improved.py  # 개선 전략 (SmoothL1, AdamW, Cosine)
│   ├── tool_C_cutmix.py    # CutMix 증강 + Early Stopping
│   ├── tool_D_mixup.py     # MixUp 증강 + Early Stopping
│   └── tool_E_aug_all.py   # 복합 증강 (3배 배치)
│
├── utils/                  # 유틸리티 모듈
│   ├── data.py             # 데이터 로드/처리
│   ├── augmentation.py     # 데이터 증강 (MixUp, CutMix)
│   ├── signal.py           # 신호 처리 (필터링)
│   ├── metrics.py          # 평가 지표 (PSNR, Pearson 등)
│   ├── plot.py             # 시각화
│   └── toon.py             # TOON 설정 파서
│
├── configs/                # 모델별 TOON 설정 파일
│   ├── mamba.toon
│   ├── lstm.toon
│   └── ...
│
├── real_data/              # 입력 데이터
├── results/                # 출력 (모델별 정리)
│   ├── mamba/              # mamba 모델 결과
│   ├── lstm/               # lstm 모델 결과
│   └── .../                # 각 모델별 디렉토리
│
└── run_all.py              # 전체 모델/전략 일괄 실행
```

## 빠른 시작

### 1. 의존성 설치

```bash
pip install torch numpy scipy scikit-learn matplotlib mamba-ssm
```

### 2. 학습 실행

#### 개별 실행

```bash
# Mamba 모델 + Baseline 전략
python tools/tool_A_baseline.py --config configs/mamba.toon

# LSTM 모델 + MixUp 전략
python tools/tool_D_mixup.py --config configs/lstm.toon

# 특정 모델로 오버라이드
python tools/tool_B_improved.py --config configs/mamba.toon --model bilstm
```

#### 일괄 실행 (run_all.py)

```bash
# 전체 실행 (8 모델 × 5 전략 = 40 실험)
python run_all.py

# 특정 모델만 실행 (5 전략씩)
python run_all.py --models mamba lstm

# 특정 전략만 실행 (8 모델씩)
python run_all.py --strategies A B C

# 특정 모델 + 특정 전략 조합
python run_all.py --models mamba lstm --strategies A B

# config 파일 지정
python run_all.py --config configs/custom.toon
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--models`, `-m` | 실행할 모델 목록 | 전체 8개 모델 |
| `--strategies`, `-s` | 실행할 전략 (A~E) | 전체 A~E |
| `--config`, `-c` | 기본 config 파일 | configs/mamba.toon |

### 3. 출력 확인

학습 후 `results/{model}/` 디렉토리에 저장:
- `{model}_{strategy}.pth` - 모델 가중치
- `{model}_{strategy}_loss.png` - 학습/검증 손실 곡선
- `{model}_{strategy}_prediction.npy` - 최종 예측값
- `{model}_{strategy}_best.pth` - 최적 체크포인트 (Early Stopping 시)

예시 결과 구조:
```
results/
├── mamba/
│   ├── mamba_baseline.pth
│   ├── mamba_baseline_loss.png
│   ├── mamba_baseline_prediction.npy
│   ├── mamba_improved.pth
│   ├── mamba_cutmix.pth
│   └── ...
├── lstm/
│   ├── lstm_baseline.pth
│   └── ...
└── ...
```

## 학습 전략

| Tool | 전략명 | 증강 | 손실 | 옵티마이저 | Early Stop |
|------|--------|------|------|------------|------------|
| A | baseline | 없음 | MSE | Adam | 없음 |
| B | improved | 없음 | SmoothL1 | AdamW | 없음 |
| C | cutmix | CutMix | SmoothL1 | AdamW | patience=8 |
| D | mixup | MixUp | SmoothL1 | AdamW | patience=8 |
| E | aug_all | 복합 | SmoothL1 | AdamW | patience=12 |

## 모델 목록

| 모델 | 설명 | 특징 |
|------|------|------|
| mamba | MambaModel | SSM 기반, 6개 블록 |
| lstm | LSTMModel | 기본 LSTM |
| bilstm | BiLSTMModel | 양방향 LSTM |
| rnn | RNNModel | 기본 RNN |
| physmamba | PhysMambaModel | BiMamba + SE 어텐션 |
| rhythmmamba | RhythmMambaModel | 리듬 특화 Mamba |
| physdiff | PhysDiffModel | Transformer + 신호 분해 |
| transformer | RadiantModel | ViT 기반 |

## TOON 설정 파일

[TOON (Token-Oriented Object Notation)](https://github.com/toon-format) 형식의 설정 파일:

```toon
# configs/mamba.toon
model: mamba

data:
  pred_path: real_data/PURE_POS_prediction.npy
  label_path: real_data/PURE_label.npy

train:
  seq_len: 300
  batch_size: 32
  epochs: 100
  split: 0.8
```

> **참고**: 결과는 자동으로 `results/{model}/`에 저장됩니다.

## 새 모델 추가하기

### 1. 모델 정의 (`models/mymodel.py`)

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # 모델 구조 정의

    def forward(self, x):
        # x: (B, L, 1) -> output: (B, L)
        return output
```

### 2. 모델 export (`models/__init__.py`)

```python
from .mymodel import MyModel

__all__ = [..., "MyModel"]
```

### 3. 레지스트리 등록 (`src/registry.py`)

```python
from models import MyModel

MODEL_REGISTRY = {
    ...
    "mymodel": lambda seq_len: MyModel(),
}
```

### 4. 설정 파일 생성 (`configs/mymodel.toon`)

```toon
model: mymodel

data:
  pred_path: real_data/PURE_POS_prediction.npy
  label_path: real_data/PURE_label.npy

train:
  seq_len: 300
  batch_size: 32
```

### 5. 실행

```bash
python tools/tool_B_improved.py --config configs/mymodel.toon
```

## 새 전략 추가하기

`tools/tool_A_baseline.py`를 복사하여 `tools/tool_F_custom.py` 생성 후 `STRATEGY` 딕셔너리 수정:

```python
STRATEGY_NAME = "custom"

STRATEGY = {
    "augment": None,
    "loss": "mse",
    "optimizer": "sgd",
    "lr": 0.01,
    ...
}
```

## 데이터 형식

- **입력**: `real_data/PURE_POS_prediction.npy` - rPPG 신호 (1D numpy array)
- **레이블**: `real_data/PURE_label.npy` - PPG 신호 (1D numpy array)

다른 데이터셋 사용 시 TOON 설정의 `data.pred_path`, `data.label_path` 수정.

## 평가 지표

학습 완료 시 출력:
- **Pearson 상관계수**: 예측값과 실제값의 선형 상관
- **PSNR**: Peak Signal-to-Noise Ratio (dB)

추가 평가는 `utils/metrics.py` 참조:
- MAE, RMSE, MAPE 등

## 요구사항

- Python 3.10+
- PyTorch 2.0+
- CUDA (권장, GPU 학습)

## 라이선스

MIT License
