# rPPG to PPG Regression Pipeline

비디오에서 추출한 원격 심박신호(rPPG)를 직접 측정한 심박신호(PPG)로 변환하는 딥러닝 파이프라인입니다.

---

## 목차

- [프로젝트 구조](#프로젝트-구조)
- [빠른 시작](#빠른-시작)
- [학습 실행](#학습-실행)
- [컴포넌트 추가 가이드](#컴포넌트-추가-가이드)
  - [새 모델 추가](#1-새-모델-추가)
  - [새 전략 추가](#2-새-전략-추가)
  - [새 데이터셋 추가](#3-새-데이터셋-추가)
- [아키텍처](#아키텍처)
- [참조](#참조)

---

## 프로젝트 구조

```
rppg2ppg/
│
├── src/                        # 핵심 파이프라인
│   ├── registry.py             # 통합 레지스트리 (데코레이터 기반 자동 등록)
│   ├── strategy.py             # 전략 정의 (A~E)
│   ├── factory.py              # 컴포넌트 팩토리 (loss, optimizer, scheduler)
│   ├── runner.py               # 통합 실행 함수
│   ├── train.py                # 학습 루프 (fit)
│   ├── trainer.py              # 저수준 학습 엔진 (배치 처리, AMP)
│   ├── predictor.py            # 추론 엔진
│   └── evaluate.py             # 평가 모듈
│
├── models/                     # 신경망 모델 (데코레이터로 자동 등록)
│   ├── lstm.py                 # @register_model(name="lstm")
│   ├── bilstm.py               # @register_model(name="bilstm")
│   ├── rnn.py                  # @register_model(name="rnn")
│   ├── mamba.py                # @register_model(name="mamba", requires="mamba-ssm")
│   ├── physmamba_td.py         # @register_model(name="physmamba_td")
│   ├── physmamba_sssd.py       # @register_model(name="physmamba_sssd")
│   ├── rhythmmamba.py          # @register_model(name="rhythmmamba")
│   ├── physdiff.py             # @register_model(name="physdiff")
│   └── transformer.py          # @register_model(name="transformer")
│
├── datasets/                   # 데이터셋 정의 (데코레이터로 자동 등록)
│   ├── pure.py                 # @register_dataset(name="pure")
│   ├── ubfc.py                 # @register_dataset(name="ubfc")
│   └── cohface.py              # @register_dataset(name="cohface")
│
├── strategies/                 # 전략 실행 스크립트 (간소화됨)
│   ├── strategy_A_baseline.py  # → run_strategy("baseline", ...)
│   ├── strategy_B_improved.py  # → run_strategy("improved", ...)
│   ├── strategy_C_cutmix.py    # → run_strategy("cutmix", ...)
│   ├── strategy_D_mixup.py     # → run_strategy("mixup", ...)
│   └── strategy_E_aug_all.py   # → run_strategy("aug_all", ...)
│
├── utils/                      # 유틸리티
│   ├── data.py                 # 데이터 로드/처리
│   ├── augmentation.py         # MixUp, CutMix
│   ├── metrics.py              # 평가 지표
│   ├── metrics_biosignal.py    # 생체신호 특화 지표
│   ├── early_stopping.py       # Early Stopping
│   └── ...
│
├── configs/                    # TOON 설정 파일
│   ├── mamba.toon
│   ├── lstm.toon
│   └── ...
│
├── data/                       # 입력 데이터
│   ├── PURE_label.npy
│   ├── PURE_prediction/
│   └── ...
│
└── results/                    # 출력 (모델별)
    └── {model}/
        ├── checkpoints/
        ├── plots/
        └── predictions/
```

---

## 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 활성화
pyenv activate rppg-310    # 또는 conda activate customenv

# 자동 설치 (CUDA/MPS 자동 감지)
python setup/install.py

# CPU 전용
python setup/install.py --cpu
```

### 2. 환경 프로파일링

```bash
# 하드웨어 최적화 환경변수 설정
eval "$(python setup/auto_profile.py)"
```

### 3. 학습 실행

```bash
# 단일 실행
python strategies/strategy_A_baseline.py --config configs/mamba.toon

# 일괄 실행
python run_all.py --models lstm bilstm --strategies A B C
```

---

## 학습 실행

### 개별 실행

```bash
# 기본 사용법
python strategies/strategy_{A~E}_{strategy}.py --config configs/{model}.toon

# 예시
python strategies/strategy_A_baseline.py --config configs/mamba.toon
python strategies/strategy_D_mixup.py --config configs/lstm.toon

# 모델 오버라이드
python strategies/strategy_B_improved.py --config configs/mamba.toon --model bilstm
```

### 일괄 실행

```bash
# 전체 (9 모델 × 5 전략 = 45 실험)
python run_all.py

# 특정 모델만
python run_all.py --models mamba lstm bilstm

# 특정 전략만
python run_all.py --strategies A B C

# 조합
python run_all.py --models mamba lstm --strategies A B
```

### 학습 전략

| Strategy | 이름 | 증강 | 손실함수 | 옵티마이저 | 스케줄러 | Early Stop |
|:----:|:----:|:----:|:--------:|:----------:|:--------:|:----------:|
| **A** | baseline | - | MSE | Adam | - | 10 |
| **B** | improved | - | SmoothL1 | AdamW | Cosine | 10 |
| **C** | cutmix | CutMix | SmoothL1 | AdamW | Cosine | 8 |
| **D** | mixup | MixUp | SmoothL1 | AdamW | Cosine | 8 |
| **E** | aug_all | All | SmoothL1 | AdamW | Cosine | 12 |

### 출력 구조

```
results/{model}/
├── checkpoints/
│   ├── {model}_{A~E}_{strategy}.pth
│   └── {model}_{A~E}_{strategy}_best.pth
├── plots/
│   └── {model}_{A~E}_{strategy}_loss.png
└── predictions/
    └── {model}_{A~E}_{strategy}_prediction.npy
```

---

## 컴포넌트 추가 가이드

데코레이터 기반 자동 등록 시스템으로 **파일 1개만 추가**하면 새 컴포넌트가 등록됩니다.

### 1. 새 모델 추가

> **파일 1개만 생성하면 끝!**

#### Step 1: 모델 파일 생성

`models/custom_model.py` 파일을 생성합니다:

```python
"""
Custom Model
"""
import torch
import torch.nn as nn
from src.registry import register_model


@register_model(
    name="custommodel",               # 모델 키 (필수)
    requires=None,                     # 의존 패키지 (없으면 None)
    default_params={                   # 기본 파라미터
        "hidden_size": 128,
        "num_layers": 3,
    },
    description="Custom model",        # 설명 (선택)
)
class CustomModel(nn.Module):
    """
    커스텀 모델

    Args:
        seq_len: 입력 시퀀스 길이 (자동 전달됨)
        hidden_size: 은닉층 크기
        num_layers: 레이어 수
    """

    def __init__(
        self,
        seq_len: int = 300,
        hidden_size: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        self.seq_len = seq_len

        # 모델 구조 정의
        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.decoder = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, 1) - 입력 rPPG 신호

        Returns:
            (batch, seq_len) - 출력 PPG 신호
        """
        out, _ = self.encoder(x)      # (B, L, H*2)
        out = self.decoder(out)        # (B, L, 1)
        return out.squeeze(-1)         # (B, L)
```

#### Step 2: 설정 파일 생성 (선택)

`configs/custommodel.toon`:

```toon
model: custommodel

data:
  pred_path: data/PURE_prediction/PURE_POS_prediction.npy
  label_path: data/PURE_label.npy

train:
  seq_len: 300
  batch_size: 32
  epochs: 100
```

#### Step 3: 실행

```bash
# 새 모델로 학습
python strategies/strategy_A_baseline.py --config configs/custommodel.toon

# 또는 기존 config에서 모델만 변경
python strategies/strategy_B_improved.py --config configs/mamba.toon --model custommodel
```

#### 모델 데코레이터 옵션

| 파라미터 | 타입 | 설명 | 예시 |
|----------|------|------|------|
| `name` | str | 모델 식별 키 (필수) | `"custommodel"` |
| `requires` | str | 필요 패키지 | `"mamba-ssm"` |
| `default_params` | dict | 기본 생성자 인자 | `{"hidden_size": 64}` |
| `description` | str | 모델 설명 | `"Custom model"` |

#### 주의사항

- `forward()` 입출력: `(B, L, 1)` → `(B, L)`
- `seq_len`이 필요한 경우 생성자에 포함 (자동 전달됨)
- Mamba 계열 모델은 `requires="mamba-ssm"` 필수

---

### 2. 새 전략 추가

> **strategy.py에 클래스 1개 추가 + strategy 스크립트 1개 생성**

#### Step 1: 전략 클래스 추가

`src/strategy.py` 파일에 새 전략 클래스를 추가합니다:

```python
@register_strategy(name="advanced", code="F")
class AdvancedStrategy(StrategyConfig):
    """
    Strategy F: 고급 전략

    - Huber Loss + Label Smoothing
    - AdamW + Lookahead
    - OneCycleLR 스케줄러
    """

    # 증강 설정
    augment = "cutmix"          # None, "cutmix", "mixup", "all"
    augment_alpha = 0.4         # Beta 분포 파라미터

    # 손실함수
    loss = "huber"              # "mse", "smoothl1", "huber"

    # 옵티마이저
    optimizer = "adamw"         # "adam", "adamw", "sgd"
    lr = 1e-3
    weight_decay = 1e-4

    # 스케줄러
    scheduler = "cosine"        # None, "cosine", "step"
    scheduler_T0 = 5
    scheduler_Tmult = 2

    # 그래디언트 클리핑
    clip_grad = 0.5

    # 학습 설정
    epochs = 100
    patience = 15

    # 예측 설정
    overlap = 0.5               # 윈도우 중첩률
    fill_tail = True
```

#### Step 2: Strategy 스크립트 생성

`strategies/strategy_F_advanced.py`:

```python
#!/usr/bin/env python3
"""
Strategy F: Advanced 전략

고급 학습 전략입니다.
- 증강: CutMix (alpha=0.4)
- 손실함수: Huber Loss
- 옵티마이저: AdamW (weight_decay=1e-4)
- 스케줄러: Cosine Annealing
- Early Stopping: patience=15

Usage:
    python strategies/strategy_F_advanced.py --config configs/mamba.toon
    python strategies/strategy_F_advanced.py --config configs/lstm.toon --model lstm
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runner import run_strategy


def main():
	    parser = argparse.ArgumentParser(description="Run advanced strategy (Strategy F)")
    parser.add_argument("--config", type=Path, default=Path("configs/mamba.toon"))
    parser.add_argument("--model", type=str, default=None, help="모델 키 (config 오버라이드)")
    args = parser.parse_args()

    run_strategy(
        strategy_name="advanced",    # ← 전략 이름
        model_key=args.model,
        config_path=args.config,
        verbose=True,
    )


if __name__ == "__main__":
    main()
```

#### Step 3: 실행

```bash
python strategies/strategy_F_advanced.py --config configs/mamba.toon
```

#### 전략 설정 옵션

| 속성 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `augment` | str | `None` | 증강 방식 (`None`, `"cutmix"`, `"mixup"`, `"all"`) |
| `augment_alpha` | float | `0.2` | Beta 분포 파라미터 |
| `loss` | str | `"mse"` | 손실함수 (`"mse"`, `"smoothl1"`, `"huber"`, `"physmamba_sssd"`) |
| `optimizer` | str | `"adam"` | 옵티마이저 (`"adam"`, `"adamw"`, `"sgd"`) |
| `lr` | float | `1e-4` | 학습률 |
| `weight_decay` | float | `0.0` | L2 정규화 계수 |
| `scheduler` | str | `None` | 스케줄러 (`None`, `"cosine"`, `"step"`) |
| `scheduler_T0` | int | `10` | Cosine 스케줄러 첫 주기 |
| `scheduler_Tmult` | int | `2` | Cosine 스케줄러 주기 배수 |
| `clip_grad` | float | `None` | 그래디언트 클리핑 임계값 |
| `epochs` | int | `100` | 최대 에폭 수 |
| `patience` | int | `10` | Early Stopping patience |
| `overlap` | float | `0.0` | 예측 윈도우 중첩률 |
| `fill_tail` | bool | `True` | 시퀀스 끝 패딩 여부 |

---

### 3. 새 데이터셋 추가

> **파일 1개만 생성하면 끝!**

#### Step 1: 데이터셋 파일 생성

`datasets/custom_dataset.py`:

```python
"""
Custom Dataset
"""
from src.registry import register_dataset


@register_dataset(
	    name="customdataset",          # 데이터셋 키 (필수)
	    display_name="CustomDataset",  # 표시 이름 (기본: key.upper())
	    fs=25.0,                       # 샘플링 주파수
	    label_filename="CustomDataset_label.npy",           # 라벨 파일명
	    prediction_dirname="CustomDataset_prediction",       # 예측 디렉토리명
)
class CustomDataset:
    """
	    Custom Dataset

    - 총 50명 피험자
    - 각 피험자당 다양한 샘플 수
    - 샘플링 주파수: 25Hz
    """

    # 피험자별 샘플 수 (필수)
    SUBJECT_COUNTS = [
        # Subject 1~10
        120, 115, 130, 125, 118,
        122, 128, 135, 140, 125,
        # Subject 11~20
        130, 125, 120, 135, 128,
        122, 118, 140, 132, 126,
        # Subject 21~30
        128, 130, 125, 120, 135,
        140, 132, 128, 125, 130,
        # Subject 31~40
        125, 128, 132, 135, 140,
        125, 130, 128, 135, 132,
        # Subject 41~50
        130, 125, 128, 135, 140,
        132, 128, 125, 130, 135,
    ]
```

#### Step 2: 데이터 파일 배치

```
data/
├── CustomDataset_label.npy              # 라벨 (PPG 신호)
└── CustomDataset_prediction/
    └── CustomDataset_POS_prediction.npy # rPPG 예측값
```

#### Step 3: 설정 파일에서 사용

`configs/customdataset.toon`:

```toon
model: lstm

data:
  pred_path: data/CustomDataset_prediction/CustomDataset_POS_prediction.npy
  label_path: data/CustomDataset_label.npy

train:
  seq_len: 250       # 25Hz × 10초 = 250 샘플
  batch_size: 32
```

#### Step 4: 코드에서 접근

```python
from src.registry import get_dataset_info, list_datasets

# 등록된 데이터셋 확인
print(list_datasets())  # ['pure', 'ubfc', 'cohface', 'customdataset']

# 데이터셋 정보 조회
info = get_dataset_info("customdataset")
print(f"Name: {info.display_name}")
print(f"Sampling Rate: {info.fs}Hz")
print(f"Total Subjects: {info.total_subjects}")
print(f"Total Samples: {info.total_samples}")
```

#### 데이터셋 데코레이터 옵션

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `name` | str | (필수) | 데이터셋 키 |
| `display_name` | str | `name.upper()` | 표시 이름 |
| `fs` | float | `30.0` | 샘플링 주파수 (Hz) |
| `label_filename` | str | `{DISPLAY}_label.npy` | 라벨 파일명 |
| `prediction_dirname` | str | `{DISPLAY}_prediction` | 예측 디렉토리명 |

#### SUBJECT_COUNTS 형식

- 피험자 순서대로 각 피험자의 샘플(윈도우) 수
- LOSO (Leave-One-Subject-Out) 교차검증에 사용됨

---

## 아키텍처

### 레지스트리 시스템

```
┌─────────────────────────────────────────────────────────────┐
│                     src/registry.py                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   MODEL     │  │  STRATEGY   │  │   DATASET   │          │
│  │  REGISTRY   │  │  REGISTRY   │  │  REGISTRY   │          │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
│         │                │                │                 │
│  @register_model  @register_strategy  @register_dataset     │
└─────────┼────────────────┼────────────────┼─────────────────┘
          │                │                │
          ▼                ▼                ▼
    ┌─────────┐      ┌─────────┐      ┌─────────┐
    │ models/ │      │   src/  │      │datasets/│
    │ *.py    │      │strategy │      │  *.py   │
    └─────────┘      │  .py    │      └─────────┘
                     └─────────┘
```

### 학습 파이프라인

```
┌──────────────────────────────────────────────────────────────┐
│                  strategies/strategy_*.py                    │
│                            │                                 │
│                   run_strategy(name, ...)                    │
└────────────────────────────┼─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                      src/runner.py                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │ get_model() │  │get_strategy │  │ get_dataset │           │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘           │
│         │                │                │                  │
│         └────────────────┼────────────────┘                  │
│                          ▼                                   │
│              ┌─────────────────────┐                         │
│              │  build_from_strategy │ ← src/factory.py       │
│              └──────────┬──────────┘                         │
│                         │                                    │
│         ┌───────────────┼───────────────┐                    │
│         ▼               ▼               ▼                    │
│    criterion       optimizer       scheduler                 │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                      src/train.py                            │
│                                                              │
│                    fit(trainer, ...)                         │
│                          │                                   │
│              ┌───────────┴───────────┐                       │
│              ▼                       ▼                       │
│       Early Stopping            Callbacks                    │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                     src/trainer.py                           │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │ train_epoch │  │  val_epoch  │  │     AMP     │           │
│  └─────────────┘  └─────────────┘  └─────────────┘           │
└──────────────────────────────────────────────────────────────┘
```

### 컴포넌트 역할 분리

| 모듈 | 역할 | 레벨 |
|------|------|------|
| `strategies/` | 전략 실행 진입점 | High |
| `runner.py` | 전체 파이프라인 조율 | High |
| `strategy.py` | 전략 설정 정의 | High |
| `factory.py` | 컴포넌트 생성 | Mid |
| `train.py` | 에폭 루프, Early Stopping | Mid |
| `trainer.py` | 배치 처리, AMP, 그래디언트 | Low |

---

## 참조

### 모델 목록

| 키 | 클래스 | 설명 | 의존성 |
|----|--------|------|--------|
| `lstm` | LSTMModel | 기본 LSTM | - |
| `bilstm` | BiLSTMModel | 양방향 LSTM | - |
| `rnn` | RNNModel | 기본 RNN | - |
| `transformer` | RadiantModel | ViT 기반 | - |
| `physdiff` | PhysDiffModel | Transformer + 신호 분해 | - |
| `mamba` | MambaModel | SSM 기반 | mamba-ssm |
| `physmamba_td` | PhysMamba_TD | BiMamba + SE (Temporal Diff) | mamba-ssm |
| `physmamba_sssd` | PhysMamba_SSSD | Dual-Pathway SSM | mamba-ssm |
| `rhythmmamba` | RhythmMambaModel | 리듬 특화 Mamba | mamba-ssm |

### 데이터셋 목록

| 키 | 표시명 | 샘플링 | 피험자 수 | 총 샘플 |
|----|--------|--------|-----------|---------|
| `pure` | PURE | 30Hz | 59명 | 2,142 |
| `ubfc` | UBFC | 30Hz | 40명 | 1,588 |
| `cohface` | cohface | 30Hz | 144명 | 4,320 |

### 플랫폼 지원

| 플랫폼 | GPU | mamba-ssm | 사용 가능 모델 |
|--------|-----|-----------|----------------|
| Linux | CUDA | ✅ | 전체 9개 |
| Windows | CUDA | ❌ | lstm, bilstm, rnn, transformer, physdiff |
| macOS (Intel) | CPU | ❌ | lstm, bilstm, rnn, transformer, physdiff |
| macOS (M1/M2) | MPS | ❌ | lstm, bilstm, rnn, transformer, physdiff |

### 환경변수

| 변수 | 설명 | 예시 |
|------|------|------|
| `DEVICE` | 학습 디바이스 | `cuda:0`, `mps`, `cpu` |
| `NUM_WORKERS` | DataLoader 워커 수 | `4`, `8` |
| `USE_AMP` | Mixed Precision | `true`, `false` |
| `AMP_DTYPE` | AMP 타입 | `fp16`, `bf16` |

---

## 라이선스

MIT License
