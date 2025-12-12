#!/usr/bin/env python3
"""
Tool D: MixUp 전략

MixUp 데이터 증강을 적용한 학습 전략입니다.
- 증강: MixUp (샘플 간 선형 보간)
- 손실함수: SmoothL1 (Huber Loss)
- 옵티마이저: AdamW
- 스케줄러: Cosine Annealing Warm Restarts
- Early Stopping: patience=8

Usage:
    python tools/tool_D_mixup.py --config configs/mamba.toon
    python tools/tool_D_mixup.py --config configs/lstm.toon --model lstm
"""
from __future__ import annotations

import argparse
from pathlib import Path
from functools import partial
import sys

# Ensure project root is in sys.path so local package imports work when run from tools/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src import get_model, Trainer, Predictor
from utils import load_and_scale, split_train_test, make_loaders, load_toon
from utils import apply_mixup, plot_losses, psnr, compute_pearson
from utils import get_env_config, get_device, get_dataloader_kwargs


# ===========================================================================
# 전략 설정 - MixUp
# ===========================================================================
STRATEGY_NAME = "mixup"

STRATEGY = {
    "augment": "mixup",        # MixUp 증강
    "augment_alpha": 0.2,      # 베타 분포 파라미터
    "loss": "smoothl1",        # Smooth L1 Loss
    "optimizer": "adamw",      # AdamW optimizer
    "lr": 3e-4,                # 학습률
    "weight_decay": 1e-5,      # 가중치 감쇠
    "scheduler": "cosine",     # Cosine Annealing
    "scheduler_T0": 10,        # 첫 번째 재시작 주기
    "scheduler_Tmult": 2,      # 재시작 주기 배수
    "clip_grad": 1.0,          # 그래디언트 클리핑
    "epochs": 65,              # 에폭 수
    "patience": 8,             # Early stopping patience
    "overlap": 0.5,            # 예측 시 50% 중첩
    "fill_tail": True,         # 마지막 부분 채우기
}
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(description=f"Run {STRATEGY_NAME} strategy")
    parser.add_argument("--config", type=Path, default=Path("configs/mamba.toon"))
    parser.add_argument("--model", type=str, default=None, help="모델 키 (config 오버라이드)")
    args = parser.parse_args()

    # 환경 설정 로드 (auto_profile.py 환경변수 기반)
    env_config = get_env_config()
    device = get_device()
    loader_kwargs = get_dataloader_kwargs()

    # 설정 로드
    config = load_toon(args.config)
    model_key = args.model or config.get("model", "mamba")

    data_cfg = config.get("data", {})
    pred_path = Path(data_cfg.get("pred_path", "real_data/PURE_POS_prediction.npy"))
    label_path = Path(data_cfg.get("label_path", "real_data/PURE_label.npy"))

    # 결과 저장 경로: results/{model_key}/
    output_dir = Path("results") / model_key
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = config.get("train", {})
    seq_len = int(train_cfg.get("seq_len", 300))
    batch_size = int(train_cfg.get("batch_size", 32))
    epochs = int(train_cfg.get("epochs", STRATEGY["epochs"]))
    split_ratio = float(train_cfg.get("split", 0.8))

    # 데이터 로드 (환경변수 기반 DataLoader 설정)
    raw_pred, raw_label, x_all, y_all, sx, sy = load_and_scale(pred_path, label_path)
    train_x, test_x, train_y, test_y = split_train_test(x_all, y_all, split=split_ratio)
    train_loader, test_loader = make_loaders(
        train_x, train_y, test_x, test_y,
        seq_len=seq_len, batch_size=batch_size,
        **loader_kwargs,
    )

    # 모델 및 학습 구성 (환경변수 기반 AMP)
    model = get_model(model_key, seq_len).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=STRATEGY["lr"],
        weight_decay=STRATEGY["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=STRATEGY["scheduler_T0"],
        T_mult=STRATEGY["scheduler_Tmult"],
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        clip_grad=STRATEGY["clip_grad"],
        use_amp=env_config.use_amp,
        amp_dtype=env_config.amp_dtype,
    )

    # 증강 함수
    augment_fn = partial(apply_mixup, alpha=STRATEGY["augment_alpha"])

    # 학습 루프 (Early Stopping 포함)
    train_losses, val_losses = [], []
    best_val = float("inf")
    wait = 0
    patience = STRATEGY["patience"]

    amp_info = f"AMP={env_config.use_amp}({env_config.amp_dtype})" if env_config.use_amp else "AMP=off"
    print(f"\n[{model_key}:{STRATEGY_NAME}] Training started on {device}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, Seq: {seq_len}, {amp_info}")
    print(f"  Augment: MixUp (alpha={STRATEGY['augment_alpha']})")
    print(f"  Early Stopping: patience={patience}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_epoch(train_loader, augment_fn=augment_fn)
        val_loss = trainer.validate(test_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"  Epoch {epoch:3d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        # Early Stopping 체크
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            trainer.save_checkpoint(output_dir / f"{model_key}_{STRATEGY_NAME}_best.pth")
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # 결과 저장
    plot_losses(output_dir / f"{model_key}_{STRATEGY_NAME}_loss.png", train_losses, val_losses)
    trainer.save_checkpoint(output_dir / f"{model_key}_{STRATEGY_NAME}.pth")

    # 예측 및 평가
    predictor = Predictor(model, device)
    preds_norm = predictor.predict(
        x_all, seq_len=seq_len,
        overlap=STRATEGY["overlap"], fill_tail=STRATEGY["fill_tail"],
    )
    preds_final = predictor.postprocess(preds_norm, sy, len(raw_label))

    corr = compute_pearson(raw_label[:len(preds_final)], preds_final)
    snr = psnr(raw_label[:len(preds_final)], preds_final)

    print("-" * 60)
    print(f"[{model_key}:{STRATEGY_NAME}] Final Pearson: {corr:.4f}, PSNR: {snr:.2f} dB")

    np.save(output_dir / f"{model_key}_{STRATEGY_NAME}_prediction.npy", preds_final.astype(np.float32))
    print(f"  Saved: {output_dir}/{model_key}_{STRATEGY_NAME}_*.{{pth,png,npy}}")


if __name__ == "__main__":
    main()
