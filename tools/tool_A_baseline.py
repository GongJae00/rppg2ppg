#!/usr/bin/env python3
"""
Tool A: Baseline 전략

가장 기본적인 학습 전략입니다.
- 증강: 없음
- 손실함수: MSE
- 옵티마이저: Adam
- 스케줄러: 없음
- Early Stopping: 없음

Usage:
    python tools/tool_A_baseline.py --config configs/mamba.toon
    python tools/tool_A_baseline.py --config configs/lstm.toon --model lstm
"""
from __future__ import annotations

import argparse
from pathlib import Path
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
from utils import plot_losses, psnr, compute_pearson


# ===========================================================================
# 전략 설정 - Baseline
# ===========================================================================
STRATEGY_NAME = "baseline"

STRATEGY = {
    "augment": None,           # 증강 없음
    "loss": "mse",             # Mean Squared Error
    "optimizer": "adam",       # Adam optimizer
    "lr": 1e-4,                # 학습률
    "weight_decay": 0.0,       # 가중치 감쇠 없음
    "scheduler": None,         # 스케줄러 없음
    "clip_grad": None,         # 그래디언트 클리핑 없음
    "epochs": 100,             # 에폭 수
    "patience": None,          # Early stopping 없음
    "overlap": 0.0,            # 예측 시 윈도우 중첩 없음
    "fill_tail": False,        # 마지막 부분 채우기 없음
}
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(description=f"Run {STRATEGY_NAME} strategy")
    parser.add_argument("--config", type=Path, default=Path("configs/mamba.toon"))
    parser.add_argument("--model", type=str, default=None, help="모델 키 (config 오버라이드)")
    args = parser.parse_args()

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
    num_workers = int(train_cfg.get("num_workers", 0))
    pin_memory = bool(train_cfg.get("pin_memory", False))
    epochs = int(train_cfg.get("epochs", STRATEGY["epochs"]))
    split_ratio = float(train_cfg.get("split", 0.8))

    # 데이터 로드
    raw_pred, raw_label, x_all, y_all, sx, sy = load_and_scale(pred_path, label_path)
    train_x, test_x, train_y, test_y = split_train_test(x_all, y_all, split=split_ratio)
    train_loader, test_loader = make_loaders(
        train_x, train_y, test_x, test_y,
        seq_len=seq_len, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # 모델 및 학습 구성
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_key, seq_len).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=STRATEGY["lr"])

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=None,
        clip_grad=STRATEGY["clip_grad"],
    )

    # 학습 루프
    train_losses, val_losses = [], []
    print(f"\n[{model_key}:{STRATEGY_NAME}] Training started on {device}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, Seq: {seq_len}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_epoch(train_loader, augment_fn=None)
        val_loss = trainer.validate(test_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"  Epoch {epoch:3d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

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
