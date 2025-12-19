"""
runner.py - 통합 실행 함수

모든 전략의 통합 진입점을 제공합니다.
strategies/strategy_*.py는 이 모듈의 run_strategy()를 호출하여 코드 중복을 제거합니다.

사용법:
    from src.runner import run_strategy

    # 전략 이름으로 실행
    run_strategy("baseline", model_key="lstm", config_path="configs/lstm.toon")

    # 전략 코드로 실행
    run_strategy_by_code("A", model_key="lstm", config_path="configs/lstm.toon")
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np

import torch

from src.registry import get_model, StrategyConfig
from src.strategy import get_strategy, get_strategy_by_code
from src.factory import build_from_strategy
from src.trainer import Trainer
from src.train import fit, TrainResult
from src.predictor import Predictor

from utils.data import load_and_scale, split_train_test, make_loaders
from utils.toon import load_toon
from utils.env_config import get_env_config, get_device, get_dataloader_kwargs
from utils.early_stopping import EarlyStopping
from utils.output_paths import build_output_paths
from utils.plot import plot_losses
from utils.metrics import compute_pearson, psnr


# ===========================================================================
# 통합 실행 함수
# ===========================================================================


def run_strategy(
    strategy_name: str,
    model_key: Optional[str] = None,
    config_path: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """
    전략 실행의 통합 진입점

    모든 전략(A~E)이 이 함수를 재사용하여 코드 중복을 제거합니다.

    Args:
        strategy_name: 전략 이름 (baseline, improved, cutmix, mixup, aug_all)
        model_key: 모델 키 (lstm, mamba 등). None이면 config에서 읽음
        config_path: 설정 파일 경로. None이면 기본값 사용
        verbose: 진행 상황 출력 여부

    Returns:
        {
            "result": TrainResult,
            "predictions": np.ndarray,
            "metrics": {"pearson": float, "psnr": float},
            "paths": dict,
        }
    """
    # 1. 환경 설정 로드
    env_config = get_env_config()
    device = get_device()
    loader_kwargs = get_dataloader_kwargs()

    # 2. 전략 설정 로드
    strategy = get_strategy(strategy_name)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy.name} (Code: {strategy.code})")
        print(f"{'='*60}")

    # 3. Config 파일 로드
    config_path = config_path or Path("configs/mamba.toon")
    config = load_toon(config_path)

    # 모델 키 결정
    model_key = model_key or config.get("model", "mamba")

    if verbose:
        print(f"Model: {model_key}")
        print(f"Config: {config_path}")

    # 3.1 전략 오버라이드 (config 기준, 선택)
    strategy_override = config.get("strategy_override") or config.get("strategy_overrides")
    if strategy_override:
        data = vars(strategy).copy()
        for key, value in strategy_override.items():
            if key in data:
                data[key] = value
            elif verbose:
                print(f"[runner] Unknown strategy override key: {key}")
        strategy = StrategyConfig(**data)

    # 4. 데이터 경로 설정
    data_cfg = config.get("data", {})
    pred_path = Path(data_cfg.get("pred_path", "data/PURE_prediction/PURE_POS_prediction.npy"))
    label_path = Path(data_cfg.get("label_path", "data/PURE_label.npy"))

    # 5. 하이퍼파라미터
    train_cfg = config.get("train", {})
    seq_len = int(train_cfg.get("seq_len", 300))
    batch_size = int(train_cfg.get("batch_size", 32))
    split_ratio = float(train_cfg.get("split", 0.8))

    # 6. 출력 경로 설정
    paths = build_output_paths(model_key, strategy.name, strategy_code=strategy.code)

    if verbose:
        print(f"Output: {paths['base']}")
        print(f"Epochs: {strategy.epochs}, Patience: {strategy.patience}")
        print()

    # 7. 데이터 로드 및 전처리
    raw_pred, raw_label, x_all, y_all, sx, sy = load_and_scale(pred_path, label_path)
    train_x, test_x, train_y, test_y = split_train_test(x_all, y_all, split=split_ratio)
    train_loader, test_loader = make_loaders(
        train_x, train_y, test_x, test_y,
        seq_len=seq_len,
        batch_size=batch_size,
        **loader_kwargs,
    )

    if verbose:
        print(f"Data: {len(raw_pred):,} samples | Train: {len(train_x):,} | Test: {len(test_x):,}")

    # 8. 모델 생성
    model = get_model(model_key, seq_len=seq_len).to(device)

    if verbose:
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model params: {param_count:,}")

    # 9. 컴포넌트 빌드
    components = build_from_strategy(strategy, model, device)

    # 10. Trainer 생성
    trainer = Trainer(
        model=model,
        criterion=components["criterion"],
        optimizer=components["optimizer"],
        device=device,
        scheduler=components["scheduler"],
        clip_grad=strategy.clip_grad,
        use_amp=env_config.use_amp,
        amp_dtype=env_config.amp_dtype,
    )

    # 11. Early Stopping 설정
    early_stopping = EarlyStopping(
        patience=strategy.patience,
        save_path=paths["best_ckpt"],
        verbose=verbose,
    )

    # 12. 학습 실행
    if verbose:
        print(f"\nTraining started...")

    result = fit(
        trainer=trainer,
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=strategy.epochs,
        augment_fn=components["augment_fn"],
        augment_fns=components["augment_fns"],
        early_stopping=early_stopping,
        verbose=verbose,
    )

    # 13. 최고 모델 로드
    early_stopping.load_best(model)

    # 14. 손실 그래프 저장
    plot_losses(paths["loss_plot"], result.train_losses, result.val_losses)

    # 15. 체크포인트 저장
    trainer.save_checkpoint(paths["ckpt"])

    # 16. 예측 및 후처리
    predictor = Predictor(model, device)
    preds_norm = predictor.predict(
        x_all,
        seq_len=seq_len,
        overlap=strategy.overlap,
        fill_tail=strategy.fill_tail,
    )
    preds_final = predictor.postprocess(preds_norm, sy, len(raw_label))

    # 17. 평가
    corr = compute_pearson(raw_label[: len(preds_final)], preds_final)
    snr = psnr(raw_label[: len(preds_final)], preds_final)

    if verbose:
        print(f"\nResults:")
        print(f"  Pearson: {corr:.4f}")
        print(f"  PSNR: {snr:.2f} dB")
        print(f"  Best epoch: {result.best_epoch}")
        print(f"  Stopped early: {result.stopped_early}")

    # 18. 예측 결과 저장
    np.save(paths["prediction"], preds_final.astype(np.float32))

    if verbose:
        print(f"\nSaved:")
        print(f"  Checkpoint: {paths['ckpt']}")
        print(f"  Loss plot: {paths['loss_plot']}")
        print(f"  Prediction: {paths['prediction']}")

    return {
        "result": result,
        "predictions": preds_final,
        "metrics": {"pearson": corr, "psnr": snr},
        "paths": paths,
    }


def run_strategy_by_code(
    code: str,
    model_key: Optional[str] = None,
    config_path: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """
    전략 코드(A, B, C, ...)로 실행

    Args:
        code: 전략 코드 (A, B, C, D, E)
        model_key: 모델 키
        config_path: 설정 파일 경로
        verbose: 진행 상황 출력 여부

    Returns:
        run_strategy()와 동일
    """
    strategy = get_strategy_by_code(code)
    return run_strategy(
        strategy_name=strategy.name,
        model_key=model_key,
        config_path=config_path,
        verbose=verbose,
    )


# ===========================================================================
# Export
# ===========================================================================

__all__ = [
    "run_strategy",
    "run_strategy_by_code",
]
