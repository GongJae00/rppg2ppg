"""
공통 결과 경로 유틸리티

모델/전략별 결과를 일관된 디렉토리 구조로 생성합니다.
"""
from __future__ import annotations

from pathlib import Path


def build_output_paths(
    model_key: str,
    strategy_name: str,
    strategy_code: str | None = None,
    root: Path | str = "results",
) -> dict[str, Path]:
    """
    결과 저장 경로를 생성하고 반환합니다.

    Structure:
        results/{model_key}/
            checkpoints/
                {model_key}_{strategy_code}_{strategy_name}.pth
                {model_key}_{strategy_code}_{strategy_name}_best.pth
            plots/
                {model_key}_{strategy_code}_{strategy_name}_loss.png
            predictions/
                {model_key}_{strategy_code}_{strategy_name}_prediction.npy
    """
    base = Path(root) / model_key
    checkpoints = base / "checkpoints"
    plots = base / "plots"
    predictions = base / "predictions"

    for path in (checkpoints, plots, predictions):
        path.mkdir(parents=True, exist_ok=True)

    tag_parts = [model_key]
    if strategy_code:
        tag_parts.append(str(strategy_code))
    tag_parts.append(strategy_name)
    base_tag = "_".join(tag_parts)

    return {
        "base": base,
        "checkpoints": checkpoints,
        "plots": plots,
        "predictions": predictions,
        "ckpt": checkpoints / f"{base_tag}.pth",
        "best_ckpt": checkpoints / f"{base_tag}_best.pth",
        "loss_plot": plots / f"{base_tag}_loss.png",
        "prediction": predictions / f"{base_tag}_prediction.npy",
    }
