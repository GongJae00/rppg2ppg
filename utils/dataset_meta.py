"""
데이터셋 메타 정보

평가/학습에서 데이터셋별 고정 정보를 한 곳에서 관리합니다.
main_eval(origin)에서 사용하던 subject_counts, 기본 fs, 라벨/입력 경로를
현행 rppg2ppg 디렉토리 구조(data/)에 맞춰 제공합니다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class DatasetMeta:
    key: str
    label_path: Path
    prediction_dir: Path
    subject_counts: List[int]
    fs: float = 30.0


_DATASET_KEYMAP: Dict[str, str] = {
    "pure": "PURE",
    "ubfc": "UBFC",
    "cohface": "cohface",
}


_SUBJECT_COUNTS: Dict[str, List[int]] = {
    "ubfc": [
        6, 6, 6, 6, 6, 6, 6, 6, 5, 6, 6, 6, 6, 4, 4, 4, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 4, 6, 6,
    ],
    "pure": [
        6, 6, 8, 8, 6, 6, 6, 6, 7, 7, 6, 6, 6, 6, 7, 7, 6, 6, 6, 6,
        7, 7, 6, 6, 6, 6, 7, 7, 6, 6, 6, 7, 7, 6, 6, 6, 7, 7, 7, 6, 6,
        6, 6, 7, 7, 6, 6, 6, 6, 7, 8, 6, 6, 6, 6, 7, 7, 6, 6,
    ],
    "cohface": [
        4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    ],
}


def canonicalize_dataset(name: str) -> str:
    key = name.strip().lower()
    if key not in _DATASET_KEYMAP:
        raise ValueError(f"Unknown dataset: {name}")
    return key


def get_dataset_meta(dataset: str, root: Path | str = "data") -> DatasetMeta:
    """
    데이터셋 메타 반환.

    Args:
        dataset: PURE/UBFC/cohface 등
        root: data 디렉토리 루트
    """
    key = canonicalize_dataset(dataset)
    root_path = Path(root)
    ds_tag = _DATASET_KEYMAP[key]
    return DatasetMeta(
        key=key,
        label_path=root_path / f"{ds_tag}_label.npy",
        prediction_dir=root_path / f"{ds_tag}_prediction",
        subject_counts=_SUBJECT_COUNTS[key],
        fs=30.0,
    )


def list_datasets() -> List[str]:
    return list(_DATASET_KEYMAP.keys())

