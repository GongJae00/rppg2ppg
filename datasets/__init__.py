"""
datasets/ - 데이터셋 플러그인

데코레이터 기반 자동 등록 시스템을 사용합니다.
새 데이터셋을 추가하려면 이 디렉토리에 파일을 추가하면 됩니다.

새 데이터셋 추가 방법:
    # datasets/custom_dataset.py
    from src.registry import register_dataset

    @register_dataset(name="customdataset", display_name="CustomDataset", fs=30.0)
    class CustomDatasetMeta:
        SUBJECT_COUNTS = [10, 10, 10, ...]

사용법:
    from src.registry import get_dataset_info, list_datasets

    info = get_dataset_info("pure")
    print(info.subject_counts)
    print(info.get_label_path("data"))
"""
import importlib
from pathlib import Path


# 디렉토리 내 모든 데이터셋 모듈 자동 import
_current_dir = Path(__file__).parent

for _f in _current_dir.glob("*.py"):
    if _f.stem.startswith("_"):
        continue
    try:
        importlib.import_module(f".{_f.stem}", __package__)
    except ImportError as e:
        import warnings
        warnings.warn(f"Failed to import dataset module '{_f.stem}': {e}")
