"""
UBFC 데이터셋 정의
"""
from src.registry import register_dataset


@register_dataset(name="ubfc", display_name="UBFC", fs=30.0)
class UBFCDatasetMeta:
    """
    UBFC 데이터셋 메타정보

    - 40명의 피험자
    - 각 피험자당 4~6개의 세그먼트
    - 총 228개의 샘플
    """

    SUBJECT_COUNTS = [
        6, 6, 6, 6, 6, 6, 6, 6, 5, 6, 6, 6, 6, 4, 4, 4, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 4, 6, 6,
    ]
