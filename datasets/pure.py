"""
PURE 데이터셋 정의
"""
from src.registry import register_dataset


@register_dataset(name="pure", display_name="PURE", fs=30.0)
class PUREDatasetMeta:
    """
    PURE 데이터셋 메타정보

    - 59명의 피험자
    - 각 피험자당 6~8개의 세그먼트
    - 총 390개의 샘플
    """

    SUBJECT_COUNTS = [
        6, 6, 8, 8, 6, 6, 6, 6, 7, 7, 6, 6, 6, 6, 7, 7, 6, 6, 6, 6,
        7, 7, 6, 6, 6, 6, 7, 7, 6, 6, 6, 7, 7, 6, 6, 6, 7, 7, 7, 6, 6,
        6, 6, 7, 7, 6, 6, 6, 6, 7, 8, 6, 6, 6, 6, 7, 7, 6, 6,
    ]
