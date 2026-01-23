# core/annotation_parts/params.py
from dataclasses import dataclass


@dataclass
class AnnotationParams:
    voxel: float = 0.003
    k: int = 25
    smooth_win: int = 9  # 仍保留但叶长不再平滑
    label_radius: float = 0.01
    graph_radius: float = 0.02

    # 旧“最大叶宽”估计参数（用于推荐）
    step: float = 0.01
    slab_half: float = 0.01
    radius: float = 0.01
    min_slice_pts: int = 60
    stem_diameter_segments: int = 0
    stem_length_segments: int = 0
    stem_segments: int = 0  # 旧字段，兼容单一分段数
    stem_step: float = 0.01  # 旧字段，兼容 step
    stem_diameter_percentile: float = 95.0
    stem_length_percentile: float = 95.0


