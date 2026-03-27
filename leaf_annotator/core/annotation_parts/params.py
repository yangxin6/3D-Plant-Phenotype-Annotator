# core/annotation_parts/params.py
from dataclasses import dataclass


@dataclass
class AnnotationParams:
    voxel: float = 0.003
    k: int = 25
    smooth_win: int = 9  # kept for compatibility
    label_radius: float = 0.01
    graph_radius: float = 0.02

    # Width recommendation parameters
    step: float = 0.01
    slab_half: float = 0.01
    radius: float = 0.01
    min_slice_pts: int = 60

    # Leaf angle parameters (default: 1/5)
    leaf_inclination_ratio: float = 0.2
    leaf_inclination_radius: float = 0.01
    leaf_stem_ratio: float = 0.2

    stem_diameter_segments: int = 0
    stem_length_segments: int = 0
    stem_segments: int = 0  # legacy field
    stem_step: float = 0.01  # legacy field
    stem_diameter_percentile: float = 95.0
    stem_length_percentile: float = 95.0
    stem_denoise_knn_k: int = 8
    stem_denoise_knn_sigma: float = 2.5
    stem_denoise_radial_sigma: float = 2.5

