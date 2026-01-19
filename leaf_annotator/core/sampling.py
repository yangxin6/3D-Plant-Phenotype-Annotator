import numpy as np
from dataclasses import dataclass


@dataclass
class Downsampled:
    points: np.ndarray
    src_indices: np.ndarray  # indices into original (leaf_pts) rows


class VoxelSampler:
    @staticmethod
    def voxel_downsample_with_index(points: np.ndarray, voxel: float) -> Downsampled:
        idx = np.arange(len(points), dtype=np.int64)
        return Downsampled(points=points, src_indices=idx)
