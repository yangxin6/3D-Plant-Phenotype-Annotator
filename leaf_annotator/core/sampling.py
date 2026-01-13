import numpy as np
from dataclasses import dataclass


@dataclass
class Downsampled:
    points: np.ndarray
    src_indices: np.ndarray  # indices into original (leaf_pts) rows


class VoxelSampler:
    @staticmethod
    def voxel_downsample_with_index(points: np.ndarray, voxel: float) -> Downsampled:
        if voxel <= 0:
            raise ValueError("voxel 必须 > 0")
        q = np.floor(points / voxel).astype(np.int64)
        _, idx = np.unique(q, axis=0, return_index=True)
        idx = np.asarray(idx, dtype=np.int64)
        return Downsampled(points=points[idx], src_indices=idx)
