import numpy as np


class VoxelSampler:
    @staticmethod
    def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
        if voxel <= 0:
            raise ValueError("voxel 必须 > 0")
        q = np.floor(points / voxel).astype(np.int64)
        _, idx = np.unique(q, axis=0, return_index=True)
        return points[idx]
