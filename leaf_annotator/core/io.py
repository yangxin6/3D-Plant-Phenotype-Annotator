import os
import numpy as np
import pyvista as pv


class PointCloudIO:
    """
    推荐：整株点云用 txt/csv/xyz/npy（NxD，多列）
      xyz rgb? sem inst
    其中 xyz 固定 :3；rgb 可能不存在；sem/inst 默认最后两列。
    """
    SUPPORTED = {".ply", ".vtp", ".vtk", ".obj", ".stl", ".pcd", ".xyz", ".txt", ".csv", ".npy"}

    @staticmethod
    def load_array(path: str) -> np.ndarray:
        ext = os.path.splitext(path)[1].lower()
        if ext not in PointCloudIO.SUPPORTED:
            raise ValueError(f"不支持的格式: {ext}")

        if ext == ".npy":
            return np.asarray(np.load(path))

        if ext in [".xyz", ".txt", ".csv"]:
            arr = np.loadtxt(path, delimiter="," if ext == ".csv" else None)
            return np.asarray(arr)

        # ply/pcd/vtk 等：通常只含 xyz(/rgb)，不含 sem/inst
        mesh = pv.read(path)
        return np.asarray(mesh.points)
