import os
import numpy as np
import pyvista as pv


class PointCloudIO:
    SUPPORTED = {".ply", ".vtp", ".vtk", ".obj", ".stl", ".pcd", ".xyz", ".txt", ".csv"}

    @staticmethod
    def load_points(path: str) -> np.ndarray:
        ext = os.path.splitext(path)[1].lower()
        if ext not in PointCloudIO.SUPPORTED:
            raise ValueError(f"不支持的格式: {ext}")

        if ext in [".ply", ".vtp", ".vtk", ".obj", ".stl", ".pcd"]:
            mesh = pv.read(path)
            pts = np.asarray(mesh.points)
            if pts.ndim != 2 or pts.shape[1] != 3:
                raise ValueError("读取到的点不是 Nx3。")
            return pts.astype(np.float64)

        # 文本
        pts = np.loadtxt(path, delimiter="," if ext == ".csv" else None)
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim == 2 and pts.shape[1] >= 3:
            return pts[:, :3].copy()
        raise ValueError("文本点云需要至少三列 x y z。")
