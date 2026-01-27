# core/annotation_parts/plant.py
from typing import Optional, Dict, Any, Tuple

import numpy as np


class PlantMixin:
    def _normalize_vec(self, vec) -> Optional[np.ndarray]:
        if vec is None:
            return None
        v = np.asarray(vec, dtype=np.float64).reshape(3,)
        n = float(np.linalg.norm(v))
        if n <= 1e-12:
            return None
        return v / n


    def _principal_axis(self, pts: np.ndarray) -> Optional[np.ndarray]:
        if pts is None or len(pts) < 3:
            return None
        centered = pts - pts.mean(axis=0)
        cov = np.cov(centered, rowvar=False)
        if cov.shape != (3, 3):
            return None
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, int(np.argmax(eigvals))]
        return self._normalize_vec(axis)


    def _build_growth_basis(self, direction: np.ndarray) -> Optional[np.ndarray]:
        z = self._normalize_vec(direction)
        if z is None:
            return None
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(ref, z))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x = np.cross(ref, z)
        x = self._normalize_vec(x)
        if x is None:
            return None
        y = np.cross(z, x)
        y = self._normalize_vec(y)
        if y is None:
            return None
        return np.column_stack([x, y, z])


    def has_growth_direction(self) -> bool:
        return self.growth_direction is not None


    def has_plant_measurements(self) -> bool:
        return isinstance(self.plant_measurements, dict) and len(self.plant_measurements) > 0


    def has_plant_data(self) -> bool:
        return self.has_growth_direction() or self.has_plant_measurements()


    def set_growth_direction(
        self,
        origin: Optional[np.ndarray],
        direction: Optional[np.ndarray],
        method: Optional[str] = None,
        basis: Optional[np.ndarray] = None,
    ):
        self.growth_origin = None if origin is None else np.asarray(origin, dtype=np.float64).reshape(3,)
        self.growth_direction = self._normalize_vec(direction)
        if basis is None and self.growth_direction is not None:
            basis = self._build_growth_basis(self.growth_direction)
        self.growth_basis = None if basis is None else np.asarray(basis, dtype=np.float64)
        self.growth_method = None if method is None else str(method)
        self.plant_measurements = {}


    def _rotation_matrix_xyz(self, rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
        rx = np.deg2rad(rx_deg)
        ry = np.deg2rad(ry_deg)
        rz = np.deg2rad(rz_deg)
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        Rx = np.array([[1.0, 0.0, 0.0],
                       [0.0, cx, -sx],
                       [0.0, sx, cx]], dtype=np.float64)
        Ry = np.array([[cy, 0.0, sy],
                       [0.0, 1.0, 0.0],
                       [-sy, 0.0, cy]], dtype=np.float64)
        Rz = np.array([[cz, -sz, 0.0],
                       [sz, cz, 0.0],
                       [0.0, 0.0, 1.0]], dtype=np.float64)
        return Rz @ Ry @ Rx


    def rotate_growth_direction(self, rx_deg: float, ry_deg: float, rz_deg: float):
        if self.growth_direction is None:
            raise RuntimeError("请先设置生长方向。")
        if self.growth_basis is None:
            self.growth_basis = self._build_growth_basis(self.growth_direction)
        if self.growth_basis is None:
            raise RuntimeError("无法构建生长方向坐标系。")
        R = self._rotation_matrix_xyz(rx_deg, ry_deg, rz_deg)
        self.growth_basis = R @ self.growth_basis
        self.growth_direction = self._normalize_vec(self.growth_basis[:, 2])
        self.plant_measurements = {}


    def compute_growth_direction_from_stem(self, lower_ratio: float = 0.5) -> Optional[Dict[str, Any]]:
        self._require_cloud()
        stem_label = self.semantic_map.get("stem")
        if stem_label is None:
            raise RuntimeError("请先选择茎语义标签。")
        mask = (self.cloud.sem == int(stem_label))
        pts = self.cloud.xyz[mask]
        if pts is None or len(pts) < 3:
            return None
        axis = self._principal_axis(pts)
        if axis is None:
            return None
        proj = pts @ axis
        tmin = float(np.min(proj))
        tmax = float(np.max(proj))
        if tmax - tmin > 1e-9:
            cutoff = tmin + float(lower_ratio) * (tmax - tmin)
            lower_pts = pts[proj <= cutoff]
            if lower_pts is not None and len(lower_pts) >= 3:
                axis_lower = self._principal_axis(lower_pts)
                if axis_lower is not None:
                    if float(np.dot(axis_lower, axis)) < 0:
                        axis_lower = -axis_lower
                    axis = axis_lower
        proj = pts @ axis
        idx_min = int(np.argmin(proj))
        idx_max = int(np.argmax(proj))
        origin = np.asarray(pts[idx_min], dtype=np.float64)
        if float(np.dot(axis, pts[idx_max] - origin)) < 0:
            axis = -axis
        basis = self._build_growth_basis(axis)
        self.set_growth_direction(origin=origin, direction=axis, method="stem_lower", basis=basis)
        return {
            "origin": origin.tolist(),
            "direction": axis.tolist(),
            "method": "stem_lower",
        }


    def compute_plant_height_crown(self) -> Optional[Tuple[float, float]]:
        self._require_cloud()
        if self.growth_direction is None:
            raise RuntimeError("请先确定生长方向。")
        if self.growth_basis is None:
            self.growth_basis = self._build_growth_basis(self.growth_direction)
        if self.growth_basis is None:
            raise RuntimeError("无法构建生长方向坐标系。")

        z_axis = self._normalize_vec(self.growth_direction)
        if z_axis is None:
            return None
        pts = self.cloud.xyz
        if pts is None or len(pts) == 0:
            return None

        origin = self.growth_origin
        if origin is None:
            proj = pts @ z_axis
            origin = np.asarray(pts[int(np.argmin(proj))], dtype=np.float64)
            self.growth_origin = origin

        local = (pts - origin) @ self.growth_basis
        xmin = float(np.min(local[:, 0]))
        xmax = float(np.max(local[:, 0]))
        ymin = float(np.min(local[:, 1]))
        ymax = float(np.max(local[:, 1]))
        zmin = float(np.min(local[:, 2]))
        zmax = float(np.max(local[:, 2]))
        height = max(0.0, zmax - zmin)
        crown = float(np.hypot(xmax - xmin, ymax - ymin))

        self.plant_measurements = {
            "height": float(height),
            "crown_width": float(crown),
            "extents": {
                "x_min": float(xmin),
                "x_max": float(xmax),
                "y_min": float(ymin),
                "y_max": float(ymax),
                "z_min": float(zmin),
                "z_max": float(zmax),
            },
            "origin": origin.tolist(),
            "basis": self.growth_basis.tolist(),
        }
        return float(height), float(crown)


    def get_growth_info_dict(self) -> Optional[Dict[str, Any]]:
        if self.growth_direction is None and self.growth_origin is None:
            return None
        return {
            "origin": None if self.growth_origin is None else self.growth_origin.tolist(),
            "direction": None if self.growth_direction is None else self.growth_direction.tolist(),
            "basis": None if self.growth_basis is None else self.growth_basis.tolist(),
            "method": self.growth_method,
        }


    def get_plant_measurements_dict(self) -> Optional[Dict[str, Any]]:
        if not self.has_plant_measurements():
            return None
        return dict(self.plant_measurements)


    def load_growth_info_dict(self, data: Dict[str, Any]):
        if not isinstance(data, dict):
            return
        origin = data.get("origin")
        direction = data.get("direction")
        basis = data.get("basis")
        method = data.get("method")
        if origin is not None:
            try:
                origin = np.asarray(origin, dtype=np.float64).reshape(3,)
            except Exception:
                origin = None
        if direction is not None:
            try:
                direction = np.asarray(direction, dtype=np.float64).reshape(3,)
            except Exception:
                direction = None
        if basis is not None:
            try:
                basis = np.asarray(basis, dtype=np.float64).reshape(3, 3)
            except Exception:
                basis = None
        self.set_growth_direction(origin=origin, direction=direction, method=method, basis=basis)


    def load_plant_measurements_dict(self, data: Dict[str, Any]):
        if not isinstance(data, dict):
            return
        height = data.get("height")
        crown = data.get("crown_width")
        extents = data.get("extents")
        origin = data.get("origin")
        basis = data.get("basis")
        out: Dict[str, Any] = {}
        if height is not None:
            try:
                out["height"] = float(height)
            except Exception:
                pass
        if crown is not None:
            try:
                out["crown_width"] = float(crown)
            except Exception:
                pass
        if isinstance(extents, dict):
            out["extents"] = dict(extents)
        if origin is not None:
            try:
                out["origin"] = np.asarray(origin, dtype=np.float64).reshape(3,).tolist()
            except Exception:
                pass
        if basis is not None:
            try:
                out["basis"] = np.asarray(basis, dtype=np.float64).reshape(3, 3).tolist()
            except Exception:
                pass
        self.plant_measurements = out
