# core/annotation_parts/obb.py
from typing import Optional, Dict, Any

import numpy as np


class ObbMixin:
    def _compute_obb_from_points(self, pts: np.ndarray) -> Optional[Dict[str, Any]]:
        if pts is None or len(pts) < 3:
            return None
        mean = pts.mean(axis=0)
        centered = pts - mean
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        R = eigvecs[:, order]
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1
        local = centered @ R
        mins = local.min(axis=0)
        maxs = local.max(axis=0)
        lengths = (maxs - mins)
        center_local = (mins + maxs) / 2.0
        center_world = mean + center_local @ R.T
        return {
            "center": center_world.tolist(),
            "lengths": lengths.tolist(),
            "rotation": R.tolist(),
        }


    def compute_flower_fruit_obb(self) -> Dict[str, int]:
        self._require_cloud()
        sem_map = self._get_instance_sem_map()
        flower_label = self.semantic_map.get("flower")
        fruit_label = self.semantic_map.get("fruit")
        counts = {"flower": 0, "fruit": 0}
        for inst_id, sem_label in sem_map.items():
            pts = self.get_instance_points(inst_id)
            if flower_label is not None and int(sem_label) == int(flower_label):
                obb = self._compute_obb_from_points(pts)
                if obb is not None:
                    ann = self._ensure_annotation_entry(inst_id)
                    ann["flower_obb"] = obb
                    ann["flower_dims"] = self._obb_dims_from_lengths(obb.get("lengths"))
                    counts["flower"] += 1
            if fruit_label is not None and int(sem_label) == int(fruit_label):
                obb = self._compute_obb_from_points(pts)
                if obb is not None:
                    ann = self._ensure_annotation_entry(inst_id)
                    ann["fruit_obb"] = obb
                    ann["fruit_dims"] = self._obb_dims_from_lengths(obb.get("lengths"))
                    counts["fruit"] += 1
        return counts


    def _obb_dims_from_lengths(self, lengths) -> Optional[Dict[str, float]]:
        if lengths is None or len(lengths) != 3:
            return None
        vals = sorted([float(x) for x in lengths], reverse=True)
        return {"length": vals[0], "width": vals[1], "height": vals[2]}


    def compute_obb_instance(self, inst_id: int, kind: str) -> bool:
        self._require_cloud()
        pts = self.get_instance_points(inst_id)
        obb = self._compute_obb_from_points(pts)
        if obb is None:
            return False
        ann = self._ensure_annotation_entry(inst_id)
        if kind == "flower":
            ann["flower_obb"] = obb
            ann["flower_dims"] = self._obb_dims_from_lengths(obb.get("lengths"))
        elif kind == "fruit":
            ann["fruit_obb"] = obb
            ann["fruit_dims"] = self._obb_dims_from_lengths(obb.get("lengths"))
        else:
            return False
        return True
