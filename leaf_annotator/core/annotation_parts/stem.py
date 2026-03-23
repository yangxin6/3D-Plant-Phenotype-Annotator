# core/annotation_parts/stem.py
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
from scipy.spatial import cKDTree


class StemMixin:
    def _resolve_stem_segments(self, full_height: float, segments: Optional[int]) -> int:
        seg = 0
        if segments is not None:
            try:
                seg = int(segments)
            except Exception:
                seg = 0
        if seg > 0:
            return seg
        step = getattr(self.params, "stem_step", None)
        if step is not None:
            try:
                step = float(step)
            except Exception:
                step = None
        if step is not None and step > 0.0 and full_height > 0.0:
            return max(1, int(np.ceil(full_height / step)))
        return 1


    def _get_stem_diameter_percentile(self) -> float:
        pct = getattr(self.params, "stem_diameter_percentile", None)
        if pct is None:
            pct = getattr(self.params, "stem_percentile", 95.0)
        return float(pct)


    def _get_stem_length_percentile(self) -> float:
        pct = getattr(self.params, "stem_length_percentile", None)
        if pct is None:
            pct = getattr(self.params, "stem_percentile", 95.0)
        return float(pct)


    def _normalize_axis(self, axis: np.ndarray) -> Optional[np.ndarray]:
        if axis is None:
            return None
        axis = np.asarray(axis, dtype=np.float64).reshape(-1)
        if axis.shape[0] != 3:
            return None
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-12:
            return None
        axis = axis / norm
        if axis[2] < 0:
            axis = -axis
        return axis


    def _get_stem_axis(self, pts: np.ndarray) -> Optional[np.ndarray]:
        growth_axis = self._normalize_axis(getattr(self, "growth_direction", None))
        if growth_axis is not None:
            return growth_axis
        if pts is None or len(pts) < 3:
            return None
        centered = pts - pts.mean(axis=0)
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, int(np.argmax(eigvals))]
        return self._normalize_axis(axis)


    def _mad_sigma(self, values: np.ndarray) -> float:
        if values is None or len(values) == 0:
            return 0.0
        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med)))
        return 1.4826 * mad


    def _remove_knn_outliers(self, pts: np.ndarray, k: int = 8, sigma_scale: float = 2.5) -> np.ndarray:
        if pts is None or len(pts) < 12:
            return pts
        kk = min(max(2, int(k)), len(pts) - 1)
        if kk < 2:
            return pts
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=kk + 1)
        neigh_dist = np.asarray(dists[:, -1], dtype=np.float64)
        med = float(np.median(neigh_dist))
        sigma = self._mad_sigma(neigh_dist)
        if sigma <= 1e-12:
            return pts
        keep = neigh_dist <= med + sigma_scale * sigma
        if np.sum(keep) < max(10, len(pts) // 3):
            return pts
        return pts[keep]


    def _trim_axial_outliers(self, pts: np.ndarray, axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        origin = pts.mean(axis=0)
        t = (pts - origin) @ axis
        if len(pts) >= 50:
            t_lo, t_hi = np.percentile(t, [1.0, 99.0])
            keep = (t >= float(t_lo)) & (t <= float(t_hi))
            if np.sum(keep) >= max(10, len(pts) // 2):
                pts = pts[keep]
                origin = pts.mean(axis=0)
                t = (pts - origin) @ axis
        return pts, origin, t, float(t.min()), float(t.max())


    def _radial_distance_to_axis(self, pts: np.ndarray, axis: np.ndarray, origin: np.ndarray) -> np.ndarray:
        rel = pts - origin
        proj = rel @ axis
        perp = rel - np.outer(proj, axis)
        return np.linalg.norm(perp, axis=1)


    def _remove_radial_outliers(self, pts: np.ndarray, axis: np.ndarray, origin: np.ndarray, sigma_scale: float = 2.5) -> np.ndarray:
        if pts is None or len(pts) < 12:
            return pts
        radial = self._radial_distance_to_axis(pts, axis, origin)
        med = float(np.median(radial))
        sigma = self._mad_sigma(radial)
        if sigma <= 1e-12:
            keep = radial <= np.percentile(radial, 95.0)
        else:
            keep = radial <= med + sigma_scale * sigma
        if np.sum(keep) < max(10, len(pts) // 3):
            return pts
        return pts[keep]


    def _denoise_stem_points(self, pts: np.ndarray, axis: np.ndarray) -> np.ndarray:
        pts_work = np.asarray(pts, dtype=np.float64)
        denoise_k = int(getattr(self.params, "stem_denoise_knn_k", 8))
        denoise_knn_sigma = float(getattr(self.params, "stem_denoise_knn_sigma", 2.5))
        denoise_radial_sigma = float(getattr(self.params, "stem_denoise_radial_sigma", 2.5))
        denoise_k = max(2, denoise_k)
        denoise_knn_sigma = max(0.5, denoise_knn_sigma)
        denoise_radial_sigma = max(0.5, denoise_radial_sigma)

        pts_work = self._remove_knn_outliers(pts_work, k=denoise_k, sigma_scale=denoise_knn_sigma)
        if len(pts_work) < 10:
            return np.asarray(pts, dtype=np.float64)
        origin = pts_work.mean(axis=0)
        pts_work = self._remove_radial_outliers(pts_work, axis, origin, sigma_scale=denoise_radial_sigma)
        if len(pts_work) < 10:
            return np.asarray(pts, dtype=np.float64)
        return pts_work


    def _compute_segment_radius(
        self,
        seg_pts: np.ndarray,
        axis: np.ndarray,
        percentile: float,
    ) -> Optional[float]:
        if seg_pts is None or len(seg_pts) < 6:
            return None
        origin = seg_pts.mean(axis=0)
        radial = self._radial_distance_to_axis(seg_pts, axis, origin)
        if len(radial) < 3:
            return None
        med = float(np.median(radial))
        sigma = self._mad_sigma(radial)
        if sigma > 1e-12:
            keep = radial <= med + 2.5 * sigma
            if np.sum(keep) >= 4:
                radial = radial[keep]
        radius = float(np.percentile(radial, float(percentile)))
        if not np.isfinite(radius) or radius <= 0.0:
            return None
        return radius


    def _compute_stem_profile(
        self,
        pts: np.ndarray,
        segments: Optional[int] = None,
        percentile: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        if pts is None or len(pts) < 10:
            return None
        axis = self._get_stem_axis(pts)
        if axis is None:
            return None
        pts_work = self._denoise_stem_points(pts, axis)
        if pts_work is None or len(pts_work) < 10:
            return None
        pts_work, origin, t, tmin, tmax = self._trim_axial_outliers(pts_work, axis)
        if len(pts_work) < 10 or tmax <= tmin:
            return None

        if percentile is None:
            percentile = self._get_stem_diameter_percentile()
        full_height = float(tmax - tmin)
        if segments is None:
            segments = 0
        try:
            seg_val = int(segments)
        except Exception:
            seg_val = 0
        if seg_val <= 0:
            try:
                seg_val = int(getattr(self.params, "stem_segments", 0))
            except Exception:
                seg_val = 0
        seg_count = self._resolve_stem_segments(full_height, seg_val)
        edges = np.linspace(tmin, tmax, seg_count + 1) if seg_count > 1 else np.array([tmin, tmax], dtype=np.float64)
        min_pts = max(8, int(np.ceil(len(pts_work) / max(8.0, 2.0 * seg_count))))

        segment_entries: List[Dict[str, Any]] = []
        segment_radii: List[float] = []
        path_pts: List[np.ndarray] = []
        for i in range(len(edges) - 1):
            t0 = float(edges[i])
            t1 = float(edges[i + 1])
            if t1 <= t0:
                continue
            is_last = i == len(edges) - 2
            mask = (t >= t0) & (t <= t1 if is_last else t < t1)
            if int(np.sum(mask)) < min_pts:
                continue
            seg_pts = pts_work[mask]
            radius = self._compute_segment_radius(seg_pts, axis, float(percentile))
            if radius is None:
                continue
            center = origin + axis * (0.5 * (t0 + t1))
            center_bottom = origin + axis * t0
            center_top = origin + axis * t1
            path_pts.append(center)
            segment_entries.append({
                "t0": t0,
                "t1": t1,
                "center": center.tolist(),
                "center_bottom": center_bottom.tolist(),
                "center_top": center_top.tolist(),
                "radius": float(radius),
                "diameter": float(2.0 * radius),
            })
            segment_radii.append(float(radius))

        if not segment_radii:
            radius = self._compute_segment_radius(pts_work, axis, float(percentile))
            if radius is None:
                return None
            center = origin + axis * (0.5 * (tmin + tmax))
            center_bottom = origin + axis * tmin
            center_top = origin + axis * tmax
            segment_entries = [{
                "t0": tmin,
                "t1": tmax,
                "center": center.tolist(),
                "center_bottom": center_bottom.tolist(),
                "center_top": center_top.tolist(),
                "radius": float(radius),
                "diameter": float(2.0 * radius),
            }]
            segment_radii = [float(radius)]
            path_pts = [center_bottom, center_top]

        rep_radius = float(np.median(np.asarray(segment_radii, dtype=np.float64)))
        cyl_center = origin + axis * (0.5 * (tmin + tmax))
        cyl_bottom = origin + axis * tmin
        cyl_top = origin + axis * tmax
        best_cyl = {
            "center": cyl_center.tolist(),
            "center_bottom": cyl_bottom.tolist(),
            "center_top": cyl_top.tolist(),
            "axis": axis.tolist(),
            "height": float(full_height),
            "radius": float(rep_radius),
            "diameter": float(2.0 * rep_radius),
            "point_count": int(len(pts_work)),
        }

        path_arr = np.asarray(path_pts, dtype=np.float64)
        if len(path_arr) < 2:
            path_arr = np.asarray([cyl_bottom, cyl_top], dtype=np.float64)

        return {
            "axis": axis.tolist(),
            "segments": segment_entries,
            "best": best_cyl,
            "length_path": [p.tolist() for p in path_arr],
            "length": float(full_height),
            "denoised_point_count": int(len(pts_work)),
        }


    def _compute_cylinder_from_points(self, pts: np.ndarray) -> Optional[Dict[str, Any]]:
        prof = self._compute_stem_profile(pts)
        if prof is None:
            return None
        return prof.get("best")


    def _compute_stem_length_from_growth(self, pts: np.ndarray) -> Optional[Dict[str, Any]]:
        axis = self._normalize_axis(getattr(self, "growth_direction", None))
        if axis is None or pts is None or len(pts) < 2:
            return None
        pts_work = self._denoise_stem_points(pts, axis)
        if pts_work is None or len(pts_work) < 2:
            return None
        origin = pts_work.mean(axis=0)
        t = (pts_work - origin) @ axis
        tmin = float(np.min(t))
        tmax = float(np.max(t))
        if tmax <= tmin:
            return None
        p1 = origin + axis * tmin
        p2 = origin + axis * tmax
        return {
            "length_path": [p1.tolist(), p2.tolist()],
            "length": float(tmax - tmin),
        }


    def compute_stem_structures(self) -> int:
        self._require_cloud()
        stem_label = self.semantic_map.get("stem")
        if stem_label is None:
            return 0
        sem_map = self._get_instance_sem_map()
        stem_pct = self._get_stem_diameter_percentile()
        length_pct = self._get_stem_length_percentile()
        stem_segments = getattr(self.params, "stem_diameter_segments", 0)
        length_segments = getattr(self.params, "stem_length_segments", 0)
        count = 0
        for inst_id, sem_label_cur in sem_map.items():
            if int(sem_label_cur) != int(stem_label):
                continue
            pts = self.get_instance_points(inst_id)
            ann = self._ensure_annotation_entry(inst_id)
            ok = False
            prof = self._compute_stem_profile(pts, segments=stem_segments, percentile=stem_pct)
            if prof is not None:
                ann["stem_cylinder"] = prof["best"]
                ann["stem_diameter"] = prof["best"]["diameter"]
                ann["stem_segments"] = prof["segments"]
                ok = True
            prof_len = self._compute_stem_profile(pts, segments=length_segments, percentile=length_pct)
            length_res = self._compute_stem_length_from_growth(pts) or prof_len
            if length_res is not None:
                ann["stem_length_path"] = length_res["length_path"]
                ann["stem_length"] = length_res["length"]
                ok = True
            if ok:
                count += 1
        return count


    def compute_stem_diameter_structures(self) -> int:
        self._require_cloud()
        stem_label = self.semantic_map.get("stem")
        if stem_label is None:
            return 0
        sem_map = self._get_instance_sem_map()
        stem_pct = self._get_stem_diameter_percentile()
        stem_segments = getattr(self.params, "stem_diameter_segments", 0)
        count = 0
        for inst_id, sem_label_cur in sem_map.items():
            if int(sem_label_cur) != int(stem_label):
                continue
            pts = self.get_instance_points(inst_id)
            prof = self._compute_stem_profile(pts, segments=stem_segments, percentile=stem_pct)
            if prof is None:
                continue
            ann = self._ensure_annotation_entry(inst_id)
            ann["stem_cylinder"] = prof["best"]
            ann["stem_diameter"] = prof["best"]["diameter"]
            ann["stem_segments"] = prof["segments"]
            count += 1
        return count


    def compute_stem_length_structures(self) -> int:
        self._require_cloud()
        stem_label = self.semantic_map.get("stem")
        if stem_label is None:
            return 0
        sem_map = self._get_instance_sem_map()
        length_pct = self._get_stem_length_percentile()
        length_segments = getattr(self.params, "stem_length_segments", 0)
        count = 0
        for inst_id, sem_label_cur in sem_map.items():
            if int(sem_label_cur) != int(stem_label):
                continue
            pts = self.get_instance_points(inst_id)
            prof = self._compute_stem_profile(pts, segments=length_segments, percentile=length_pct)
            length_res = self._compute_stem_length_from_growth(pts) or prof
            if length_res is None:
                continue
            ann = self._ensure_annotation_entry(inst_id)
            ann["stem_length_path"] = length_res["length_path"]
            ann["stem_length"] = length_res["length"]
            count += 1
        return count


    def compute_stem_diameter_instance(self, inst_id: int) -> bool:
        self._require_cloud()
        pts = self.get_instance_points(inst_id)
        stem_pct = self._get_stem_diameter_percentile()
        stem_segments = getattr(self.params, "stem_diameter_segments", 0)
        prof = self._compute_stem_profile(pts, segments=stem_segments, percentile=stem_pct)
        if prof is None:
            return False
        ann = self._ensure_annotation_entry(inst_id)
        ann["stem_cylinder"] = prof["best"]
        ann["stem_diameter"] = prof["best"]["diameter"]
        ann["stem_segments"] = prof["segments"]
        return True


    def compute_stem_length_instance(self, inst_id: int) -> bool:
        self._require_cloud()
        pts = self.get_instance_points(inst_id)
        length_pct = self._get_stem_length_percentile()
        length_segments = getattr(self.params, "stem_length_segments", 0)
        prof = self._compute_stem_profile(pts, segments=length_segments, percentile=length_pct)
        length_res = self._compute_stem_length_from_growth(pts) or prof
        if length_res is None:
            return False
        ann = self._ensure_annotation_entry(inst_id)
        ann["stem_length_path"] = length_res["length_path"]
        ann["stem_length"] = length_res["length"]
        return True


    def compute_stem_instance(self, inst_id: int) -> bool:
        self._require_cloud()
        pts = self.get_instance_points(inst_id)
        stem_pct = self._get_stem_diameter_percentile()
        length_pct = self._get_stem_length_percentile()
        stem_segments = getattr(self.params, "stem_diameter_segments", 0)
        length_segments = getattr(self.params, "stem_length_segments", 0)
        ann = self._ensure_annotation_entry(inst_id)
        ok = False
        prof = self._compute_stem_profile(pts, segments=stem_segments, percentile=stem_pct)
        if prof is not None:
            ann["stem_cylinder"] = prof["best"]
            ann["stem_diameter"] = prof["best"]["diameter"]
            ann["stem_segments"] = prof["segments"]
            ok = True
        prof_len = self._compute_stem_profile(pts, segments=length_segments, percentile=length_pct)
        length_res = self._compute_stem_length_from_growth(pts) or prof_len
        if length_res is not None:
            ann["stem_length_path"] = length_res["length_path"]
            ann["stem_length"] = length_res["length"]
            ok = True
        return ok
