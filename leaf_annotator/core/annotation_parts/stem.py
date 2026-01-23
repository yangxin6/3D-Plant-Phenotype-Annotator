# core/annotation_parts/stem.py
from typing import Optional, Dict, Any, Tuple, List

import numpy as np


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


    def _compute_stem_profile(
        self,
        pts: np.ndarray,
        segments: Optional[int] = None,
        percentile: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        if pts is None or len(pts) < 10:
            return None
        mean = pts.mean(axis=0)
        centered = pts - mean
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, int(np.argmax(eigvals))]
        axis = axis / (np.linalg.norm(axis) + 1e-12)

        t = centered @ axis
        tmin, tmax = float(t.min()), float(t.max())
        full_height = max(0.0, tmax - tmin)
        if percentile is None:
            percentile = self._get_stem_diameter_percentile()
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
        pct = float(percentile)

        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(axis[0])) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = np.cross(axis, ref)
        u = u / (np.linalg.norm(u) + 1e-12)
        v = np.cross(axis, u)

        def _fit_circle(xs: np.ndarray, ys: np.ndarray) -> Optional[Tuple[float, float, float]]:
            if len(xs) < 3:
                return None
            A = np.column_stack([xs, ys, np.ones_like(xs)])
            b = -(xs ** 2 + ys ** 2)
            try:
                params, *_ = np.linalg.lstsq(A, b, rcond=None)
                a, bcoef, c = params
                cx = -a / 2.0
                cy = -bcoef / 2.0
                r2 = cx * cx + cy * cy - c
            except Exception:
                return None
            if r2 <= 0:
                cx = 0.0
                cy = 0.0
            d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
            radius = float(np.percentile(d, pct))
            return cx, cy, radius

        edges = np.linspace(tmin, tmax, seg_count + 1) if seg_count > 1 else np.array([tmin, tmax])

        def _build_segments(min_pts: int) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], float]:
            segments: List[Dict[str, Any]] = []
            best_radius = 0.0
            best_seg: Optional[Dict[str, Any]] = None
            for i in range(len(edges) - 1):
                t0 = float(edges[i])
                t1 = float(edges[i + 1])
                if t1 <= t0:
                    continue
                is_last = (i == len(edges) - 2)
                mask = (t >= t0) & (t <= t1 if is_last else t < t1)
                if np.sum(mask) < min_pts:
                    continue
                seg_t = t[mask]
                seg_pts = centered[mask]
                proj = seg_pts - np.outer(seg_t, axis)
                xs = proj @ u
                ys = proj @ v
                fit = _fit_circle(xs, ys)
                if fit is None:
                    radial = np.linalg.norm(proj, axis=1)
                    if len(radial) < 1:
                        continue
                    radius = float(np.percentile(radial, pct))
                    cx = 0.0
                    cy = 0.0
                else:
                    cx, cy, radius = fit
                offset = u * cx + v * cy
                center = mean + axis * (0.5 * (t0 + t1)) + offset
                center_bottom = mean + axis * t0 + offset
                center_top = mean + axis * t1 + offset
                seg = {
                    "t0": float(t0),
                    "t1": float(t1),
                    "center": center.tolist(),
                    "center_bottom": center_bottom.tolist(),
                    "center_top": center_top.tolist(),
                    "radius": float(radius),
                }
                segments.append(seg)
                if radius > best_radius:
                    best_radius = float(radius)
                    best_seg = seg
            return segments, best_seg, best_radius

        segments: List[Dict[str, Any]] = []
        best_seg: Optional[Dict[str, Any]] = None

        if full_height <= 1e-9 or seg_count <= 1:
            proj = centered - np.outer(t, axis)
            xs = proj @ u
            ys = proj @ v
            fit = _fit_circle(xs, ys)
            if fit is None:
                radial = np.linalg.norm(proj, axis=1)
                radius = float(np.percentile(radial, pct))
                cx = 0.0
                cy = 0.0
            else:
                cx, cy, radius = fit
            t0 = tmin
            t1 = tmax
            offset = u * cx + v * cy
            center = mean + axis * (0.5 * (t0 + t1)) + offset
            center_bottom = mean + axis * t0 + offset
            center_top = mean + axis * t1 + offset
            seg = {
                "t0": float(t0),
                "t1": float(t1),
                "center": center.tolist(),
                "center_bottom": center_bottom.tolist(),
                "center_top": center_top.tolist(),
                "radius": float(radius),
            }
            segments.append(seg)
            best_seg = seg
        else:
            segments, best_seg, _ = _build_segments(10)
            if not segments or best_seg is None:
                segments, best_seg, _ = _build_segments(3)
            if not segments or best_seg is None:
                segments, best_seg, _ = _build_segments(1)

        if not segments or best_seg is None:
            return None

        segments = sorted(segments, key=lambda s: s["t0"])
        path_pts = [np.asarray(seg["center_bottom"], dtype=np.float64) for seg in segments]
        path_pts.append(np.asarray(segments[-1]["center_top"], dtype=np.float64))
        stem_length = _polyline_length(np.asarray(path_pts, dtype=np.float64))

        best_cyl = {
            "center": best_seg["center"],
            "axis": axis.tolist(),
            "height": float(best_seg["t1"] - best_seg["t0"]),
            "radius": float(best_seg["radius"]),
            "diameter": float(best_seg["radius"]) * 2.0,
        }

        return {
            "axis": axis.tolist(),
            "segments": segments,
            "best": best_cyl,
            "length_path": [p.tolist() for p in path_pts],
            "length": float(stem_length if stem_length is not None else 0.0),
        }


    def _compute_cylinder_from_points(self, pts: np.ndarray) -> Optional[Dict[str, Any]]:
        prof = self._compute_stem_profile(pts)
        if prof is None:
            return None
        return prof.get("best")


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
        for inst_id, sem_label in sem_map.items():
            if int(sem_label) != int(stem_label):
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
            if prof_len is not None:
                ann["stem_length_path"] = prof_len["length_path"]
                ann["stem_length"] = prof_len["length"]
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
        for inst_id, sem_label in sem_map.items():
            if int(sem_label) != int(stem_label):
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
        for inst_id, sem_label in sem_map.items():
            if int(sem_label) != int(stem_label):
                continue
            pts = self.get_instance_points(inst_id)
            prof = self._compute_stem_profile(pts, segments=length_segments, percentile=length_pct)
            if prof is None:
                continue
            ann = self._ensure_annotation_entry(inst_id)
            ann["stem_length_path"] = prof["length_path"]
            ann["stem_length"] = prof["length"]
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
        if prof is None:
            return False
        ann = self._ensure_annotation_entry(inst_id)
        ann["stem_length_path"] = prof["length_path"]
        ann["stem_length"] = prof["length"]
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
        if prof_len is not None:
            ann["stem_length_path"] = prof_len["length_path"]
            ann["stem_length"] = prof_len["length"]
            ok = True
        return ok
