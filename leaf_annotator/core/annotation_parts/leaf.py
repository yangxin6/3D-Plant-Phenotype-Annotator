# core/annotation_parts/leaf.py
from typing import List, Optional, Tuple
from types import SimpleNamespace

import numpy as np
from scipy.spatial import ConvexHull

from core.centerline import CenterlineExtractor
from core.width import WidthEstimator
from core.annotation_parts.utils import (
    _polyline_length,
    _resample_polyline_step,
    _smooth_polyline_window,
    _build_knn_graph,
    _build_radius_graph,
    _shortest_path_indices,
)


class LeafMixin:
    def compute_centerline(self, use_ctrl: bool = True):
        """
        叶长（中心线）两步：
        1）在 downsample 点云上构建半径图，分段最短路径：
           - use_ctrl=False：B1 -> T1
           - use_ctrl=True： B1 -> C1 -> ... -> T1
        2）对最短路按 step 等距重采样，并在每个 step 内局部“薄片”修正到质心附近的点。
        结果写入 self.centerline_result（含 smooth_points/length/path_indices）。
        """
        self._require_instance()
        self._require_base_tip()

        chain: List[int] = [int(self.base_idx)]
        if use_ctrl and len(self.ctrl_indices) > 0:
            chain += [int(i) for i in self.ctrl_indices]
        chain.append(int(self.tip_idx))

        pts = self.ds.points
        G = _build_knn_graph(pts, k=self.params.k)

        path_all: List[int] = []
        for a, b in zip(chain[:-1], chain[1:]):
            pidx = _shortest_path_indices(G, int(a), int(b))
            if pidx is None or len(pidx) == 0:
                raise RuntimeError("半径图不连通：起点到终点不可达，请增大 graph_radius。")
            if path_all:
                pidx = pidx[1:]
            path_all.extend([int(i) for i in pidx])

        path_all_arr = np.asarray(path_all, dtype=np.int64)
        poly = pts[path_all_arr]

        # Step 等距重采样 + 局部圆柱修正（按 step 线段）
        step = float(self.params.step)
        radius = float(self.params.radius)
        min_slice_pts = int(self.params.min_slice_pts)

        P_rs = _resample_polyline_step(poly, step=step)

        refined_idx: List[int] = []
        refined_pts: List[np.ndarray] = []

        def _dist_to_segment(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
            ab = b - a
            denom = float(np.dot(ab, ab))
            if denom <= 1e-12:
                return np.linalg.norm(points - a, axis=1)
            t = (points - a) @ ab / denom
            t = np.clip(t, 0.0, 1.0)
            proj = a + np.outer(t, ab)
            return np.linalg.norm(points - proj, axis=1)

        for i, p in enumerate(P_rs):
            p = np.asarray(p, dtype=np.float64)
            if i == 0:
                a = p
                b = P_rs[min(1, len(P_rs) - 1)]
            elif i == len(P_rs) - 1:
                a = P_rs[i - 1]
                b = p
            else:
                a = P_rs[i - 1]
                b = P_rs[i + 1]

            seg_len = float(np.linalg.norm(b - a))
            search_r = radius + 0.5 * seg_len
            cand_idx_list = self.ds_tree.query_ball_point(p, r=search_r) if self.ds_tree is not None else []
            if cand_idx_list is None:
                cand_idx_list = []
            cand_idx = np.asarray(cand_idx_list, dtype=np.int64)
            cand_pts = pts[cand_idx] if len(cand_idx) > 0 else np.empty((0, 3))

            # 圆柱约束：点到 step 线段的垂直距离 <= radius
            if len(cand_pts) > 0:
                d = _dist_to_segment(cand_pts, a, b)
                mask = d <= radius
                cand_idx = cand_idx[mask]
                cand_pts = cand_pts[mask]

            if len(cand_pts) >= max(1, min_slice_pts):
                c = cand_pts.mean(axis=0)
                d = np.linalg.norm(cand_pts - c, axis=1)
                j = int(np.argmin(d))
                refined_idx.append(int(cand_idx[j]))
                refined_pts.append(cand_pts[j])
            elif len(cand_pts) > 0:
                # 有点但不足 min_slice_pts，取最近
                d = np.linalg.norm(cand_pts - p, axis=1)
                j = int(np.argmin(d))
                refined_idx.append(int(cand_idx[j]))
                refined_pts.append(cand_pts[j])
            else:
                # 无候选，直接 snap 最近点
                _, j = self.ds_tree.query(p, k=1) if self.ds_tree is not None else (0.0, 0)
                refined_idx.append(int(j))
                refined_pts.append(pts[int(j)])

        refined_poly = np.asarray(refined_pts, dtype=np.float64)
        if len(refined_pts) > 0:
            if self.base_idx is not None:
                refined_idx[0] = int(self.base_idx)
                refined_pts[0] = pts[int(self.base_idx)]
            if self.tip_idx is not None:
                refined_idx[-1] = int(self.tip_idx)
                refined_pts[-1] = pts[int(self.tip_idx)]
            refined_poly = np.asarray(refined_pts, dtype=np.float64)
        L = _polyline_length(refined_poly)
        result = SimpleNamespace(
            smooth_points=refined_poly,
            length=float(L if L is not None else 0.0),
            path_indices=np.asarray(refined_idx, dtype=np.int64),
        )
        self._set_centerline_result(result, "recommended")
        return self.centerline_result


    def compute_centerline_polyline(self, use_ctrl: bool = True):
        self._require_instance()
        self._require_base_tip()

        poly = self.build_length_polyline(use_ctrl=use_ctrl)
        L = _polyline_length(poly)
        result = SimpleNamespace(
            smooth_points=poly,
            length=float(L if L is not None else 0.0),
            path_indices=None,
        )
        self._set_centerline_result(result, "polyline")
        return self.centerline_result


    def recommend_width_endpoints(self, overwrite: bool = False) -> bool:
        """
        用旧的“最大叶宽”算法给出 W1/W2 推荐点（snap 到 ds 索引）。
        返回是否成功写入推荐。
        - overwrite=False：仅当用户没设 W1/W2 且未推荐过时才写入
        """
        if self.leaf_pts is None or self.ds is None:
            return False
        if self.base_idx is None or self.tip_idx is None:
            return False

        # 不覆盖用户选择；也避免用户删除后自动再冒出来
        if not overwrite:
            if self._width_recommended_once:
                return False
            if self.width_w1_idx is not None or self.width_w2_idx is not None:
                return False

        # 叶宽端点推荐：优先使用控制点折线，保证稳定可控
        try:
            length_poly = self.build_length_polyline(use_ctrl=True)
        except Exception:
            length_poly = None
        if length_poly is None and self.centerline_result is not None:
            length_poly = np.asarray(self.centerline_result.smooth_points, dtype=np.float64)
        if length_poly is None:
            return False

        # 用旧 WidthEstimator 找最大叶宽段
        we = WidthEstimator(
            step=self.params.step,
            slab_half=self.params.slab_half,
            radius=self.params.radius,
            min_slice_pts=self.params.min_slice_pts
        )
        wr = we.compute(self.leaf_pts, length_poly)
        if wr is None or wr.max_item is None:
            self._width_recommended_once = True
            return False

        pL = np.asarray(wr.max_item.pL, dtype=np.float64)
        pR = np.asarray(wr.max_item.pR, dtype=np.float64)

        # snap 到 ds 索引
        w1 = self.snap_to_ds_index(pL)
        w2 = self.snap_to_ds_index(pR)

        self.width_w1_idx = int(w1)
        self.width_w2_idx = int(w2)

        self._width_recommended_once = True
        return True


    def recommend_width_endpoints_auto(self) -> bool:
        return self.recommend_width_endpoints(overwrite=False)

    # ---------- compute ----------

    def compute(self):
        """
        - 叶长：优先用半径图最短路径 + step 重采样薄片修正（B1->ctrl...->T1），失败则回退为直连折线
        - 叶宽：若有 W1/W2（用户或推荐），则计算最短路径（kNN 图）
        """
        self._require_instance()
        self._require_base_tip()

        try:
            self.compute_centerline(use_ctrl=True)
        except Exception:
            try:
                self.compute_centerline_polyline(use_ctrl=True)
            except Exception:
                length_poly = self.build_length_polyline(use_ctrl=True)
                L = _polyline_length(length_poly)
                result = SimpleNamespace(
                    smooth_points=length_poly,
                    length=float(L if L is not None else 0.0),
                    path_indices=None,
                )
                self._set_centerline_result(result, "polyline")

        # 如果用户没设置 W1/W2，就推荐一次
        self.recommend_width_endpoints(overwrite=False)

        # width shortest path (optional)
        self.width_path_points = None
        self.width_path_length = None

        if self.width_w1_idx is not None and self.width_w2_idx is not None:
            pts = self.ds.points
            G = _build_knn_graph(pts, k=self.params.k)
            chain: List[int] = [int(self.width_w1_idx)]
            if len(self.width_ctrl_indices) > 0:
                chain += [int(i) for i in self.get_sorted_width_ctrl_indices()]
            chain.append(int(self.width_w2_idx))

            path_all: List[int] = []
            ok = True
            for a, b in zip(chain[:-1], chain[1:]):
                pidx = _shortest_path_indices(G, int(a), int(b))
                if pidx is None or len(pidx) == 0:
                    ok = False
                    break
                if path_all:
                    pidx = pidx[1:]
                path_all.extend([int(i) for i in pidx])

            if ok and len(path_all) >= 2:
                path_idx = np.asarray(path_all, dtype=np.int64)
                path_pts = pts[path_idx]
                self.width_path_points = path_pts
                self.width_path_length = _polyline_length(path_pts)
                self.width_path_indices = path_idx
            else:
                self.width_path_indices = None
        else:
            self.width_path_indices = None

        self.point_labels = self.compute_point_labels(self.params.label_radius)


    def compute_width_path(self):
        self._require_instance()
        self._require_width_endpoints()

        pts = self.ds.points
        G = _build_radius_graph(pts, radius=self.params.graph_radius)
        chain: List[int] = [int(self.width_w1_idx)]
        if len(self.width_ctrl_indices) > 0:
            chain += [int(i) for i in self.get_sorted_width_ctrl_indices()]
        chain.append(int(self.width_w2_idx))

        path_all: List[int] = []
        for a, b in zip(chain[:-1], chain[1:]):
            pidx = _shortest_path_indices(G, int(a), int(b))
            if pidx is None or len(pidx) == 0:
                raise RuntimeError("叶宽最短路径不可达，请调整 W1/W2 或增加 k。")
            if path_all:
                pidx = pidx[1:]
            path_all.extend([int(i) for i in pidx])

        if len(path_all) < 2:
            raise RuntimeError("叶宽路径过短，无法生成。")

        path_idx = np.asarray(path_all, dtype=np.int64)
        path_pts = pts[path_idx]
        self.width_path_points = path_pts
        self.width_path_length = _polyline_length(path_pts)
        self.width_path_indices = path_idx


    def _get_length_polyline_for_area(self) -> np.ndarray:
        if self.centerline_result is not None and getattr(self.centerline_result, "smooth_points", None) is not None:
            return np.asarray(self.centerline_result.smooth_points, dtype=np.float64)
        self._require_base_tip()
        return self.build_length_polyline(use_ctrl=True)


    def compute_leaf_area_instance(self, step: Optional[float] = None) -> Optional[float]:
        self._require_instance()
        if self.current_inst_id is None:
            return None
        length_poly = self._get_length_polyline_for_area()
        if length_poly is None or len(length_poly) < 2:
            return None
        step_val = float(self.params.step if step is None else step)
        if step_val <= 0:
            raise ValueError("step must be > 0")

        we = WidthEstimator(
            step=step_val,
            slab_half=self.params.slab_half,
            radius=self.params.radius,
            min_slice_pts=self.params.min_slice_pts,
        )
        wr = we.compute(self.leaf_pts, length_poly)
        if wr is None or not wr.profile:
            return None

        C = CenterlineExtractor.resample_by_step(length_poly, step=step_val)
        if len(C) < 2:
            return None

        area = 0.0
        for item in wr.profile:
            s = int(item.s_index)
            if s < 0 or s + 1 >= len(C):
                continue
            seg_len = float(np.linalg.norm(C[s + 1] - C[s]))
            area += float(item.width) * seg_len

        ann = self._ensure_annotation_entry(self.current_inst_id)
        ann["leaf_area"] = float(area)
        return float(area)


    def compute_leaf_projected_area_instance(self) -> Optional[float]:
        self._require_instance()
        if self.current_inst_id is None:
            return None
        if self.leaf_pts is None or len(self.leaf_pts) < 3:
            return None
        xy = np.asarray(self.leaf_pts[:, :2], dtype=np.float64)
        try:
            hull = ConvexHull(xy)
        except Exception:
            return None
        area = float(hull.volume)
        ann = self._ensure_annotation_entry(self.current_inst_id)
        ann["leaf_projected_area"] = float(area)
        return float(area)


    def _compute_leaf_plane_normal(
        self,
        center: Optional[np.ndarray] = None,
        radius: Optional[float] = None
    ) -> Optional[np.ndarray]:
        if self.leaf_pts is None or len(self.leaf_pts) < 3:
            return None
        pts = np.asarray(self.leaf_pts, dtype=np.float64)
        if center is not None and radius is not None and radius > 0:
            center = np.asarray(center, dtype=np.float64)
            dist = np.linalg.norm(pts - center[None, :], axis=1)
            local = pts[dist <= float(radius)]
            if len(local) >= 3:
                pts = local
        centered = pts - pts.mean(axis=0, keepdims=True)
        cov = np.cov(centered, rowvar=False)
        if cov.shape != (3, 3):
            return None
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, int(np.argmin(eigvals))]
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            return None
        return normal / norm


    def compute_leaf_inclination_instance(self) -> Optional[float]:
        self._require_instance()
        if self.current_inst_id is None:
            return None
        length_poly = self._get_length_polyline_for_area()
        if length_poly is None or len(length_poly) < 2:
            return None
        mid = np.asarray(length_poly[len(length_poly) // 2], dtype=np.float64)
        radius = max(
            float(getattr(self.params, "radius", 0.0)),
            float(getattr(self.params, "slab_half", 0.0)),
            float(getattr(self.params, "voxel", 0.0)),
        )
        if radius <= 0:
            radius = None
        normal = self._compute_leaf_plane_normal(center=mid, radius=radius)
        if normal is None:
            return None
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        cos_val = float(np.dot(normal, z_axis))
        cos_val = abs(cos_val)
        cos_val = float(np.clip(cos_val, -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_val)))
        ann = self._ensure_annotation_entry(self.current_inst_id)
        ann["leaf_inclination"] = float(angle_deg)
        return float(angle_deg)


    def _path_direction(self, path: np.ndarray, index: int) -> Optional[np.ndarray]:
        if path is None or len(path) < 2:
            return None
        idx = int(np.clip(int(index), 0, len(path) - 1))
        if idx <= 0:
            vec = path[1] - path[0]
        elif idx >= len(path) - 1:
            vec = path[-1] - path[-2]
        else:
            vec = path[idx + 1] - path[idx - 1]
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-12:
            return None
        return vec / norm


    def _leaf_initial_direction(self, length_poly: np.ndarray) -> Optional[np.ndarray]:
        if length_poly is None or len(length_poly) < 2:
            return None
        total_len = _polyline_length(length_poly)
        if total_len is None or total_len <= 1e-12:
            vec = length_poly[1] - length_poly[0]
            norm = float(np.linalg.norm(vec))
            if norm <= 1e-12:
                return None
            return vec / norm
        target = 0.1 * float(total_len)
        cum = 0.0
        idx = 1
        for i in range(len(length_poly) - 1):
            seg = float(np.linalg.norm(length_poly[i + 1] - length_poly[i]))
            cum += seg
            if cum >= target:
                idx = i + 1
                break
        vec = length_poly[idx] - length_poly[0]
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-12:
            return None
        return vec / norm


    def _stem_direction_near_point(self, point_xyz: np.ndarray) -> Optional[np.ndarray]:
        stem_label = self.semantic_map.get("stem")
        if stem_label is None:
            return None
        sem_map = self._get_instance_sem_map()
        best_dir = None
        best_dist = None
        point_xyz = np.asarray(point_xyz, dtype=np.float64)
        for inst_id, ann in self.annotations.items():
            if sem_map.get(int(inst_id)) != int(stem_label):
                continue
            path = ann.get("stem_length_path")
            if path is None:
                continue
            pts = np.asarray(path, dtype=np.float64)
            if len(pts) < 2:
                continue
            dist = np.linalg.norm(pts - point_xyz[None, :], axis=1)
            idx = int(np.argmin(dist))
            dir_vec = self._path_direction(pts, idx)
            if dir_vec is None:
                continue
            dmin = float(dist[idx])
            if best_dist is None or dmin < best_dist:
                best_dist = dmin
                best_dir = dir_vec
        if best_dir is None:
            return None
        if best_dir[2] < 0:
            best_dir = -best_dir
        return best_dir


    def compute_leaf_stem_angle_instance(self) -> Optional[float]:
        self._require_instance()
        if self.current_inst_id is None:
            return None
        length_poly = self._get_length_polyline_for_area()
        if length_poly is None or len(length_poly) < 2:
            return None
        leaf_dir = self._leaf_initial_direction(length_poly)
        if leaf_dir is None:
            return None
        stem_dir = self._stem_direction_near_point(length_poly[0])
        if stem_dir is None:
            return None
        cos_val = float(np.dot(leaf_dir, stem_dir))
        cos_val = float(np.clip(cos_val, -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_val)))
        ann = self._ensure_annotation_entry(self.current_inst_id)
        ann["leaf_stem_angle"] = float(angle_deg)
        return float(angle_deg)


    def smooth_leaf_paths(self, win: int) -> Tuple[bool, bool]:
        self._require_instance()
        w = max(3, int(win) | 1)
        updated_len = False
        updated_wid = False

        if self.centerline_result is not None and getattr(self.centerline_result, "smooth_points", None) is not None:
            smoothed = _smooth_polyline_window(self.centerline_result.smooth_points, w)
            if smoothed is not None and len(smoothed) >= 2:
                self.centerline_result.smooth_points = smoothed
                L = _polyline_length(smoothed)
                self.centerline_result.length = float(L if L is not None else 0.0)
                updated_len = True

        if self.width_path_points is not None:
            smoothed = _smooth_polyline_window(self.width_path_points, w)
            if smoothed is not None and len(smoothed) >= 2:
                self.width_path_points = smoothed
                self.width_path_length = _polyline_length(smoothed)
                self.width_path_indices = None
                updated_wid = True

        if updated_len or updated_wid:
            if self.centerline_result is not None and self.width_path_points is not None:
                self.point_labels = self.compute_point_labels(self.params.label_radius)

        return updated_len, updated_wid


    def compute_point_labels(self, radius: float = 0.01) -> Optional[np.ndarray]:
        if self.leaf_pts is None:
            return None
        if self.centerline_result is None or self.width_path_points is None:
            return None
        n = len(self.leaf_pts)
        if n == 0:
            return np.zeros((0,), dtype=np.int64)
        labels = np.full(n, -1, dtype=np.int64)
        r = float(radius)
        if r <= 0:
            return labels

        def _points_within_polyline_radius(poly: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if poly is None or len(poly) < 2:
                return None
            P = np.asarray(poly, dtype=np.float64)
            A = P[:-1]
            B = P[1:]
            AB = B - A
            denom = (AB * AB).sum(axis=1)
            denom = np.maximum(denom, 1e-12)
            pts = self.leaf_pts.astype(np.float64)
            hit = np.zeros(len(pts), dtype=bool)
            for i in range(len(A)):
                a = A[i]
                ab = AB[i]
                d = denom[i]
                ap = pts - a
                t = (ap @ ab) / d
                t = np.clip(t, 0.0, 1.0)
                proj = a + t[:, None] * ab
                dist = np.linalg.norm(pts - proj, axis=1)
                hit |= dist <= r
            idx = np.where(hit)[0]
            return idx

        length_pts = None
        if self.centerline_result is not None and self.centerline_result.smooth_points is not None:
            length_pts = np.asarray(self.centerline_result.smooth_points, dtype=np.float64)
        elif self.base_idx is not None and self.tip_idx is not None:
            length_pts = self.build_length_polyline(use_ctrl=True)

        idx_len = _points_within_polyline_radius(length_pts)
        idx_w = _points_within_polyline_radius(self.width_path_points)
        if idx_len is not None and len(idx_len) > 0:
            labels[idx_len] = 0
        if idx_w is not None and len(idx_w) > 0:
            labels[idx_w] = 1
        if idx_len is not None and len(idx_len) > 0 and idx_w is not None and len(idx_w) > 0:
            overlap = np.intersect1d(idx_len, idx_w, assume_unique=False)
            if len(overlap) > 0:
                labels[overlap] = 100

        return labels

    # ---------- cache helpers ----------
