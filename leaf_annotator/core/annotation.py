import json
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from types import SimpleNamespace

from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from core.io import PointCloudIO
from core.schema import CloudSchema, CloudParser, ParsedCloud
from core.sampling import VoxelSampler, Downsampled
from core.width import WidthEstimator  # ✅ 恢复：用于推荐叶宽点


@dataclass
class AnnotationParams:
    voxel: float = 0.003
    k: int = 25
    smooth_win: int = 9  # 仍保留但叶长不再平滑

    # 旧“最大叶宽”估计参数（用于推荐）
    step: float = 0.01
    slab_half: float = 0.1
    radius: float = 0.1
    min_slice_pts: int = 60


def _polyline_length(P: Optional[np.ndarray]) -> Optional[float]:
    if P is None or len(P) < 2:
        return None
    return float(np.linalg.norm(P[1:] - P[:-1], axis=1).sum())


def _resample_polyline_step(P: np.ndarray, step: float) -> np.ndarray:
    if P is None or len(P) < 2:
        return np.asarray(P).copy()
    seg = np.linalg.norm(P[1:] - P[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = s[-1]
    if L <= 1e-12 or step <= 0:
        return P.copy()
    m = max(2, int(np.floor(L / step)) + 1)
    s_new = np.linspace(0, L, m)

    out = []
    j = 0
    for sn in s_new:
        while j < len(s) - 2 and s[j + 1] < sn:
            j += 1
        denom = (s[j + 1] - s[j])
        t = 0.0 if abs(denom) < 1e-12 else (sn - s[j]) / denom
        out.append((1 - t) * P[j] + t * P[j + 1])
    return np.array(out)


def _build_knn_graph(points: np.ndarray, k: int) -> csr_matrix:
    n = int(points.shape[0])
    if n == 0:
        return csr_matrix((0, 0), dtype=np.float64)

    kk = max(1, int(k))
    tree = cKDTree(points)

    dists, nbrs = tree.query(points, k=min(kk + 1, n))
    if dists.ndim == 1:
        dists = dists[:, None]
        nbrs = nbrs[:, None]

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for i in range(n):
        for jpos in range(1, nbrs.shape[1]):
            j = int(nbrs[i, jpos])
            if j < 0 or j >= n or j == i:
                continue
            w = float(dists[i, jpos])
            rows.append(i); cols.append(j); data.append(w)
            rows.append(j); cols.append(i); data.append(w)

    G = csr_matrix(
        (np.array(data, dtype=np.float64),
         (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(n, n)
    )
    return G


def _shortest_path_indices(G: csr_matrix, start: int, end: int) -> Optional[List[int]]:
    n = G.shape[0]
    if start < 0 or start >= n or end < 0 or end >= n:
        return None
    if start == end:
        return [start]

    dist, pred = dijkstra(G, indices=start, return_predecessors=True)
    if np.isinf(dist[end]):
        return None

    path = [end]
    cur = end
    while cur != start:
        cur = int(pred[cur])
        if cur < 0:
            return None
        path.append(cur)
    path.reverse()
    return path


class LeafAnnotationSession:
    def __init__(self, params: Optional[AnnotationParams] = None, schema: Optional[CloudSchema] = None):
        self.params = params or AnnotationParams()
        self.schema = schema or CloudSchema()

        self.file_path: Optional[str] = None
        self.cloud: Optional[ParsedCloud] = None

        self.current_inst_id: Optional[int] = None
        self.leaf_pts: Optional[np.ndarray] = None
        self.leaf_global_idx: Optional[np.ndarray] = None

        self.ds: Optional[Downsampled] = None
        self.ds_tree: Optional[cKDTree] = None

        # saved picks (ds index)
        self.base_idx: Optional[int] = None
        self.tip_idx: Optional[int] = None
        self.ctrl_indices: List[int] = []

        # width endpoints (ds index)
        self.width_w1_idx: Optional[int] = None
        self.width_w2_idx: Optional[int] = None

        # 推荐端点只自动给一次；用户删除后不再自动冒出来
        self._width_recommended_once: bool = False

        # results
        self.centerline_result = None              # 有 smooth_points / length
        self.width_path_points: Optional[np.ndarray] = None
        self.width_path_length: Optional[float] = None

        self.annotations: Dict[int, Dict[str, Any]] = {}

    # ---------- full cloud ----------
    def load(self, path: str):
        arr = PointCloudIO.load_array(path)
        self.cloud = CloudParser.parse(arr, self.schema)
        self.file_path = path
        self.annotations = {}
        self.clear_instance_state()

    def has_rgb(self) -> bool:
        return (self.cloud is not None) and (self.cloud.rgb is not None)

    def get_full_xyz(self) -> np.ndarray:
        if self.cloud is None:
            raise RuntimeError("未加载点云。")
        return self.cloud.xyz

    def get_full_rgb(self) -> Optional[np.ndarray]:
        if self.cloud is None:
            raise RuntimeError("未加载点云。")
        return self.cloud.rgb

    def get_full_sem(self) -> np.ndarray:
        if self.cloud is None:
            raise RuntimeError("未加载点云。")
        return self.cloud.sem

    def get_full_inst(self) -> np.ndarray:
        if self.cloud is None:
            raise RuntimeError("未加载点云。")
        return self.cloud.inst

    def list_instance_ids(self) -> np.ndarray:
        if self.cloud is None:
            return np.array([], dtype=np.int64)
        ids = np.unique(self.cloud.inst)
        ids = ids[ids >= 0]
        return ids.astype(np.int64)

    # ---------- instance ----------
    def clear_instance_state(self):
        self.current_inst_id = None
        self.leaf_pts = None
        self.leaf_global_idx = None
        self.ds = None
        self.ds_tree = None

        self.base_idx = None
        self.tip_idx = None
        self.ctrl_indices = []

        self.width_w1_idx = None
        self.width_w2_idx = None
        self._width_recommended_once = False

        self.centerline_result = None
        self.width_path_points = None
        self.width_path_length = None

    def select_instance(self, inst_id: int):
        if self.cloud is None:
            raise RuntimeError("未加载点云。")
        inst_id = int(inst_id)
        mask = (self.cloud.inst == inst_id)
        if not np.any(mask):
            raise ValueError(f"inst_id={inst_id} 不存在。")

        self.current_inst_id = inst_id
        self.leaf_global_idx = np.where(mask)[0].astype(np.int64)
        self.leaf_pts = self.cloud.xyz[mask]

        self.ds = VoxelSampler.voxel_downsample_with_index(self.leaf_pts, self.params.voxel)
        self.ds_tree = cKDTree(self.ds.points)

        self.base_idx = None
        self.tip_idx = None
        self.ctrl_indices = []

        self.width_w1_idx = None
        self.width_w2_idx = None
        self._width_recommended_once = False

        self.centerline_result = None
        self.width_path_points = None
        self.width_path_length = None

        self.restore_picks_from_cache(inst_id)

    def get_ds_points(self) -> np.ndarray:
        if self.ds is None:
            raise RuntimeError("请先选择实例。")
        return self.ds.points

    def get_ds_global_indices(self) -> np.ndarray:
        if self.ds is None or self.leaf_global_idx is None:
            raise RuntimeError("请先选择实例。")
        leaf_local = self.ds.src_indices.astype(np.int64)
        return self.leaf_global_idx[leaf_local].astype(np.int64)

    def snap_to_ds_index(self, point_xyz: np.ndarray) -> int:
        if self.ds_tree is None:
            raise RuntimeError("请先选择一个实例。")
        _, idx = self.ds_tree.query(point_xyz, k=1)
        return int(idx)

    def set_base(self, ds_index: int): self.base_idx = int(ds_index)
    def set_tip(self, ds_index: int): self.tip_idx = int(ds_index)
    def add_ctrl(self, ds_index: int): self.ctrl_indices.append(int(ds_index))
    def extend_ctrl(self, ds_indices: List[int]): self.ctrl_indices.extend([int(i) for i in ds_indices])
    def clear_ctrl(self): self.ctrl_indices = []

    # ---------- leaf length polyline ----------
    def build_length_polyline(self) -> np.ndarray:
        """
        叶长：直接连接 B1 -> ctrl... -> T1
        """
        if self.ds is None:
            raise RuntimeError("请先选择实例。")
        if self.base_idx is None or self.tip_idx is None:
            raise RuntimeError("请先选择叶基(B1)和叶尖(T1)。")

        seq = [int(self.base_idx)]
        seq += [int(i) for i in self.ctrl_indices]
        seq.append(int(self.tip_idx))

        # 去掉连续重复点
        clean = [seq[0]]
        for i in seq[1:]:
            if i != clean[-1]:
                clean.append(i)

        pts = self.ds.points[np.array(clean, dtype=np.int64)]
        return pts

    def compute_centerline(self, use_ctrl: bool = True):
        """
        叶长（中心线）两步：
        1）在 downsample 点云上构建 kNN 图，分段最短路径：
           - use_ctrl=False：B1 -> T1
           - use_ctrl=True： B1 -> C1 -> ... -> T1
        2）对最短路按 step 等距重采样，并在每个 step 内局部“薄片”修正到质心附近的点。
        结果写入 self.centerline_result（含 smooth_points/length/path_indices）。
        """
        if self.ds is None:
            raise RuntimeError("请先选择实例。")
        if self.base_idx is None or self.tip_idx is None:
            raise RuntimeError("请先选择叶基(B1)和叶尖(T1)。")

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
                raise RuntimeError("kNN 图不连通：起点到终点不可达。请增大 k 或减小 voxel。")
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
        L = _polyline_length(refined_poly)
        self.centerline_result = SimpleNamespace(
            smooth_points=refined_poly,
            length=float(L if L is not None else 0.0),
            path_indices=np.asarray(refined_idx, dtype=np.int64),
        )
        return self.centerline_result

    # ---------- width recommendation ----------
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

        # 优先使用已生成的叶长（最短路径/带控制点），否则用直连折线
        if self.centerline_result is not None:
            length_poly = np.asarray(self.centerline_result.smooth_points, dtype=np.float64)
        else:
            length_poly = self.build_length_polyline()

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

    # ---------- compute ----------
    def compute(self):
        """
        - 叶长：优先用 kNN 图最短路径 + step 重采样薄片修正（B1->ctrl...->T1），失败则回退为直连折线
        - 叶宽：若有 W1/W2（用户或推荐），则计算最短路径（kNN 图）
        """
        if self.leaf_pts is None or self.ds is None:
            raise RuntimeError("请先选择一个实例。")
        if self.base_idx is None or self.tip_idx is None:
            raise RuntimeError("请先选择叶基(B1)和叶尖(T1)。")

        try:
            self.compute_centerline(use_ctrl=True)
        except Exception:
            length_poly = self.build_length_polyline()
            L = _polyline_length(length_poly)
            self.centerline_result = SimpleNamespace(
                smooth_points=length_poly,
                length=float(L if L is not None else 0.0),
                path_indices=None,
            )

        # 如果用户没设置 W1/W2，就推荐一次
        self.recommend_width_endpoints(overwrite=False)

        # width shortest path (optional)
        self.width_path_points = None
        self.width_path_length = None

        if self.width_w1_idx is not None and self.width_w2_idx is not None:
            pts = self.ds.points
            G = _build_knn_graph(pts, k=self.params.k)
            pidx = _shortest_path_indices(G, int(self.width_w1_idx), int(self.width_w2_idx))
            if pidx is not None and len(pidx) >= 2:
                path_pts = pts[np.array(pidx, dtype=np.int64)]
                self.width_path_points = path_pts
                self.width_path_length = _polyline_length(path_pts)

    # ---------- cache helpers ----------
    def get_annotated_ids(self) -> List[int]:
        return sorted(self.annotations.keys())

    def get_annotations_count(self) -> int:
        return len(self.annotations)

    def is_current_annotated(self) -> bool:
        return (self.current_inst_id is not None) and (int(self.current_inst_id) in self.annotations)

    def get_cached_display(self, inst_id: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        ann = self.annotations.get(int(inst_id))
        if ann is None:
            return None, None
        cl = np.asarray(ann.get("centerline"), dtype=np.float64) if ann.get("centerline") is not None else None
        wp = np.asarray(ann.get("width_path"), dtype=np.float64) if ann.get("width_path") is not None else None
        return cl, wp

    def restore_picks_from_cache(self, inst_id: int):
        ann = self.annotations.get(int(inst_id))
        if ann is None or self.cloud is None:
            return
        picked = ann.get("picked", {})
        xyz_full = self.cloud.xyz

        def global_to_ds(gidx: int) -> int:
            p = xyz_full[int(gidx)]
            return self.snap_to_ds_index(p)

        if picked.get("base_global") is not None:
            self.base_idx = global_to_ds(picked["base_global"])
        elif picked.get("base_ds") is not None:
            self.base_idx = int(picked["base_ds"])

        if picked.get("tip_global") is not None:
            self.tip_idx = global_to_ds(picked["tip_global"])
        elif picked.get("tip_ds") is not None:
            self.tip_idx = int(picked["tip_ds"])

        ctrl_ds: List[int] = []
        if picked.get("ctrl_global") is not None:
            for g in picked["ctrl_global"]:
                ctrl_ds.append(global_to_ds(g))
        elif picked.get("ctrl_ds") is not None:
            ctrl_ds = [int(i) for i in picked["ctrl_ds"]]
        self.ctrl_indices = ctrl_ds

        if picked.get("w1_global") is not None:
            self.width_w1_idx = global_to_ds(picked["w1_global"])
        elif picked.get("w1_ds") is not None:
            self.width_w1_idx = int(picked["w1_ds"])

        if picked.get("w2_global") is not None:
            self.width_w2_idx = global_to_ds(picked["w2_global"])
        elif picked.get("w2_ds") is not None:
            self.width_w2_idx = int(picked["w2_ds"])

        # 只要是从缓存恢复的，就认为“推荐过/用户已有”，避免自动覆盖
        if (self.width_w1_idx is not None) or (self.width_w2_idx is not None):
            self._width_recommended_once = True

    def commit_current(self, include_width_profile: bool = False):
        if self.current_inst_id is None:
            raise RuntimeError("未选择实例。")
        if self.centerline_result is None:
            raise RuntimeError("当前实例还未计算中心线/宽度。")

        inst_id = int(self.current_inst_id)

        def ds_to_global(ds_idx: int) -> int:
            g = self.get_ds_global_indices()
            return int(g[int(ds_idx)])

        picked: Dict[str, Any] = {
            "base_ds": self.base_idx,
            "tip_ds": self.tip_idx,
            "ctrl_ds": self.ctrl_indices,
            "w1_ds": self.width_w1_idx,
            "w2_ds": self.width_w2_idx,

            "base_global": None if self.base_idx is None else ds_to_global(self.base_idx),
            "tip_global": None if self.tip_idx is None else ds_to_global(self.tip_idx),
            "ctrl_global": [ds_to_global(i) for i in self.ctrl_indices],

            "w1_global": None if self.width_w1_idx is None else ds_to_global(self.width_w1_idx),
            "w2_global": None if self.width_w2_idx is None else ds_to_global(self.width_w2_idx),
        }

        ann: Dict[str, Any] = {
            "inst_id": inst_id,
            "picked": picked,
            "centerline": self.centerline_result.smooth_points.tolist(),
            "length": float(self.centerline_result.length),

            "width_path": None if self.width_path_points is None else self.width_path_points.tolist(),
            "width_path_length": None if self.width_path_length is None else float(self.width_path_length),
        }

        # 兼容字段
        ann["max_width"] = None
        ann["max_width_segment"] = None
        if include_width_profile:
            ann["width_profile"] = []

        self.annotations[inst_id] = ann

    # ---------- export all ----------
    def export_all_json(self, out_path: str):
        if self.cloud is None:
            raise RuntimeError("未加载点云。")
        if len(self.annotations) == 0:
            raise RuntimeError("当前文件还没有任何实例被标注。")

        out = {
            "input": self.file_path,
            "schema": {
                "xyz": ":3",
                "sem_col": self.schema.sem_col,
                "inst_col": self.schema.inst_col,
                "rgb": None if self.schema.rgb_slice is None else f"{self.schema.rgb_slice.start}:{self.schema.rgb_slice.stop}",
            },
            "params": vars(self.params),
            "annotations": [self.annotations[k] for k in sorted(self.annotations.keys())]
        }

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
