import json
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

from scipy.spatial import cKDTree

from core.io import PointCloudIO
from core.schema import CloudSchema, CloudParser, ParsedCloud
from core.sampling import VoxelSampler, Downsampled
from core.centerline import CenterlineExtractor
from core.width import WidthEstimator


@dataclass
class AnnotationParams:
    voxel: float = 0.003
    k: int = 25
    smooth_win: int = 9
    step: float = 0.01
    slab_half: float = 0.005
    radius: float = 0.10
    min_slice_pts: int = 60


class LeafAnnotationSession:
    """
    - 整株加载，浏览 RGB/语义/实例
    - 标注：选 inst -> 只显示该实例
    - 点选逻辑：base/tip 单个；ctrl 多个
    - 缓存每个 inst 的标注结果，并支持切换 inst 自动显示中心线/宽线 + 恢复 base/tip/ctrl
    - 整株一次性导出 annotations[]
    """
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

        # base/tip 单个，ctrl 多个（这些是“已保存点”）
        self.base_idx: Optional[int] = None
        self.tip_idx: Optional[int] = None
        self.ctrl_indices: List[int] = []

        self.centerline_result = None
        self.width_result = None

        # inst_id -> annotation dict
        self.annotations: Dict[int, Dict[str, Any]] = {}

    # ---------- full cloud ----------
    def load(self, path: str):
        arr = PointCloudIO.load_array(path)
        self.cloud = CloudParser.parse(arr, self.schema)
        self.file_path = path

        # 新文件：清空缓存
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

        self.centerline_result = None
        self.width_result = None

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

        # 切换实例：先清空当前已保存点/结果
        self.base_idx = None
        self.tip_idx = None
        self.ctrl_indices = []
        self.centerline_result = None
        self.width_result = None

        # 如果该 inst 已经标注过：恢复 base/tip/ctrl（用于继续编辑）
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

    # ---------- compute ----------
    def compute(self):
        if self.leaf_pts is None or self.ds is None:
            raise RuntimeError("请先选择一个实例。")
        if self.base_idx is None or self.tip_idx is None:
            raise RuntimeError("请先选择叶基和叶尖。")

        cl = CenterlineExtractor(k=self.params.k, smooth_win=self.params.smooth_win)
        self.centerline_result = cl.extract(
            ds_points=self.ds.points,
            base_idx=self.base_idx,
            tip_idx=self.tip_idx,
            ctrl_indices=self.ctrl_indices
        )

        we = WidthEstimator(
            step=self.params.step,
            slab_half=self.params.slab_half,
            radius=self.params.radius,
            min_slice_pts=self.params.min_slice_pts
        )
        self.width_result = we.compute(self.leaf_pts, self.centerline_result.smooth_points)

    # ---------- cache helpers ----------
    def get_annotated_ids(self) -> List[int]:
        return sorted(self.annotations.keys())

    def get_annotations_count(self) -> int:
        return len(self.annotations)

    def is_current_annotated(self) -> bool:
        return (self.current_inst_id is not None) and (int(self.current_inst_id) in self.annotations)

    def get_cached_display(self, inst_id: int) -> Tuple[Optional[np.ndarray], Optional[Tuple[np.ndarray, np.ndarray]]]:
        ann = self.annotations.get(int(inst_id))
        if ann is None:
            return None, None
        cl = np.asarray(ann.get("centerline"), dtype=np.float64) if ann.get("centerline") is not None else None
        seg = ann.get("max_width_segment")
        if seg is None:
            return cl, None
        pL = np.asarray(seg["pL"], dtype=np.float64)
        pR = np.asarray(seg["pR"], dtype=np.float64)
        return cl, (pL, pR)

    def restore_picks_from_cache(self, inst_id: int):
        """
        恢复该 inst 已保存的 base/tip/ctrl。
        优先使用 global 索引映射回当前 ds（更稳）；没有 global 才退回 ds 索引。
        """
        ann = self.annotations.get(int(inst_id))
        if ann is None or self.cloud is None:
            return
        picked = ann.get("picked", {})
        xyz_full = self.cloud.xyz

        def global_to_ds(gidx: int) -> int:
            p = xyz_full[int(gidx)]
            return self.snap_to_ds_index(p)

        # base
        if picked.get("base_global") is not None:
            self.base_idx = global_to_ds(picked["base_global"])
        elif picked.get("base_ds") is not None:
            self.base_idx = int(picked["base_ds"])

        # tip
        if picked.get("tip_global") is not None:
            self.tip_idx = global_to_ds(picked["tip_global"])
        elif picked.get("tip_ds") is not None:
            self.tip_idx = int(picked["tip_ds"])

        # ctrl (many)
        ctrl_ds = []
        if picked.get("ctrl_global") is not None:
            for g in picked["ctrl_global"]:
                ctrl_ds.append(global_to_ds(g))
        elif picked.get("ctrl_ds") is not None:
            ctrl_ds = [int(i) for i in picked["ctrl_ds"]]
        self.ctrl_indices = ctrl_ds

    def commit_current(self, include_width_profile: bool = False):
        """
        把当前实例的“已保存点 + 计算结果”写入缓存（覆盖该 inst）。
        """
        if self.current_inst_id is None:
            raise RuntimeError("未选择实例。")
        if self.centerline_result is None:
            raise RuntimeError("当前实例还未计算中心线/宽度。")

        inst_id = int(self.current_inst_id)

        def ds_to_global(ds_idx: int) -> int:
            g = self.get_ds_global_indices()
            return int(g[int(ds_idx)])

        max_item = None
        if self.width_result is not None and self.width_result.max_item is not None:
            max_item = self.width_result.max_item

        ann: Dict[str, Any] = {
            "inst_id": inst_id,
            "picked": {
                "base_ds": self.base_idx,
                "tip_ds": self.tip_idx,
                "ctrl_ds": self.ctrl_indices,
                "base_global": None if self.base_idx is None else ds_to_global(self.base_idx),
                "tip_global": None if self.tip_idx is None else ds_to_global(self.tip_idx),
                "ctrl_global": [ds_to_global(i) for i in self.ctrl_indices],
            },
            "centerline": self.centerline_result.smooth_points.tolist(),
            "length": float(self.centerline_result.length),
            "max_width": None if max_item is None else float(max_item.width),
            "max_width_segment": None if max_item is None else {
                "pL": max_item.pL.tolist(),
                "pR": max_item.pR.tolist()
            }
        }

        if include_width_profile:
            wr = self.width_result
            ann["width_profile"] = [] if (wr is None) else [
                {
                    "s_index": int(it.s_index),
                    "p": it.p.tolist(),
                    "width": float(it.width),
                    "pL": it.pL.tolist(),
                    "pR": it.pR.tolist(),
                    "slice_n": int(it.slice_n)
                } for it in wr.profile
            ]

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
