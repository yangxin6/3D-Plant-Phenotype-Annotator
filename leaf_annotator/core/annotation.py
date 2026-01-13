import json
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from scipy.spatial import cKDTree

from core.io import PointCloudIO
from core.sampling import VoxelSampler
from core.centerline import CenterlineExtractor, CenterlineResult
from core.width import WidthEstimator, WidthResult


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
    def __init__(self, params: Optional[AnnotationParams] = None):
        self.params = params or AnnotationParams()

        self.file_path: Optional[str] = None
        self.leaf_pts: Optional[np.ndarray] = None
        self.ds_pts: Optional[np.ndarray] = None
        self.ds_tree: Optional[cKDTree] = None

        self.base_idx: Optional[int] = None
        self.tip_idx: Optional[int] = None
        self.ctrl_indices: List[int] = []

        self.centerline_result: Optional[CenterlineResult] = None
        self.width_result: Optional[WidthResult] = None

    def load(self, path: str):
        pts = PointCloudIO.load_points(path)
        ds = VoxelSampler.voxel_downsample(pts, self.params.voxel)
        self.file_path = path
        self.leaf_pts = pts
        self.ds_pts = ds
        self.ds_tree = cKDTree(ds)

        self.base_idx = None
        self.tip_idx = None
        self.ctrl_indices = []
        self.centerline_result = None
        self.width_result = None

    def snap_to_ds_index(self, point_xyz: np.ndarray) -> int:
        if self.ds_tree is None:
            raise RuntimeError("未加载点云。")
        _, idx = self.ds_tree.query(point_xyz, k=1)
        return int(idx)

    def set_base(self, ds_index: int):
        self.base_idx = int(ds_index)

    def set_tip(self, ds_index: int):
        self.tip_idx = int(ds_index)

    def add_ctrl(self, ds_index: int):
        self.ctrl_indices.append(int(ds_index))

    def clear_ctrl(self):
        self.ctrl_indices = []

    def compute(self):
        if self.ds_pts is None or self.leaf_pts is None:
            raise RuntimeError("未加载点云。")
        if self.base_idx is None or self.tip_idx is None:
            raise RuntimeError("请先选择叶基和叶尖。")

        cl = CenterlineExtractor(k=self.params.k, smooth_win=self.params.smooth_win)
        self.centerline_result = cl.extract(
            ds_points=self.ds_pts,
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

    def export_json(self, out_path: str):
        if self.centerline_result is None:
            raise RuntimeError("尚未计算中心线/宽度。")

        wr = self.width_result
        max_item = None if (wr is None or wr.max_item is None) else wr.max_item

        out = {
            "input": self.file_path,
            "params": vars(self.params),
            "picked": {
                "base_idx_ds": self.base_idx,
                "tip_idx_ds": self.tip_idx,
                "ctrl_idx_ds": self.ctrl_indices,
                "base_point": None if self.base_idx is None else self.ds_pts[self.base_idx].tolist(),
                "tip_point": None if self.tip_idx is None else self.ds_pts[self.tip_idx].tolist(),
            },
            "centerline": self.centerline_result.smooth_points.tolist(),
            "length": self.centerline_result.length,
            "width_profile": [] if (wr is None) else [
                {
                    "s_index": it.s_index,
                    "p": it.p.tolist(),
                    "width": it.width,
                    "pL": it.pL.tolist(),
                    "pR": it.pR.tolist(),
                    "slice_n": it.slice_n
                } for it in wr.profile
            ],
            "max_width": None if max_item is None else max_item.width,
            "max_width_segment": None if max_item is None else {
                "pL": max_item.pL.tolist(),
                "pR": max_item.pR.tolist()
            }
        }

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
