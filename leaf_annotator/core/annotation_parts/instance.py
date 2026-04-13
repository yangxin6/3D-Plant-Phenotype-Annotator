# core/annotation_parts/instance.py
from typing import List
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from core.sampling import VoxelSampler


class InstanceMixin:
    def clear_instance_state(self):
        self.current_inst_id = None
        self.leaf_pts = None
        self.leaf_global_idx = None
        self.ds = None
        self.ds_tree = None

        self.base_idx = None
        self.tip_idx = None
        self.ctrl_indices = []
        self.ctrl_ids = []
        self._next_ctrl_id = 10
        self.width_ctrl_indices = []
        self.width_ctrl_ids = []
        self._next_width_ctrl_id = 10

        self.width_w1_idx = None
        self.width_w2_idx = None
        self._width_recommended_once = False

        self.centerline_result = None
        self.centerline_source = None
        self.width_path_points = None
        self.width_path_length = None
        self.width_path_indices = None
        self.point_labels = None


    def select_instance(self, inst_id: int):
        self._require_cloud()
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
        self.ctrl_ids = []
        self._next_ctrl_id = 10
        self.width_ctrl_indices = []
        self.width_ctrl_ids = []
        self._next_width_ctrl_id = 10

        self.width_w1_idx = None
        self.width_w2_idx = None
        self._width_recommended_once = False

        self.centerline_result = None
        self.centerline_source = None
        self.width_path_points = None
        self.width_path_length = None
        self.width_path_indices = None
        self.point_labels = None

        self.restore_picks_from_cache(inst_id)
        if self.full_point_labels is not None and self.leaf_global_idx is not None:
            if len(self.full_point_labels) == len(self.cloud.xyz):
                self.point_labels = self.full_point_labels[self.leaf_global_idx]


    def get_ds_points(self) -> np.ndarray:
        self._require_instance()
        return self.ds.points


    def get_ds_global_indices(self) -> np.ndarray:
        self._require_instance()
        if self.leaf_global_idx is None:
            raise RuntimeError("请先选择实例。")
        leaf_local = self.ds.src_indices.astype(np.int64)
        return self.leaf_global_idx[leaf_local].astype(np.int64)


    def snap_to_ds_index(self, point_xyz: np.ndarray) -> int:
        if self.ds_tree is None:
            raise RuntimeError("请先选择实例。")
        _, idx = self.ds_tree.query(point_xyz, k=1)
        return int(idx)


    def set_base(self, ds_index: int): self.base_idx = int(ds_index)

    def set_tip(self, ds_index: int): self.tip_idx = int(ds_index)

    def add_ctrl(self, ds_index: int):
        self.ctrl_indices.append(int(ds_index))
        self.ctrl_ids.append(self._next_ctrl_id)
        self._next_ctrl_id += 10


    def extend_ctrl(self, ds_indices: List[int]):
        for i in ds_indices:
            self.add_ctrl(int(i))


    def clear_ctrl(self):
        self.ctrl_indices = []
        self.ctrl_ids = []
        self._next_ctrl_id = 10


    def add_width_ctrl(self, ds_index: int):
        self.width_ctrl_indices.append(int(ds_index))
        self.width_ctrl_ids.append(self._next_width_ctrl_id)
        self._next_width_ctrl_id += 10


    def extend_width_ctrl(self, ds_indices: List[int]):
        for i in ds_indices:
            self.add_width_ctrl(int(i))


    def clear_width_ctrl(self):
        self.width_ctrl_indices = []
        self.width_ctrl_ids = []
        self._next_width_ctrl_id = 10


    def get_sorted_ctrl_indices(self) -> List[int]:
        pairs = list(zip(self.ctrl_ids, self.ctrl_indices))
        pairs.sort(key=lambda x: x[0])
        return [idx for _, idx in pairs]


    def get_sorted_width_ctrl_indices(self) -> List[int]:
        pairs = list(zip(self.width_ctrl_ids, self.width_ctrl_indices))
        pairs.sort(key=lambda x: x[0])
        return [idx for _, idx in pairs]


    def _set_centerline_result(self, result: SimpleNamespace, source: str):
        if result is None:
            return
        if source == "polyline":
            self.centerline_result = result
            self.centerline_source = "polyline"
            return
        if source == "recommended":
            if self.centerline_source == "polyline":
                return
            self.centerline_result = result
            self.centerline_source = "recommended"

    # ---------- leaf length polyline ----------

    def build_length_polyline(self, use_ctrl: bool = True) -> np.ndarray:
        """
        叶长：直接连接 B1 -> ctrl... -> T1
        """
        self._require_instance()
        self._require_base_tip()

        seq = [int(self.base_idx)]
        if use_ctrl and len(self.ctrl_indices) > 0:
            seq += [int(i) for i in self.get_sorted_ctrl_indices()]
        seq.append(int(self.tip_idx))

        # 去掉连续重复点
        clean = [seq[0]]
        for i in seq[1:]:
            if i != clean[-1]:
                clean.append(i)

        pts = self.ds.points[np.array(clean, dtype=np.int64)]
        return pts
