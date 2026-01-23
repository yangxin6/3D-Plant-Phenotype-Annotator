# core/annotation_parts/base.py
from typing import Optional

import numpy as np


class BaseMixin:
    def _require_cloud(self):
        if self.cloud is None:
            raise RuntimeError("请先加载点云文件。")


    def _require_instance(self):
        if self.ds is None or self.leaf_pts is None:
            raise RuntimeError("请先选择实例。")


    def _require_base_tip(self):
        if self.base_idx is None or self.tip_idx is None:
            raise RuntimeError("请先选择叶基(B1)和叶尖(T1)。")


    def _require_width_endpoints(self):
        if self.width_w1_idx is None or self.width_w2_idx is None:
            raise RuntimeError("请先选择叶宽端点 W1/W2。")

    # ---------- full cloud ----------

    def has_rgb(self) -> bool:
        return (self.cloud is not None) and (self.cloud.rgb is not None)


    def get_full_xyz(self) -> np.ndarray:
        self._require_cloud()
        return self.cloud.xyz


    def get_full_rgb(self) -> Optional[np.ndarray]:
        self._require_cloud()
        return self.cloud.rgb


    def get_full_sem(self) -> np.ndarray:
        self._require_cloud()
        return self.cloud.sem


    def get_full_inst(self) -> np.ndarray:
        self._require_cloud()
        return self.cloud.inst


    def get_instance_points(self, inst_id: int) -> np.ndarray:
        self._require_cloud()
        mask = (self.cloud.inst == int(inst_id))
        return self.cloud.xyz[mask]


    def list_instance_ids(self) -> np.ndarray:
        if self.cloud is None:
            return np.array([], dtype=np.int64)
        ids = np.unique(self.cloud.inst)
        ids = ids[ids >= 0]
        return ids.astype(np.int64)

    # ---------- instance ----------
