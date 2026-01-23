# core/annotation_parts/cache.py
from typing import List, Optional, Dict, Any, Tuple
from types import SimpleNamespace

import numpy as np

from core.annotation_parts.utils import _polyline_length


class CacheMixin:
    def _ensure_annotation_entry(self, inst_id: int) -> Dict[str, Any]:
        inst_id = int(inst_id)
        ann = self.annotations.get(inst_id)
        if ann is None:
            ann = {"inst_id": inst_id}
            self.annotations[inst_id] = ann
        return ann


    def get_annotated_ids(self) -> List[int]:
        return sorted(self.annotations.keys())


    def get_annotations_count(self) -> int:
        return len(self.annotations)


    def is_current_annotated(self) -> bool:
        return (self.current_inst_id is not None) and (int(self.current_inst_id) in self.annotations)


    def _build_export_annotations(self) -> List[Dict[str, Any]]:
        keys = set(self.annotations.keys()) | set(self.instance_meta.keys())
        if not keys:
            return []
        out: List[Dict[str, Any]] = []
        for inst_id in sorted(keys):
            if inst_id in self.annotations:
                out.append(self.annotations[inst_id])
                continue
            meta = self.instance_meta.get(inst_id, {})
            ann: Dict[str, Any] = {"inst_id": int(inst_id)}
            if "remark" in meta:
                ann["remark"] = str(meta.get("remark") or "")
            if "label_desc" in meta:
                ann["label_desc"] = str(meta.get("label_desc") or "")
            out.append(ann)
        return out


    def get_instance_meta(self, inst_id: int) -> Tuple[str, str]:
        meta = self.instance_meta.get(int(inst_id), {})
        remark = str(meta.get("remark") or "")
        label_desc = str(meta.get("label_desc") or "")
        return remark, label_desc


    def set_instance_meta(self, inst_id: int, remark: str, label_desc: str):
        inst_id = int(inst_id)
        meta = self.instance_meta.setdefault(inst_id, {})
        meta["remark"] = "" if remark is None else str(remark)
        meta["label_desc"] = "" if label_desc is None else str(label_desc)
        ann = self.annotations.get(inst_id)
        if ann is not None:
            ann["remark"] = meta["remark"]
            ann["label_desc"] = meta["label_desc"]


    def get_cached_display(self, inst_id: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.use_cached_results:
            return None, None
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

        if picked.get("base_ds") is not None:
            self.base_idx = int(picked["base_ds"])
        elif picked.get("base_global") is not None:
            self.base_idx = global_to_ds(picked["base_global"])

        if picked.get("tip_ds") is not None:
            self.tip_idx = int(picked["tip_ds"])
        elif picked.get("tip_global") is not None:
            self.tip_idx = global_to_ds(picked["tip_global"])

        ctrl_ds: List[int] = []
        if picked.get("ctrl_ds") is not None:
            ctrl_ds = [int(i) for i in picked["ctrl_ds"]]
        elif picked.get("ctrl_global") is not None:
            for g in picked["ctrl_global"]:
                ctrl_ds.append(global_to_ds(g))
        self.ctrl_indices = ctrl_ds
        self.ctrl_ids = [int(i) for i in ann.get("ctrl_ids", list(range(1, len(ctrl_ds) + 1)))]
        self._next_ctrl_id = ((max(self.ctrl_ids, default=0) // 10) + 1) * 10

        wctrl_ds: List[int] = []
        if picked.get("wctrl_ds") is not None:
            wctrl_ds = [int(i) for i in picked["wctrl_ds"]]
        elif picked.get("wctrl_global") is not None:
            for g in picked["wctrl_global"]:
                wctrl_ds.append(global_to_ds(g))
        self.width_ctrl_indices = wctrl_ds
        self.width_ctrl_ids = [int(i) for i in ann.get("wctrl_ids", list(range(1, len(wctrl_ds) + 1)))]
        self._next_width_ctrl_id = ((max(self.width_ctrl_ids, default=0) // 10) + 1) * 10

        if picked.get("w1_ds") is not None:
            self.width_w1_idx = int(picked["w1_ds"])
        elif picked.get("w1_global") is not None:
            self.width_w1_idx = global_to_ds(picked["w1_global"])

        if picked.get("w2_ds") is not None:
            self.width_w2_idx = int(picked["w2_ds"])
        elif picked.get("w2_global") is not None:
            self.width_w2_idx = global_to_ds(picked["w2_global"])

        if self.use_cached_results:
            cl = ann.get("centerline")
            if cl is not None:
                cl_arr = np.asarray(cl, dtype=np.float64)
                L = _polyline_length(cl_arr)
                self.centerline_result = SimpleNamespace(
                    smooth_points=cl_arr,
                    length=float(L if L is not None else 0.0),
                    path_indices=None,
                )
                self.centerline_source = ann.get("centerline_source") or "polyline"

            wp = ann.get("width_path")
            if wp is not None:
                wp_arr = np.asarray(wp, dtype=np.float64)
                self.width_path_points = wp_arr
                self.width_path_length = _polyline_length(wp_arr)
                self.width_path_indices = None

            labels = ann.get("point_labels")
            if labels is not None and self.leaf_pts is not None:
                labels_arr = np.asarray(labels, dtype=np.int64)
                if len(labels_arr) == len(self.leaf_pts):
                    self.point_labels = labels_arr

        # 只要是从缓存恢复的，就认为“推荐过/用户已有”，避免自动覆盖
        if (self.width_w1_idx is not None) or (self.width_w2_idx is not None):
            self._width_recommended_once = True


    def commit_current(self, include_width_profile: bool = False):
        if self.current_inst_id is None:
            raise RuntimeError("请先选择实例。")

        inst_id = int(self.current_inst_id)
        prev = self.annotations.get(inst_id, {})
        meta = self.instance_meta.get(inst_id, {})
        remark = meta.get("remark", prev.get("remark", ""))
        label_desc = meta.get("label_desc", prev.get("label_desc", ""))

        def ds_to_global(ds_idx: int) -> int:
            g = self.get_ds_global_indices()
            return int(g[int(ds_idx)])

        picked: Dict[str, Any] = {
            "base_ds": self.base_idx,
            "tip_ds": self.tip_idx,
            "ctrl_ds": self.ctrl_indices,
            "wctrl_ds": self.width_ctrl_indices,
            "w1_ds": self.width_w1_idx,
            "w2_ds": self.width_w2_idx,

            "base_global": None if self.base_idx is None else ds_to_global(self.base_idx),
            "tip_global": None if self.tip_idx is None else ds_to_global(self.tip_idx),
            "ctrl_global": [ds_to_global(i) for i in self.ctrl_indices],
            "wctrl_global": [ds_to_global(i) for i in self.width_ctrl_indices],

            "w1_global": None if self.width_w1_idx is None else ds_to_global(self.width_w1_idx),
            "w2_global": None if self.width_w2_idx is None else ds_to_global(self.width_w2_idx),
        }

        ann: Dict[str, Any] = dict(prev)
        ann.update({
            "inst_id": inst_id,
            "picked": picked,
            "ctrl_ids": self.ctrl_ids,
            "wctrl_ids": self.width_ctrl_ids,
            "width_path": None if self.width_path_points is None else self.width_path_points.tolist(),
            "width_path_length": None if self.width_path_length is None else float(self.width_path_length),
            "remark": remark,
            "label_desc": label_desc,
        })
        if self.centerline_result is not None:
            ann["centerline"] = self.centerline_result.smooth_points.tolist()
            ann["length"] = float(self.centerline_result.length)
            ann["centerline_source"] = self.centerline_source
        if self.point_labels is not None:
            ann["point_labels"] = self.point_labels.tolist()
            ann["point_label_global_indices"] = None if self.leaf_global_idx is None else self.leaf_global_idx.tolist()

        # 兼容字段
        ann["max_width"] = None
        ann["max_width_segment"] = None
        if include_width_profile:
            ann["width_profile"] = []

        self.annotations[inst_id] = ann
        self.instance_meta[inst_id] = {"remark": remark, "label_desc": label_desc}

    # ---------- export all ----------
