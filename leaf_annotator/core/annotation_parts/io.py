# core/annotation_parts/io.py
import csv
import json
import os
from typing import Dict, Any

import numpy as np

from core.io import PointCloudIO
from core.schema import CloudParser


class IoMixin:
    def load(self, path: str):
        arr = PointCloudIO.load_array(path)
        self.cloud = CloudParser.parse(arr, self.schema)
        self.file_path = path
        self.annotations = {}
        self.instance_meta = {}
        self.full_point_labels = None
        self.clear_instance_state()


    def load_annotations_json(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "plant_type" in data:
            self.plant_type = str(data.get("plant_type") or self.plant_type)
        if "semantic_map" in data and isinstance(data["semantic_map"], dict):
            for k in ["leaf", "stem", "flower", "fruit"]:
                if k in data["semantic_map"]:
                    v = data["semantic_map"].get(k)
                    if v is None:
                        self.semantic_map[k] = None
                    else:
                        vv = int(v)
                        self.semantic_map[k] = None if vv < 0 else vv
        if "semantic_label_names" in data and isinstance(data["semantic_label_names"], dict):
            for label_str, name in data["semantic_label_names"].items():
                if not isinstance(name, str):
                    continue
                key = name.strip()
                if key not in self.semantic_map:
                    continue
                if self.semantic_map.get(key) is not None:
                    continue
                try:
                    label_val = int(label_str)
                except Exception:
                    continue
                if label_val >= 0:
                    self.semantic_map[key] = label_val
        if "params" in data and isinstance(data["params"], dict):
            self._apply_params_from_dict(data["params"])
        ann_list = data.get("annotations", [])
        self.annotations = {}
        self.instance_meta = {}
        for ann in ann_list:
            inst_id = int(ann.get("inst_id"))
            self.annotations[inst_id] = ann
            meta = {}
            if "remark" in ann:
                meta["remark"] = str(ann.get("remark") or "")
            if "label_desc" in ann:
                meta["label_desc"] = str(ann.get("label_desc") or "")
            if meta:
                self.instance_meta[inst_id] = meta


    def _apply_params_from_dict(self, params: Dict[str, Any]):
        if not isinstance(params, dict):
            return
        for key, value in params.items():
            if not hasattr(self.params, key):
                continue
            current = getattr(self.params, key)
            try:
                if isinstance(current, bool):
                    casted = bool(value)
                elif isinstance(current, int) and not isinstance(current, bool):
                    casted = int(value)
                elif isinstance(current, float):
                    casted = float(value)
                else:
                    casted = value
            except Exception:
                casted = value
            setattr(self.params, key, casted)

        if "stem_diameter_step" in params or "stem_length_step" in params:
            if "stem_step" not in params:
                legacy_src = params.get("stem_diameter_step", params.get("stem_length_step"))
                try:
                    legacy = float(legacy_src)
                except Exception:
                    legacy = None
                if legacy is not None:
                    self.params.stem_step = legacy
        if "stem_segments" in params:
            try:
                legacy = int(params["stem_segments"])
            except Exception:
                legacy = None
            if legacy is not None and legacy > 0:
                if "stem_diameter_segments" not in params:
                    self.params.stem_diameter_segments = legacy
                if "stem_length_segments" not in params:
                    self.params.stem_length_segments = legacy
        if "stem_percentile" in params:
            try:
                legacy = float(params["stem_percentile"])
            except Exception:
                legacy = None
            if legacy is not None:
                if "stem_diameter_percentile" not in params:
                    self.params.stem_diameter_percentile = legacy
                if "stem_length_percentile" not in params:
                    self.params.stem_length_percentile = legacy


    def export_all_json(self, out_path: str):
        self._require_cloud()
        ann_list = self._build_export_annotations()
        if len(ann_list) == 0:
            raise RuntimeError("当前文件还没有任何实例被标注。")

        semantic_map = {}
        semantic_label_names = {}
        for k in ["leaf", "stem", "flower", "fruit"]:
            v = self.semantic_map.get(k)
            semantic_map[k] = -1 if v is None else int(v)
            if v is not None:
                semantic_label_names[str(int(v))] = k

        out = {
            "input": self.file_path,
            "plant_type": self.plant_type,
            "semantic_map": semantic_map,
            "semantic_label_names": semantic_label_names,
            "schema": {
                "xyz": ":3",
                "sem_col": self.schema.sem_col,
                "inst_col": self.schema.inst_col,
                "rgb": None if self.schema.rgb_slice is None else f"{self.schema.rgb_slice.start}:{self.schema.rgb_slice.stop}",
            },
            "params": vars(self.params),
            "annotations": ann_list
        }

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


    def export_labeled_point_cloud(self, out_path: str):
        self._require_cloud()
        if self.point_labels is None or self.leaf_global_idx is None:
            raise RuntimeError("请先生成点标签。")
        full_labels = np.full((len(self.cloud.xyz),), -1, dtype=np.int64)
        if len(self.point_labels) == len(self.leaf_global_idx):
            full_labels[self.leaf_global_idx] = self.point_labels
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        np.savetxt(out_path, full_labels, fmt="%d")

    def export_phenotype_csv(self, out_path: str):
        self._require_cloud()
        sem_map = self._get_instance_sem_map()
        label_to_name = {}
        for key, name in [("leaf", "叶子"), ("stem", "茎"), ("flower", "花"), ("fruit", "果")]:
            v = self.semantic_map.get(key)
            if v is not None:
                label_to_name[int(v)] = name

        rows = []
        for inst_id, ann in self.annotations.items():
            sem_label = sem_map.get(int(inst_id), None)
            sem_text = "-" if sem_label is None else label_to_name.get(int(sem_label), str(int(sem_label)))
            if "length" in ann and ann.get("length") is not None:
                rows.append([inst_id, sem_text, "叶长", f"{float(ann['length']):.3f}"])
            if "width_path_length" in ann and ann.get("width_path_length") is not None:
                rows.append([inst_id, sem_text, "叶宽", f"{float(ann['width_path_length']):.3f}"])
            if "leaf_area" in ann and ann.get("leaf_area") is not None:
                rows.append([inst_id, sem_text, "叶面积", f"{float(ann['leaf_area']):.3f}"])
            if "leaf_projected_area" in ann and ann.get("leaf_projected_area") is not None:
                rows.append([inst_id, sem_text, "投影面积", f"{float(ann['leaf_projected_area']):.3f}"])
            if "leaf_inclination" in ann and ann.get("leaf_inclination") is not None:
                rows.append([inst_id, sem_text, "叶倾角", f"{float(ann['leaf_inclination']):.1f}"])
            if "leaf_stem_angle" in ann and ann.get("leaf_stem_angle") is not None:
                rows.append([inst_id, sem_text, "叶夹角", f"{float(ann['leaf_stem_angle']):.1f}"])
            if "stem_diameter" in ann and ann.get("stem_diameter") is not None:
                rows.append([inst_id, sem_text, "茎粗", f"{float(ann['stem_diameter']):.3f}"])
            if "stem_length" in ann and ann.get("stem_length") is not None:
                rows.append([inst_id, sem_text, "茎长", f"{float(ann['stem_length']):.3f}"])
            if "flower_obb" in ann and isinstance(ann.get("flower_obb"), dict):
                lengths = ann["flower_obb"].get("lengths")
                if lengths is not None and len(lengths) == 3:
                    rows.append([inst_id, sem_text, "花OBB", f"{lengths[0]:.3f},{lengths[1]:.3f},{lengths[2]:.3f}"])
            if "fruit_obb" in ann and isinstance(ann.get("fruit_obb"), dict):
                lengths = ann["fruit_obb"].get("lengths")
                if lengths is not None and len(lengths) == 3:
                    rows.append([inst_id, sem_text, "果OBB", f"{lengths[0]:.3f},{lengths[1]:.3f},{lengths[2]:.3f}"])

        if not rows:
            raise RuntimeError("当前没有可导出的表型结果。")

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["实例", "语义", "表型名字", "表型值"])
            writer.writerows(rows)
