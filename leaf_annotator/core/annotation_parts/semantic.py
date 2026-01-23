# core/annotation_parts/semantic.py
from typing import Dict

import numpy as np


class SemanticMixin:
    def _get_instance_sem_map(self) -> Dict[int, int]:
        self._require_cloud()
        inst = self.cloud.inst
        sem = self.cloud.sem
        mask = inst >= 0
        inst = inst[mask]
        sem = sem[mask]
        out: Dict[int, int] = {}
        for inst_id in np.unique(inst):
            vals = sem[inst == inst_id]
            if len(vals) == 0:
                continue
            uvals, counts = np.unique(vals, return_counts=True)
            out[int(inst_id)] = int(uvals[int(np.argmax(counts))])
        return out


    def compute_semantic_structures(self) -> Dict[str, int]:
        self._require_cloud()
        sem_map = self._get_instance_sem_map()
        stem_label = self.semantic_map.get("stem")
        flower_label = self.semantic_map.get("flower")
        fruit_label = self.semantic_map.get("fruit")
        counts = {"stem": 0, "flower": 0, "fruit": 0}
        stem_pct = self._get_stem_diameter_percentile()
        length_pct = self._get_stem_length_percentile()
        stem_segments = getattr(self.params, "stem_diameter_segments", 0)
        length_segments = getattr(self.params, "stem_length_segments", 0)

        for inst_id, sem_label in sem_map.items():
            pts = self.get_instance_points(inst_id)
            if stem_label is not None and sem_label == int(stem_label):
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
                    counts["stem"] += 1
            if flower_label is not None and sem_label == int(flower_label):
                obb = self._compute_obb_from_points(pts)
                if obb is not None:
                    ann = self._ensure_annotation_entry(inst_id)
                    ann["flower_obb"] = obb
                    counts["flower"] += 1
            if fruit_label is not None and sem_label == int(fruit_label):
                obb = self._compute_obb_from_points(pts)
                if obb is not None:
                    ann = self._ensure_annotation_entry(inst_id)
                    ann["fruit_obb"] = obb
                    counts["fruit"] += 1

        return counts
