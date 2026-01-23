# ui/main_window_parts/semantics.py
import numpy as np
from PyQt5 import QtWidgets


class SemanticMixin:
    def _update_semantic_map_from_ui(self):
        def _val(combo: QtWidgets.QComboBox):
            text = combo.currentText()
            if not text:
                return None
            v = int(text)
            return None if v < 0 else v

        self.session.semantic_map = {
            "leaf": _val(self.combo_sem_filter),
            "stem": _val(self.combo_stem_filter),
            "flower": _val(self.combo_flower_filter),
            "fruit": _val(self.combo_fruit_filter),
        }


    def _refresh_anno_semantic_options(self):
        if not hasattr(self, "combo_anno_semantic"):
            return
        model = self.combo_anno_semantic.model()
        if model is None or not hasattr(model, "item"):
            return
        mapping = {"叶": "leaf", "茎": "stem", "花": "flower", "果": "fruit"}
        enabled_any = False
        for i in range(self.combo_anno_semantic.count()):
            text = self.combo_anno_semantic.itemText(i)
            key = mapping.get(text)
            enabled = True
            if key is not None:
                val = self.session.semantic_map.get(key)
                enabled = val is not None and int(val) >= 0
            item = model.item(i)
            if item is not None:
                item.setEnabled(enabled)
            enabled_any = enabled_any or enabled
        if not enabled_any:
            return
        cur_idx = self.combo_anno_semantic.currentIndex()
        if cur_idx >= 0:
            item = model.item(cur_idx)
            if item is not None and not item.isEnabled():
                for i in range(self.combo_anno_semantic.count()):
                    item = model.item(i)
                    if item is not None and item.isEnabled():
                        self.combo_anno_semantic.setCurrentIndex(i)
                        break


    def _get_annotation_semantic_key(self) -> str:
        text = self.combo_anno_semantic.currentText()
        mapping = {"叶": "leaf", "茎": "stem", "花": "flower", "果": "fruit"}
        return mapping.get(text, "leaf")


    def _refresh_instance_list_for_annotation(self):
        key = self._get_annotation_semantic_key()
        sem_label = self.session.semantic_map.get(key)
        if sem_label is None:
            self.combo_inst.clear()
            self._update_buttons()
            return
        self._refresh_instance_list(target_sem=int(sem_label), keep_current=False)
        self._update_buttons()


    def on_anno_semantic_changed(self):
        key = self._get_annotation_semantic_key()
        self.annotate_semantic = key
        self._settings.setValue("anno_semantic", self.combo_anno_semantic.currentText())
        if not self.annotating and self.session.cloud is not None:
            self._refresh_instance_list_for_annotation()
        self._update_buttons()


    def _get_instance_sem_map(self) -> dict:
        if self.session.cloud is None:
            return {}
        inst = self.session.get_full_inst()
        sem = self.session.get_full_sem()
        mask = inst >= 0
        inst = inst[mask]
        sem = sem[mask]
        if len(inst) == 0:
            return {}
        out = {}
        for inst_id in np.unique(inst):
            vals = sem[inst == inst_id]
            if len(vals) == 0:
                continue
            uvals, counts = np.unique(vals, return_counts=True)
            out[int(inst_id)] = int(uvals[int(np.argmax(counts))])
        return out


    def _refresh_instance_list(self, target_sem: int = None, keep_current: bool = True):
        if self.session.cloud is None:
            return
        sem_map = self._get_instance_sem_map()
        ids = sorted(sem_map.keys())
        if target_sem is not None:
            ids = [i for i in ids if sem_map.get(i) == int(target_sem)]
        cur = self.combo_inst.currentText()
        self.combo_inst.blockSignals(True)
        self.combo_inst.clear()
        for _id in ids:
            self.combo_inst.addItem(str(int(_id)))
        self.combo_inst.blockSignals(False)
        if keep_current and cur:
            if cur in [self.combo_inst.itemText(i) for i in range(self.combo_inst.count())]:
                self.combo_inst.setCurrentText(cur)
                return
        if self.combo_inst.count() > 0:
            self.combo_inst.setCurrentIndex(0)


    def _refresh_sem_filter_options(self):
        if self.session.cloud is None:
            return
        sem_vals = self.session.get_full_sem()
        uniq = np.unique(sem_vals.astype(np.int64))
        uniq = uniq[uniq >= 0]
        prefer_vals = []
        for key in ["leaf", "stem", "flower", "fruit"]:
            val = self.session.semantic_map.get(key)
            if val is None:
                continue
            try:
                v = int(val)
            except Exception:
                continue
            if v >= 0:
                prefer_vals.append(v)
        uniq = np.unique(np.concatenate([uniq, np.asarray(prefer_vals, dtype=np.int64)])).astype(np.int64)
        opts = [-1] + uniq.tolist()
        combos = [
            self.combo_sem_filter,
            self.combo_stem_filter,
            self.combo_flower_filter,
            self.combo_fruit_filter,
        ]
        for combo in combos:
            combo.blockSignals(True)
        try:
            def _fill_combo(combo: QtWidgets.QComboBox, prefer_val, last_key: str):
                combo.clear()
                for v in opts:
                    combo.addItem(str(int(v)))
                if combo.count() == 0:
                    return
                items = [combo.itemText(i) for i in range(combo.count())]
                last_val = self._settings.value(last_key, "", type=str)
                if prefer_val is not None and str(int(prefer_val)) in items:
                    combo.setCurrentText(str(int(prefer_val)))
                elif prefer_val is None and "-1" in items:
                    combo.setCurrentText("-1")
                elif last_val and last_val in items:
                    combo.setCurrentText(last_val)
                else:
                    combo.setCurrentIndex(0)

            _fill_combo(self.combo_sem_filter, self.session.semantic_map.get("leaf"), "leaf_sem_label")
            _fill_combo(self.combo_stem_filter, self.session.semantic_map.get("stem"), "stem_sem_label")
            _fill_combo(self.combo_flower_filter, self.session.semantic_map.get("flower"), "flower_sem_label")
            _fill_combo(self.combo_fruit_filter, self.session.semantic_map.get("fruit"), "fruit_sem_label")
        finally:
            for combo in combos:
                combo.blockSignals(False)

        self._update_semantic_map_from_ui()
        self._refresh_anno_semantic_options()

    def on_sem_filter_changed(self):
        if self.session.cloud is None:
            return
        text = self.combo_sem_filter.currentText()
        if not text:
            return
        self._settings.setValue("leaf_sem_label", text)
        self._update_semantic_map_from_ui()
        self._refresh_anno_semantic_options()
        if self._get_annotation_semantic_key() == "leaf":
            sem_val = int(text)
            if sem_val < 0:
                self.combo_inst.clear()
            else:
                self._refresh_instance_list(target_sem=sem_val, keep_current=False)
        if self.annotating and self.combo_inst.count() > 0:
            self.on_inst_changed()


    def on_stem_sem_changed(self):
        if self.session.cloud is None:
            return
        text = self.combo_stem_filter.currentText()
        if not text:
            return
        self._settings.setValue("stem_sem_label", text)
        self._update_semantic_map_from_ui()
        self._refresh_anno_semantic_options()
        if self._get_annotation_semantic_key() == "stem":
            self._refresh_instance_list_for_annotation()


    def on_flower_sem_changed(self):
        if self.session.cloud is None:
            return
        text = self.combo_flower_filter.currentText()
        if not text:
            return
        self._settings.setValue("flower_sem_label", text)
        self._update_semantic_map_from_ui()
        self._refresh_anno_semantic_options()
        if self._get_annotation_semantic_key() == "flower":
            self._refresh_instance_list_for_annotation()


    def on_fruit_sem_changed(self):
        if self.session.cloud is None:
            return
        text = self.combo_fruit_filter.currentText()
        if not text:
            return
        self._settings.setValue("fruit_sem_label", text)
        self._update_semantic_map_from_ui()
        self._refresh_anno_semantic_options()
        if self._get_annotation_semantic_key() == "fruit":
            self._refresh_instance_list_for_annotation()
