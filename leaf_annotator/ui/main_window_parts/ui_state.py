# ui/main_window_parts/ui_state.py
import os
from typing import Optional

import numpy as np
from PyQt5 import QtWidgets, QtCore

from .constants import (
    PANEL_WIDTH_DEFAULT,
    PANEL_WIDTH_LEAF_ANNOTATE,
    PANEL_INNER_PADDING,
    PANEL_ROW_LABEL_SPACE,
    PANEL_COMBO_MIN_WIDTH,
    PANEL_LIST_MIN_WIDTH,
    PANEL_BUTTON_MIN_WIDTH,
)


class UiStateMixin:
    def _format_annotated_summary(self) -> str:
        ids = self.session.get_annotated_ids()
        n = len(ids)
        if n == 0:
            return "已标注实例数：0\n已标注 inst：无"
        head = ids[:30]
        tail = "" if n <= 30 else f" ...(+{n-30})"
        return f"已标注实例数：{n}\n已标注 inst：{', '.join(map(str, head))}{tail}"


    def _update_status(self, extra: str = ""):
        base = []
        if self.session.cloud is None:
            base.append("状态：未加载")
        else:
            has_rgb = "有" if self.session.has_rgb() else "无"
            base.append(f"文件：{os.path.basename(self.session.file_path)}")
            base.append(f"总点数：{len(self.session.get_full_xyz())} | RGB：{has_rgb}")
            base.append(self._format_annotated_summary())
            if self.annotating and self.session.current_inst_id is not None:
                base.append(f"标注模式：inst_id={self.session.current_inst_id} | pick={self.pick_mode}")
        if extra:
            base.append(extra)
        self.info.setText("\n".join(base))


    def _start_busy_dialog(self, text: str):
        dlg = QtWidgets.QProgressDialog(text, None, 0, 0, self)
        dlg.setWindowTitle("加载中")
        dlg.setWindowModality(QtCore.Qt.ApplicationModal)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.show()
        QtWidgets.QApplication.processEvents()
        return dlg


    def _finish_busy_dialog(self, dlg: Optional[QtWidgets.QProgressDialog]):
        if dlg is None:
            return
        dlg.close()
        dlg.deleteLater()


    def _set_instance_meta_ui(self, remark: str, label_desc: str):
        self.text_remark.blockSignals(True)
        self.combo_label_desc.blockSignals(True)
        self.text_remark.setPlainText("" if remark is None else str(remark))
        items = [self.combo_label_desc.itemText(i) for i in range(self.combo_label_desc.count())]
        if label_desc in items:
            self.combo_label_desc.setCurrentText(label_desc)
        else:
            self.combo_label_desc.setCurrentText("未选择")
        self.text_remark.blockSignals(False)
        self.combo_label_desc.blockSignals(False)


    def _refresh_instance_meta_ui(self):
        if (not self.annotating) or (self.session.current_inst_id is None):
            self._set_instance_meta_ui("", "")
            return
        remark, label_desc = self.session.get_instance_meta(self.session.current_inst_id)
        self._set_instance_meta_ui(remark, label_desc)


    def _get_current_instance_sem_label(self):
        if self.session.cloud is None:
            return None
        if self.session.current_inst_id is None or self.session.leaf_global_idx is None:
            return None
        if len(self.session.leaf_global_idx) == 0:
            return None
        sem = self.session.get_full_sem()[self.session.leaf_global_idx]
        if len(sem) == 0:
            return None
        vals, counts = np.unique(sem, return_counts=True)
        if len(vals) == 0:
            return None
        return int(vals[int(np.argmax(counts))])


    def _update_instance_sem_label(self):
        if not hasattr(self, "lbl_inst_sem"):
            return
        if (not self.annotating) or (self.session.current_inst_id is None):
            self.lbl_inst_sem.setText("语义标签：-")
            return
        sem_label = self._get_current_instance_sem_label()
        if sem_label is None:
            self.lbl_inst_sem.setText("语义标签：-")
        else:
            self.lbl_inst_sem.setText(f"语义标签：{sem_label}")


    def _set_plant_type(self, plant_type: str, update_status: bool = True, save_settings: bool = True):
        if plant_type not in self.plant_type_actions:
            return
        self.session.plant_type = plant_type
        for t, act in self.plant_type_actions.items():
            act.blockSignals(True)
            act.setChecked(t == plant_type)
            act.blockSignals(False)
        if save_settings:
            self._settings.setValue("plant_type", plant_type)
        if update_status:
            self._update_status(f"已选择植物类型：{plant_type}")


    def on_plant_type_selected(self):
        act = self.sender()
        if act is None:
            return
        plant_type = act.text()
        self._set_plant_type(plant_type)


    def _clear_layout(self, layout: QtWidgets.QLayout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
            elif item.layout() is not None:
                self._clear_layout(item.layout())
            del item


    def _set_semantic_legend(self, labels):
        return


    def on_instance_meta_changed(self):
        if (not self.annotating) or (self.session.current_inst_id is None):
            return
        remark = self.text_remark.toPlainText()
        label_desc = self.combo_label_desc.currentText()
        if label_desc == "未选择":
            label_desc = ""
        self.session.set_instance_meta(self.session.current_inst_id, remark, label_desc)


    def _apply_left_panel_sizes(self, panel_width: int):
        inner_width = max(PANEL_LIST_MIN_WIDTH, panel_width - PANEL_INNER_PADDING)
        row_combo_width = max(PANEL_COMBO_MIN_WIDTH, panel_width - PANEL_ROW_LABEL_SPACE)
        button_width = max(PANEL_BUTTON_MIN_WIDTH, panel_width - PANEL_INNER_PADDING)

        for combo in [
            self.combo_anno_semantic, self.combo_sem_filter, self.combo_stem_filter,
            self.combo_flower_filter, self.combo_fruit_filter, self.combo_inst
        ]:
            combo.setFixedWidth(row_combo_width)

        self.combo_label_desc.setFixedWidth(inner_width)
        self.text_remark.setFixedWidth(inner_width)

        for lst in [
            self.list_base, self.list_tip, self.list_ctrl, self.list_width, self.list_width_ctrl
        ]:
            lst.setFixedWidth(inner_width)

        for btn in [
            self.btn_toggle_base, self.btn_toggle_tip, self.btn_toggle_ctrl,
            self.btn_toggle_width, self.btn_toggle_width_ctrl,
            self.btn_recommend_length, self.btn_generate_length,
            self.btn_recommend_width, self.btn_generate_width,
            self.btn_compute_leaf_area, self.btn_compute_leaf_projected_area,
            self.btn_compute_leaf_inclination, self.btn_compute_leaf_stem_angle,
            self.btn_delete, self.btn_rename_ctrl, self.btn_rename_width_ctrl
        ]:
            btn.setFixedWidth(button_width)


    def _update_buttons(self):
        in_anno = self.annotating
        in_leaf_anno = self.annotating and self.annotate_semantic == "leaf"
        sem_key = self.annotate_semantic
        if not self.annotating:
            sem_key = self._get_annotation_semantic_key()
        in_stem_sem = sem_key == "stem"
        in_flower_fruit_sem = sem_key in ["flower", "fruit"]
        for b in [
            self.btn_toggle_base, self.btn_toggle_tip, self.btn_toggle_ctrl, self.btn_toggle_width,
            self.btn_recommend_length, self.btn_generate_length,
            self.btn_recommend_width, self.btn_generate_width,
            self.btn_compute_leaf_area, self.btn_compute_leaf_projected_area,
            self.btn_compute_leaf_inclination, self.btn_compute_leaf_stem_angle,
            self.btn_delete, self.btn_rename_ctrl,
            self.btn_toggle_width_ctrl, self.btn_rename_width_ctrl,
        ]:
            b.setEnabled(in_leaf_anno)
        self.meta_group.setEnabled(in_anno)
        if hasattr(self, "points_group"):
            self.points_group.setVisible(in_leaf_anno)
        if hasattr(self, "leaf_area_group"):
            self.leaf_area_group.setVisible(in_leaf_anno)
        if hasattr(self, "leaf_angle_group"):
            self.leaf_angle_group.setVisible(in_leaf_anno)
        if hasattr(self, "phenotype_group"):
            self.phenotype_group.setVisible(not in_anno)
        if hasattr(self, "bbox_info"):
            self._update_bbox_info()
        if hasattr(self, "panel_widget"):
            width = PANEL_WIDTH_LEAF_ANNOTATE if in_leaf_anno else PANEL_WIDTH_DEFAULT
            self.panel_widget.setFixedWidth(width)
            self._apply_left_panel_sizes(width)

        has_cloud = self.session.cloud is not None
        can_export = has_cloud
        self.btn_export.setEnabled(can_export)
        self.btn_back_browse.setEnabled(has_cloud)
        self.btn_start_anno.setEnabled(has_cloud and self.combo_inst.count() > 0)

        self.act_export_dir.setEnabled(True)
        self.act_back_browse.setEnabled(has_cloud)
        self.act_start_anno.setEnabled(has_cloud and self.combo_inst.count() > 0)
        self.combo_inst.setEnabled(has_cloud and self.combo_inst.count() > 0)
        self.combo_anno_semantic.setEnabled(has_cloud and not self.annotating)
        self.combo_sem_filter.setEnabled(has_cloud and not self.annotating)
        self.combo_stem_filter.setEnabled(has_cloud and not self.annotating)
        self.combo_flower_filter.setEnabled(has_cloud and not self.annotating)
        self.combo_fruit_filter.setEnabled(has_cloud and not self.annotating)
        for a in [self.act_view_rgb, self.act_view_sem, self.act_view_inst, self.act_view_label]:
            a.setEnabled(has_cloud)

        self.act_save_labels.setEnabled(can_export)
        for a in [
            self.act_recommend_length, self.act_generate_length,
            self.act_recommend_width, self.act_generate_width,
            self.act_smooth_paths, self.act_compute_leaf_area, self.act_compute_leaf_projected_area,
            self.act_compute_leaf_inclination, self.act_compute_leaf_stem_angle,
            self.act_export_labeled,
        ]:
            a.setEnabled(in_leaf_anno)
        self.act_compute_stem.setEnabled(has_cloud and in_stem_sem)
        self.act_compute_stem_length.setEnabled(has_cloud and in_stem_sem)
        self.act_compute_flower_fruit.setEnabled(has_cloud and in_flower_fruit_sem)
        self.btn_toggle_measure.setEnabled(has_cloud)
        stem_tools_visible = in_anno and self.annotate_semantic == "stem"
        if hasattr(self, "btn_toggle_stem_cyl"):
            self.btn_toggle_stem_cyl.setVisible(stem_tools_visible)
            self.btn_toggle_stem_cyl.setEnabled(has_cloud and stem_tools_visible)
        if hasattr(self, "btn_toggle_stem_path"):
            self.btn_toggle_stem_path.setVisible(stem_tools_visible)
            self.btn_toggle_stem_path.setEnabled(has_cloud and stem_tools_visible)
        if not stem_tools_visible:
            changed = False
            if hasattr(self, "btn_toggle_stem_cyl") and self.btn_toggle_stem_cyl.isChecked():
                self.btn_toggle_stem_cyl.blockSignals(True)
                self.btn_toggle_stem_cyl.setChecked(False)
                self.btn_toggle_stem_cyl.blockSignals(False)
                self._clear_stem_cylinders()
                changed = True
            if hasattr(self, "btn_toggle_stem_path") and self.btn_toggle_stem_path.isChecked():
                self.btn_toggle_stem_path.blockSignals(True)
                self.btn_toggle_stem_path.setChecked(False)
                self.btn_toggle_stem_path.blockSignals(False)
                self._clear_stem_length_paths()
                changed = True
            if changed:
                self._refresh_scene()
                self.plotter.render()
        for b in [
            self.btn_view_front, self.btn_view_side, self.btn_view_top,
            self.btn_pick_view_center, self.btn_toggle_aabb, self.btn_toggle_obb
        ]:
            b.setEnabled(has_cloud)

    # ----------------------------
    # vtk picking (Shift+Left Click)
    # ----------------------------

    def _update_phenotype_table(self):
        if not hasattr(self, "table_phenotype"):
            return
        self.table_phenotype.setRowCount(0)
        if self.session.cloud is None:
            return

        sem_map = self._get_instance_sem_map()
        label_to_name = {}
        for key, name in [("leaf", "叶"), ("stem", "茎"), ("flower", "花"), ("fruit", "果")]:
            v = self.session.semantic_map.get(key)
            if v is not None:
                label_to_name[int(v)] = name

        rows = []
        for inst_id, ann in self.session.annotations.items():
            sem_label = sem_map.get(int(inst_id), None)
            sem_text = "-" if sem_label is None else label_to_name.get(int(sem_label), str(int(sem_label)))
            if "length" in ann and ann.get("length") is not None:
                rows.append((inst_id, sem_text, "叶长", f"{float(ann['length']):.3f}"))
            if "width_path_length" in ann and ann.get("width_path_length") is not None:
                rows.append((inst_id, sem_text, "叶宽", f"{float(ann['width_path_length']):.3f}"))
            if "leaf_area" in ann and ann.get("leaf_area") is not None:
                rows.append((inst_id, sem_text, "叶面积", f"{float(ann['leaf_area']):.3f}"))
            if "leaf_projected_area" in ann and ann.get("leaf_projected_area") is not None:
                rows.append((inst_id, sem_text, "投影面积", f"{float(ann['leaf_projected_area']):.3f}"))
            if "leaf_inclination" in ann and ann.get("leaf_inclination") is not None:
                rows.append((inst_id, sem_text, "叶倾角", f"{float(ann['leaf_inclination']):.3f}"))
            if "leaf_stem_angle" in ann and ann.get("leaf_stem_angle") is not None:
                rows.append((inst_id, sem_text, "叶夹角", f"{float(ann['leaf_stem_angle']):.3f}"))
            if "stem_diameter" in ann and ann.get("stem_diameter") is not None:
                rows.append((inst_id, sem_text, "茎粗", f"{float(ann['stem_diameter']):.3f}"))
            if "stem_length" in ann and ann.get("stem_length") is not None:
                rows.append((inst_id, sem_text, "茎长", f"{float(ann['stem_length']):.3f}"))
            if "flower_obb" in ann and isinstance(ann.get("flower_obb"), dict):
                lengths = ann["flower_obb"].get("lengths")
                if lengths is not None and len(lengths) == 3:
                    rows.append((inst_id, sem_text, "花OBB", f"{lengths[0]:.3f},{lengths[1]:.3f},{lengths[2]:.3f}"))
            if "fruit_obb" in ann and isinstance(ann.get("fruit_obb"), dict):
                lengths = ann["fruit_obb"].get("lengths")
                if lengths is not None and len(lengths) == 3:
                    rows.append((inst_id, sem_text, "果OBB", f"{lengths[0]:.3f},{lengths[1]:.3f},{lengths[2]:.3f}"))

        self.table_phenotype.setRowCount(len(rows))
        for r, (inst_id, sem_text, name, value) in enumerate(rows):
            for c, val in enumerate([str(int(inst_id)), str(sem_text), str(name), str(value)]):
                it = QtWidgets.QTableWidgetItem(val)
                it.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
                self.table_phenotype.setItem(r, c, it)
