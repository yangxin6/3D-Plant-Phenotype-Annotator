# ui/main_window_parts/interaction.py
import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QMessageBox

from core.annotation_parts.utils import _polyline_length


class InteractionMixin:
    def _install_vtk_pick_observer(self):
        if self._pick_observer_id is not None:
            return

        iren = self.plotter.interactor

        def _on_left_press(obj, event):
            if self.pick_mode == self.MODE_NONE and not self._pick_view_center:
                return
            if self.pick_mode in [self.MODE_MEASURE, self.MODE_GROWTH_BASE] or self._pick_view_center:
                if self.session.cloud is None:
                    return
            else:
                if (not self.annotating) or (self.session.ds is None):
                    return
            if iren.GetShiftKey() == 0:
                return

            x, y = iren.GetEventPosition()
            renderer = self.plotter.renderer

            self._vtk_picker.PickFromListOn()
            self._vtk_picker.InitializePickList()
            if self.pick_mode in [self.MODE_MEASURE, self.MODE_GROWTH_BASE] or self._pick_view_center:
                if self.annotating and self._actor_cloud_inst is not None:
                    try:
                        self._vtk_picker.AddPickList(self._actor_cloud_inst)
                    except Exception:
                        pass
                elif self._actor_cloud_full is not None:
                    try:
                        self._vtk_picker.AddPickList(self._actor_cloud_full)
                    except Exception:
                        pass
            elif self._actor_cloud_inst is not None:
                try:
                    self._vtk_picker.AddPickList(self._actor_cloud_inst)
                except Exception:
                    pass

            ok = self._vtk_picker.Pick(x, y, 0, renderer)
            if not ok:
                return
            pid = self._vtk_picker.GetPointId()
            if pid < 0:
                return

            pick_pos = np.array(self._vtk_picker.GetPickPosition(), dtype=np.float64)
            if self._pick_view_center:
                self._pick_view_center = False
                self._set_view_center(pick_pos)
            else:
                self.on_picked_point(pick_pos)

            # ✅ 修复：不同 VTK 对象/版本不一定有 AbortFlagOn()
            try:
                obj.AbortFlagOn()
            except AttributeError:
                try:
                    obj.SetAbortFlag(1)
                except Exception:
                    pass

        self._pick_observer_id = iren.AddObserver("LeftButtonPressEvent", _on_left_press)

    # ----------------------------
    # actor helpers (incremental update)
    # ----------------------------

    def _enter_mode(self, mode: str):
        if self.pick_mode != self.MODE_NONE:
            self._exit_current_mode(commit=True)

        self.pick_mode = mode

        if mode == self.MODE_BASE:
            self.temp_base_idx = None
            self.btn_toggle_base.setText(self.tr("叶基选择：开启"))
        elif mode == self.MODE_TIP:
            self.temp_tip_idx = None
            self.btn_toggle_tip.setText(self.tr("叶尖选择：开启"))
        elif mode == self.MODE_CTRL:
            self.temp_ctrl_indices = []
            self.btn_toggle_ctrl.setText(self.tr("控制点选择：开启"))
        elif mode == self.MODE_WIDTH:
            self.temp_w1_idx = None
            self.temp_w2_idx = None
            self.btn_toggle_width.setText(self.tr("叶宽点选择：开启 (W1/W2)"))
        elif mode == self.MODE_WIDTH_CTRL:
            self.temp_width_ctrl_indices = []
            self.btn_toggle_width_ctrl.setText(self.tr("叶宽控制点选择：开启"))
        elif mode == self.MODE_MEASURE:
            self.temp_measure_p1 = None
            self.temp_measure_p2 = None
            self.btn_toggle_measure.setText(self.tr("测距：开启"))

        self._update_markers_temp()
        self._update_labels_temp()
        self.plotter.render()

        if mode == self.MODE_WIDTH:
            self._update_status(self.tr("叶宽点选择：Shift+左键依次选择 W1/W2；关闭后保存。"))
        elif mode == self.MODE_WIDTH_CTRL:
            self._update_status(self.tr("叶宽控制点：Shift+左键添加；关闭后保存。"))
        elif mode == self.MODE_MEASURE:
            self._update_status(self.tr("测距：Shift+左键选择两个点，显示连线与距离。"))
        else:
            self._update_status(self.tr("提示：Shift+左键选点；再次点击按钮关闭并保存。"))


    def _exit_current_mode(self, commit: bool = True):
        if commit:
            if self.pick_mode == self.MODE_BASE and self.temp_base_idx is not None:
                self.session.set_base(self.temp_base_idx)
            if self.pick_mode == self.MODE_TIP and self.temp_tip_idx is not None:
                self.session.set_tip(self.temp_tip_idx)
            if self.pick_mode == self.MODE_CTRL and len(self.temp_ctrl_indices) > 0:
                self.session.extend_ctrl(self.temp_ctrl_indices)
            if self.pick_mode == self.MODE_WIDTH:
                if self.temp_w1_idx is not None:
                    self.session.width_w1_idx = int(self.temp_w1_idx)
                if self.temp_w2_idx is not None:
                    self.session.width_w2_idx = int(self.temp_w2_idx)
            if self.pick_mode == self.MODE_WIDTH_CTRL and len(self.temp_width_ctrl_indices) > 0:
                self.session.extend_width_ctrl(self.temp_width_ctrl_indices)

        if self.pick_mode == self.MODE_BASE:
            self.temp_base_idx = None
        elif self.pick_mode == self.MODE_TIP:
            self.temp_tip_idx = None
        elif self.pick_mode == self.MODE_CTRL:
            self.temp_ctrl_indices = []
        elif self.pick_mode == self.MODE_WIDTH:
            self.temp_w1_idx = None
            self.temp_w2_idx = None
        elif self.pick_mode == self.MODE_WIDTH_CTRL:
            self.temp_width_ctrl_indices = []
        elif self.pick_mode == self.MODE_MEASURE:
            self.temp_measure_p1 = None
            self.temp_measure_p2 = None

        self.pick_mode = self.MODE_NONE

        for b in [
            self.btn_toggle_base, self.btn_toggle_tip, self.btn_toggle_ctrl,
            self.btn_toggle_width, self.btn_toggle_width_ctrl, self.btn_toggle_measure
        ]:
            b.blockSignals(True)
            b.setChecked(False)
            b.blockSignals(False)

        self.btn_toggle_base.setText(self.tr("叶基选择：关闭"))
        self.btn_toggle_tip.setText(self.tr("叶尖选择：关闭"))
        self.btn_toggle_ctrl.setText(self.tr("控制点选择：关闭"))
        self.btn_toggle_width.setText(self.tr("叶宽端点选择：关闭 (W1/W2)"))
        self.btn_toggle_width_ctrl.setText(self.tr("叶宽控制点选择：关闭"))
        self.btn_toggle_measure.setText(self.tr("测距：关闭"))

        self._update_markers_saved()
        self._update_markers_temp()
        self._update_labels_saved()
        self._update_labels_temp()
        self._update_lines()
        self._update_measure_display()
        self.plotter.render()

        self._refresh_point_lists()

    # ----------------------------
    # toggle slots
    # ----------------------------

    def on_toggle_base(self, checked: bool):
        if not self.annotating:
            return
        if checked:
            self._close_other_toggles(except_mode=self.MODE_BASE)
            self._enter_mode(self.MODE_BASE)
        else:
            if self.pick_mode == self.MODE_BASE:
                self._exit_current_mode(commit=True)
                self._invalidate_results_after_point_change()
                self._maybe_recommend_width_and_refresh()
                self._update_status(self.tr("叶基选择关闭：叶基点已保存并保留显示。"))


    def on_toggle_tip(self, checked: bool):
        if not self.annotating:
            return
        if checked:
            self._close_other_toggles(except_mode=self.MODE_TIP)
            self._enter_mode(self.MODE_TIP)
        else:
            if self.pick_mode == self.MODE_TIP:
                self._exit_current_mode(commit=True)
                self._invalidate_results_after_point_change()
                self._maybe_recommend_width_and_refresh()
                self._update_status(self.tr("叶尖选择关闭：叶尖点已保存并保留显示。"))


    def on_toggle_ctrl(self, checked: bool):
        if not self.annotating:
            return
        if checked:
            self._close_other_toggles(except_mode=self.MODE_CTRL)
            self._enter_mode(self.MODE_CTRL)
        else:
            if self.pick_mode == self.MODE_CTRL:
                self._exit_current_mode(commit=True)
                self._invalidate_results_after_point_change()
                self._maybe_recommend_width_and_refresh()
                self._update_status(self.tr("控制点选择关闭：控制点已保存并保留显示。"))


    def on_toggle_width(self, checked: bool):
        if not self.annotating:
            return
        if checked:
            self._close_other_toggles(except_mode=self.MODE_WIDTH)
            self._enter_mode(self.MODE_WIDTH)
        else:
            if self.pick_mode == self.MODE_WIDTH:
                self._exit_current_mode(commit=True)
                self._invalidate_results_after_point_change()
                self._update_status(self.tr("叶宽点选择关闭：W1/W2 已保存并保留显示。"))


    def on_toggle_width_ctrl(self, checked: bool):
        if not self.annotating:
            return
        if checked:
            self._close_other_toggles(except_mode=self.MODE_WIDTH_CTRL)
            self._enter_mode(self.MODE_WIDTH_CTRL)
        else:
            if self.pick_mode == self.MODE_WIDTH_CTRL:
                self._exit_current_mode(commit=True)
                self._invalidate_results_after_point_change()
                if getattr(self.session, "width_path_points", None) is not None:
                    path_pts, path_idx = self.session.build_width_polyline()
                    if path_pts is not None and path_idx is not None:
                        self.session.width_path_points = path_pts
                        self.session.width_path_indices = np.asarray(path_idx, dtype=np.int64)
                        self.session.width_path_length = _polyline_length(path_pts)
                        self._update_lines()
                        self.plotter.render()
                self._update_status(self.tr("叶宽控制点选择关闭：已保存。"))


    def on_toggle_measure(self, checked: bool):
        if self.session.cloud is None:
            return
        self.btn_toggle_measure.blockSignals(True)
        self.btn_toggle_measure.setChecked(checked)
        self.btn_toggle_measure.blockSignals(False)
        if checked:
            self._close_other_toggles(except_mode=self.MODE_MEASURE)
            self._enter_mode(self.MODE_MEASURE)
        else:
            if self.pick_mode == self.MODE_MEASURE:
                self._exit_current_mode(commit=False)
                self._update_status(self.tr("测距已关闭。"))


    def _close_other_toggles(self, except_mode: str):
        mapping = {
            self.MODE_BASE: self.btn_toggle_base,
            self.MODE_TIP: self.btn_toggle_tip,
            self.MODE_CTRL: self.btn_toggle_ctrl,
            self.MODE_WIDTH: self.btn_toggle_width,
            self.MODE_WIDTH_CTRL: self.btn_toggle_width_ctrl,
            self.MODE_MEASURE: self.btn_toggle_measure,
        }
        for mode, btn in mapping.items():
            if mode == except_mode:
                continue
            if btn.isChecked():
                self._exit_current_mode(commit=True)
                break

    # ----------------------------
    # pick callback
    # ----------------------------

    def on_picked_point(self, point_xyz):
        if point_xyz is None or self.pick_mode == self.MODE_NONE:
            return
        if self.pick_mode not in [self.MODE_MEASURE, self.MODE_GROWTH_BASE] and self.session.ds is None:
            return

        p = np.array(point_xyz, dtype=np.float64)

        if self.pick_mode == self.MODE_BASE:
            idx = self.session.snap_to_ds_index(p)
            self.temp_base_idx = idx
        elif self.pick_mode == self.MODE_TIP:
            idx = self.session.snap_to_ds_index(p)
            self.temp_tip_idx = idx
        elif self.pick_mode == self.MODE_CTRL:
            idx = self.session.snap_to_ds_index(p)
            self.temp_ctrl_indices.append(idx)
        elif self.pick_mode == self.MODE_WIDTH:
            idx = self.session.snap_to_ds_index(p)
            if self.temp_w1_idx is None:
                self.temp_w1_idx = idx
            elif self.temp_w2_idx is None:
                self.temp_w2_idx = idx
            else:
                self.temp_w2_idx = idx
        elif self.pick_mode == self.MODE_WIDTH_CTRL:
            idx = self.session.snap_to_ds_index(p)
            self.temp_width_ctrl_indices.append(idx)
        elif self.pick_mode == self.MODE_MEASURE:
            if self.temp_measure_p1 is None:
                self.temp_measure_p1 = p.copy()
            elif self.temp_measure_p2 is None:
                self.temp_measure_p2 = p.copy()
            else:
                self.temp_measure_p1 = p.copy()
                self.temp_measure_p2 = None
            self._update_measure_display()
        elif self.pick_mode == self.MODE_GROWTH_BASE:
            self.pick_mode = self.MODE_NONE
            self.session.set_growth_direction(origin=p, direction=[0.0, 0.0, 1.0], method="manual")
            self._update_buttons()
            if hasattr(self, "btn_toggle_growth_dir"):
                self.btn_toggle_growth_dir.blockSignals(True)
                self.btn_toggle_growth_dir.setChecked(True)
                self.btn_toggle_growth_dir.blockSignals(False)
            self._update_growth_direction_display()
            self._update_status(self.tr("已设置生长方向基准点：可旋转对准生长方向。"))
            self.plotter.render()
            return

        self._update_markers_temp()
        self._update_labels_temp()
        self.plotter.render()

    # ----------------------------
    # delete actions
    # ----------------------------

    def on_ctrl_context_menu(self, pos):
        it = self.list_ctrl.itemAt(pos)
        if it is None:
            return
        if not it.isSelected():
            self.list_ctrl.selectionModel().select(
                self.list_ctrl.indexFromItem(it),
                QtCore.QItemSelectionModel.ClearAndSelect
            )
        menu = QtWidgets.QMenu(self)
        act_delete = menu.addAction(self.tr("删除"))
        act_rename = menu.addAction(self.tr("修改顺序"))
        action = menu.exec_(self.list_ctrl.mapToGlobal(pos))
        if action == act_delete:
            self._delete_selected_ctrl_items()
        elif action == act_rename:
            if len(self.list_ctrl.selectedItems()) > 1:
                QMessageBox.information(self, self.tr("提示"), self.tr("修改顺序仅支持单个控制点。"))
                return
            self.on_rename_ctrl()


    def on_width_ctrl_context_menu(self, pos):
        it = self.list_width_ctrl.itemAt(pos)
        if it is None:
            return
        if not it.isSelected():
            self.list_width_ctrl.selectionModel().select(
                self.list_width_ctrl.indexFromItem(it),
                QtCore.QItemSelectionModel.ClearAndSelect
            )
        menu = QtWidgets.QMenu(self)
        act_delete = menu.addAction(self.tr("删除"))
        act_rename = menu.addAction(self.tr("修改顺序"))
        action = menu.exec_(self.list_width_ctrl.mapToGlobal(pos))
        if action == act_delete:
            self._delete_selected_width_ctrl_items()
        elif action == act_rename:
            if len(self.list_width_ctrl.selectedItems()) > 1:
                QMessageBox.information(self, self.tr("提示"), self.tr("修改顺序仅支持单个控制点。"))
                return
            self.on_rename_width_ctrl()


    def on_base_context_menu(self, pos):
        it = self.list_base.itemAt(pos)
        if it is None:
            return
        self.list_base.setCurrentItem(it)
        menu = QtWidgets.QMenu(self)
        act_delete = menu.addAction(self.tr("删除"))
        action = menu.exec_(self.list_base.mapToGlobal(pos))
        if action == act_delete:
            self.on_delete_base()


    def on_tip_context_menu(self, pos):
        it = self.list_tip.itemAt(pos)
        if it is None:
            return
        self.list_tip.setCurrentItem(it)
        menu = QtWidgets.QMenu(self)
        act_delete = menu.addAction(self.tr("删除"))
        action = menu.exec_(self.list_tip.mapToGlobal(pos))
        if action == act_delete:
            self.on_delete_tip()


    def on_width_context_menu(self, pos):
        it = self.list_width.itemAt(pos)
        if it is None:
            return
        self.list_width.setCurrentItem(it)
        menu = QtWidgets.QMenu(self)
        act_delete = menu.addAction(self.tr("删除"))
        action = menu.exec_(self.list_width.mapToGlobal(pos))
        if action == act_delete:
            self.on_delete_width()


    def _delete_selected_ctrl_items(self):
        ctrl_items = self.list_ctrl.selectedItems()
        if not ctrl_items:
            return
        indices = []
        for it in ctrl_items:
            info = it.data(0x0100)
            if info and info[0] == "ctrl":
                indices.append(int(info[1]))
        if not indices:
            return
        for idx in sorted(set(indices), reverse=True):
            if 0 <= idx < len(self.session.ctrl_indices):
                self.session.ctrl_indices.pop(idx)
                if idx < len(self.session.ctrl_ids):
                    self.session.ctrl_ids.pop(idx)
        self._invalidate_results_after_point_change()
        self._update_markers_saved()
        self._update_labels_saved()
        self.plotter.render()
        self._refresh_point_lists()
        self._maybe_recommend_width_and_refresh()
        self._update_status(self.tr("已删除选中的叶长控制点。"))


    def _delete_selected_width_ctrl_items(self):
        wctrl_items = self.list_width_ctrl.selectedItems()
        if not wctrl_items:
            return
        indices = []
        for it in wctrl_items:
            info = it.data(0x0100)
            if info and info[0] == "wctrl":
                indices.append(int(info[1]))
        if not indices:
            return
        for idx in sorted(set(indices), reverse=True):
            if 0 <= idx < len(self.session.width_ctrl_indices):
                self.session.width_ctrl_indices.pop(idx)
                if idx < len(self.session.width_ctrl_ids):
                    self.session.width_ctrl_ids.pop(idx)
        self._invalidate_results_after_point_change()
        self._update_markers_saved()
        self._update_labels_saved()
        self.plotter.render()
        self._refresh_point_lists()
        self._update_status(self.tr("已删除选中的叶宽控制点。"))


    def on_delete_selected(self):
        if self.list_base.currentItem() is not None:
            self.on_delete_base()
            return
        if self.list_tip.currentItem() is not None:
            self.on_delete_tip()
            return
        ctrl_items = self.list_ctrl.selectedItems()
        if ctrl_items:
            self._delete_selected_ctrl_items()
            return
        if self.list_width.currentItem() is not None:
            self.on_delete_width()
            return
        wctrl_items = self.list_width_ctrl.selectedItems()
        if wctrl_items:
            self._delete_selected_width_ctrl_items()
            return
        QMessageBox.information(self, self.tr("提示"), self.tr("请先在左侧列表中选择要删除的标记。"))


    def on_delete_base(self):
        if self.session.base_idx is None:
            return
        self.session.base_idx = None
        self._invalidate_results_after_point_change()
        self._update_markers_saved()
        self._update_labels_saved()
        self.plotter.render()
        self._refresh_point_lists()
        self._update_status(self.tr("已删除叶基点。"))


    def on_delete_tip(self):
        if self.session.tip_idx is None:
            return
        self.session.tip_idx = None
        self._invalidate_results_after_point_change()
        self._update_markers_saved()
        self._update_labels_saved()
        self.plotter.render()
        self._refresh_point_lists()
        self._update_status(self.tr("已删除叶尖点。"))


    def on_delete_ctrl(self):
        it = self.list_ctrl.currentItem()
        if it is None:
            return
        info = it.data(0x0100)
        if not info or info[0] != "ctrl":
            return
        idx_in_list = int(info[1])
        if 0 <= idx_in_list < len(self.session.ctrl_indices):
            self.session.ctrl_indices.pop(idx_in_list)
            if idx_in_list < len(self.session.ctrl_ids):
                self.session.ctrl_ids.pop(idx_in_list)

        self._invalidate_results_after_point_change()
        self._update_markers_saved()
        self._update_labels_saved()
        self.plotter.render()
        self._refresh_point_lists()
        self._maybe_recommend_width_and_refresh()
        self._update_status(self.tr("已删除选中的控制点。"))


    def on_rename_ctrl(self):
        it = self.list_ctrl.currentItem()
        if it is None:
            return
        info = it.data(0x0100)
        if not info or info[0] != "ctrl":
            return
        idx_in_list = int(info[1])
        if idx_in_list >= len(self.session.ctrl_ids):
            return

        cur_id = self.session.ctrl_ids[idx_in_list]
        new_id, ok = QtWidgets.QInputDialog.getInt(
            self, self.tr("重命名叶长控制点"),
            self.tr("输入新的编号（正整数）"),
            value=int(cur_id), min=1, max=100000
        )
        if not ok:
            return
        new_id = int(new_id)
        if new_id in self.session.ctrl_ids and new_id != cur_id:
            QMessageBox.information(self, self.tr("提示"), self.tr("编号已存在，请选择其他编号。"))
            return

        self.session.ctrl_ids[idx_in_list] = new_id

        self._invalidate_results_after_point_change()
        self._update_markers_saved()
        self._update_labels_saved()
        self.plotter.render()
        self._refresh_point_lists()
        self._maybe_recommend_width_and_refresh()
        self._update_status(self.tr("已修改叶长控制点编号。"))


    def on_delete_width_ctrl(self):
        it = self.list_width_ctrl.currentItem()
        if it is None:
            return
        info = it.data(0x0100)
        if not info or info[0] != "wctrl":
            return
        idx_in_list = int(info[1])
        if 0 <= idx_in_list < len(self.session.width_ctrl_indices):
            self.session.width_ctrl_indices.pop(idx_in_list)
            if idx_in_list < len(self.session.width_ctrl_ids):
                self.session.width_ctrl_ids.pop(idx_in_list)

        self._invalidate_results_after_point_change()
        self._update_markers_saved()
        self._update_labels_saved()
        self.plotter.render()
        self._refresh_point_lists()
        self._update_status(self.tr("已删除选中的叶宽控制点。"))


    def on_rename_width_ctrl(self):
        it = self.list_width_ctrl.currentItem()
        if it is None:
            return
        info = it.data(0x0100)
        if not info or info[0] != "wctrl":
            return
        idx_in_list = int(info[1])
        if idx_in_list >= len(self.session.width_ctrl_ids):
            return

        cur_id = self.session.width_ctrl_ids[idx_in_list]
        new_id, ok = QtWidgets.QInputDialog.getInt(
            self, self.tr("重命名叶宽控制点"),
            self.tr("输入新的编号（正整数）"),
            value=int(cur_id), min=1, max=100000
        )
        if not ok:
            return
        new_id = int(new_id)
        if new_id in self.session.width_ctrl_ids and new_id != cur_id:
            QMessageBox.information(self, self.tr("提示"), self.tr("编号已存在，请选择其他编号。"))
            return

        self.session.width_ctrl_ids[idx_in_list] = new_id

        self._invalidate_results_after_point_change()
        self._update_markers_saved()
        self._update_labels_saved()
        self.plotter.render()
        self._refresh_point_lists()
        self._update_status(self.tr("已修改叶宽控制点编号。"))


    def on_delete_width(self):
        it = self.list_width.currentItem()
        if it is None:
            return
        info = it.data(0x0100)
        if not info or info[0] != "w":
            return
        which = int(info[1])  # 1 or 2

        w2 = getattr(self.session, "width_w2_idx", None)

        if which == 1:
            if w2 is not None:
                self.session.width_w1_idx = int(w2)
                self.session.width_w2_idx = None
            else:
                self.session.width_w1_idx = None
        else:
            self.session.width_w2_idx = None

        self._invalidate_results_after_point_change()
        self._update_markers_saved()
        self._update_labels_saved()
        self.plotter.render()
        self._refresh_point_lists()
        self._update_status(self.tr("已删除选中的叶宽点。"))

    # ----------------------------
    # view mode changed (browse only)
    # ----------------------------
