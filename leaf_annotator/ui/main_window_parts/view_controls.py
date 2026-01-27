# ui/main_window_parts/view_controls.py
import numpy as np


class ViewControlsMixin:
    def on_view_mode_changed(self):
        self._refresh_scene()


    def _set_view_mode(self, mode: str):
        self._refresh_scene(mode=mode)


    def on_view_front(self):
        if self.session.cloud is None:
            return
        self.plotter.view_xy()
        self.plotter.render()


    def on_view_side(self):
        if self.session.cloud is None:
            return
        self.plotter.view_yz()
        self.plotter.render()


    def on_view_top(self):
        if self.session.cloud is None:
            return
        self.plotter.view_xz()
        self.plotter.render()


    def on_pick_view_center(self):
        if self.session.cloud is None:
            return
        self._pick_view_center = True
        self._view_center_point = None
        self._remove_actor(self.A_VIEW_CENTER)
        self._update_status(self.tr("选择视图中心：Shift+左键点击点云"))


    def _set_view_center(self, point_xyz):
        if point_xyz is None:
            return
        self._view_center_point = np.array(point_xyz, dtype=np.float64)
        self._add_points_actor(self.A_VIEW_CENTER, np.asarray([self._view_center_point]), "black", 18)
        self.plotter.set_focus(point_xyz)
        self.plotter.render()


    def on_toggle_aabb(self, checked: bool):
        if not checked:
            self._remove_actor(self.A_AABB)
            self._update_bbox_info()
            self.plotter.render()
            return
        self._update_aabb_actor()
        self._update_bbox_info()
        self.plotter.render()


    def on_toggle_obb(self, checked: bool):
        if not checked:
            self._remove_actor(self.A_OBB)
            self._update_bbox_info()
            self.plotter.render()
            return
        self._update_obb_actor()
        self._update_bbox_info()
        self.plotter.render()


    def on_toggle_stem_cyl(self, checked: bool):
        if not checked:
            self._clear_stem_cylinders()
            if hasattr(self, "btn_toggle_stem_path") and self.btn_toggle_stem_path.isChecked():
                self._show_stem_only_scene()
            else:
                self._refresh_scene()
            self.plotter.render()
            return
        self._show_stem_only_scene()
        self.plotter.render()


    def on_toggle_stem_path(self, checked: bool):
        if not checked:
            self._clear_stem_length_paths()
            if hasattr(self, "btn_toggle_stem_cyl") and self.btn_toggle_stem_cyl.isChecked():
                self._show_stem_only_scene()
            else:
                self._refresh_scene()
            self.plotter.render()
            return
        self._show_stem_only_scene()
        self.plotter.render()


    def on_toggle_growth_direction(self, checked: bool):
        if self.session.cloud is None:
            return
        if checked and self.session.growth_direction is None:
            self.btn_toggle_growth_dir.blockSignals(True)
            self.btn_toggle_growth_dir.setChecked(False)
            self.btn_toggle_growth_dir.blockSignals(False)
            self._update_status(self.tr("请先确定生长方向。"))
            return
        self._update_growth_direction_display()
        self.plotter.render()


    def on_toggle_plant_height(self, checked: bool):
        if self.session.cloud is None:
            return
        has_measure = hasattr(self.session, "has_plant_measurements") and self.session.has_plant_measurements()
        if checked and not has_measure:
            self.btn_toggle_plant_height.blockSignals(True)
            self.btn_toggle_plant_height.setChecked(False)
            self.btn_toggle_plant_height.blockSignals(False)
            self._update_status(self.tr("请先测量株高/冠幅。"))
            return
        self._update_plant_measurement_display()
        self.plotter.render()


    def on_toggle_plant_crown(self, checked: bool):
        if self.session.cloud is None:
            return
        has_measure = hasattr(self.session, "has_plant_measurements") and self.session.has_plant_measurements()
        if checked and not has_measure:
            self.btn_toggle_plant_crown.blockSignals(True)
            self.btn_toggle_plant_crown.setChecked(False)
            self.btn_toggle_plant_crown.blockSignals(False)
            self._update_status(self.tr("请先测量株高/冠幅。"))
            return
        self._update_plant_measurement_display()
        self.plotter.render()


    def on_start_growth_rotation(self):
        if self.session.cloud is None:
            return
        if self.session.growth_direction is None:
            self._update_status(self.tr("请先确定生长方向。"))
            return
        self._rotation_active = True
        self._update_buttons()
        self._update_status(self.tr("旋转已开启：选择旋转轴并使用旋转按钮调整。"))


    def on_finish_growth_rotation(self):
        if not self._rotation_active:
            return
        self._rotation_active = False
        self._update_buttons()
        self._update_status(self.tr("旋转已完成，可继续测量株高/冠幅。"))


    def on_rotate_growth_step(self, direction: int):
        if self.session.cloud is None:
            return
        if not self._rotation_active:
            self._update_status(self.tr("请先点击开始旋转。"))
            return
        axis = "Z"
        if hasattr(self, "combo_rotate_axis"):
            axis = self.combo_rotate_axis.currentText()
        step = 5.0
        if hasattr(self, "spin_rotate_step"):
            step = float(self.spin_rotate_step.value())
        step = float(step) * (1 if direction >= 0 else -1)
        rx = step if axis == "X" else 0.0
        ry = step if axis == "Y" else 0.0
        rz = step if axis == "Z" else 0.0
        try:
            self.session.rotate_growth_direction(rx, ry, rz)
        except Exception as e:
            self._update_status(self.tr("旋转失败：{err}", err=e))
            return
        self._update_growth_direction_display()
        self._update_plant_measurement_display()
        self._update_phenotype_table()
        self.plotter.render()


    def on_reset_growth_rotation(self):
        if self.session.growth_direction is None:
            return
        origin = self.session.growth_origin
        if origin is None:
            pts = self.session.get_full_xyz()
            origin = pts[int(np.argmin(pts[:, 2]))]
        self.session.set_growth_direction(origin=origin, direction=[0.0, 0.0, 1.0], method="manual")
        self._update_growth_direction_display()
        self._update_plant_measurement_display()
        self._update_phenotype_table()
        self._update_buttons()
        self.plotter.render()
