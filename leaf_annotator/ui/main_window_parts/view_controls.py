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
        self._update_status("选择视图中心：Shift+左键点击点云")


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
