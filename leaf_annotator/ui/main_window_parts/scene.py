# ui/main_window_parts/scene.py
from typing import Optional

import numpy as np
import pyvista as pv
import vtk
from PyQt5 import QtWidgets, QtCore

from .constants import LABEL_SWATCH_SIZE
from .utils import make_polyline_mesh, stable_color_from_id, colors_from_labels


class SceneMixin:
    def _remove_actor(self, name: str):
        try:
            self.plotter.remove_actor(name)
        except Exception:
            pass


    def _add_points_actor(self, name: str, pts: np.ndarray, color, point_size: int):
        self._remove_actor(name)
        if pts is None or len(pts) == 0:
            return None
        cam = self.plotter.camera_position
        poly = pv.PolyData(pts)
        actor = self.plotter.add_mesh(
            poly, name=name, color=color,
            render_points_as_spheres=True, point_size=point_size
        )
        if cam is not None:
            self.plotter.camera_position = cam
        return actor


    def _add_polyline_actor(self, name: str, pts: np.ndarray, color, line_width: int):
        self._remove_actor(name)
        if pts is None or len(pts) < 2:
            return None
        cam = self.plotter.camera_position
        actor = self.plotter.add_mesh(
            make_polyline_mesh(np.asarray(pts, dtype=np.float64)),
            name=name, color=color, line_width=line_width
        )
        if cam is not None:
            self.plotter.camera_position = cam
        return actor


    def _add_labels_actor(self, name: str, pts: np.ndarray, labels: list):
        self._remove_actor(name)
        if pts is None or len(pts) == 0:
            return None
        cam = self.plotter.camera_position
        actor = self.plotter.add_point_labels(
            pts, labels,
            name=name,
            point_size=0,
            font_size=14,
            shape_opacity=0.25,
            always_visible=True
        )
        if cam is not None:
            self.plotter.camera_position = cam
        return actor


    def _get_full_phenotype_labels(self) -> Optional[np.ndarray]:
        if self.session.cloud is None:
            return None
        n = len(self.session.get_full_xyz())
        full_labels = getattr(self.session, "full_point_labels", None)
        if full_labels is not None and len(full_labels) == n:
            return np.asarray(full_labels, dtype=np.int64)
        if not self.session.annotations:
            return None
        labels = np.full((n,), -1, dtype=np.int64)
        any_labels = False
        full_inst = self.session.get_full_inst()
        for inst_id, ann in self.session.annotations.items():
            pt_labels = ann.get("point_labels")
            if pt_labels is None:
                continue
            pt_labels = np.asarray(pt_labels, dtype=np.int64)
            if len(pt_labels) == 0:
                continue
            idx = ann.get("point_label_global_indices")
            if idx is not None:
                idx = np.asarray(idx, dtype=np.int64)
                if len(idx) != len(pt_labels):
                    idx = None
            if idx is None:
                idx = np.where(full_inst == int(inst_id))[0].astype(np.int64)
                if len(idx) != len(pt_labels):
                    continue
            labels[idx] = pt_labels
            any_labels = True
        return labels if any_labels else None

    # ----------------------------
    # browsing polydata
    # ----------------------------

    def _get_full_cloud_polydata(self) -> pv.PolyData:
        xyz = self.session.get_full_xyz()
        poly = pv.PolyData(xyz)

        mode = self._get_view_mode()
        if mode == self.VIEW_RGB and self.session.has_rgb():
            poly["rgb"] = self.session.get_full_rgb()
        elif mode == self.VIEW_SEM:
            poly["rgb"] = colors_from_labels(self.session.get_full_sem())
        elif mode == self.VIEW_INST:
            poly["rgb"] = colors_from_labels(self.session.get_full_inst())
        elif mode == self.VIEW_LABEL:
            full_labels = self._get_full_phenotype_labels()
            if full_labels is not None:
                poly["rgb"] = colors_from_labels(full_labels)
            else:
                poly["rgb"] = np.full((len(xyz), 3), 180, dtype=np.uint8)
        else:
            poly["rgb"] = np.full((len(xyz), 3), 180, dtype=np.uint8)
        return poly

    # ----------------------------
    # scene update
    # ----------------------------

    def _show_browse_scene(self):
        poly = self._get_full_cloud_polydata()

        self._remove_actor(self.A_CLOUD_INST)
        self._actor_cloud_inst = None

        self._remove_actor(self.A_CLOUD_FULL)
        self._actor_cloud_full = self.plotter.add_points(
            poly, scalars="rgb", rgb=True,
            name=self.A_CLOUD_FULL,
            render_points_as_spheres=True, point_size=3
        )

        for nm in [
            self.A_BASE_SAVED, self.A_TIP_SAVED, self.A_CTRL_SAVED, self.A_WIDTH_SAVED, self.A_WIDTH_CTRL_SAVED,
            self.A_BASE_TEMP, self.A_TIP_TEMP, self.A_CTRL_TEMP, self.A_WIDTH_TEMP, self.A_WIDTH_CTRL_TEMP,
            self.A_LABELS_SAVED, self.A_LABELS_TEMP,
            self.A_LINE_CENTER_CUR, self.A_LINE_WIDTH_CUR,
            self.A_LINE_CENTER_CACHED, self.A_LINE_WIDTH_CACHED,
            self.A_MEASURE_LINE, self.A_MEASURE_P1, self.A_MEASURE_P2,
            self.A_VIEW_CENTER,
            self.A_LEAF_INCLINE_N, self.A_LEAF_INCLINE_Z, self.A_LEAF_INCLINE_ARC, self.A_LEAF_INCLINE_LABEL,
            self.A_LEAF_STEM_LEAF, self.A_LEAF_STEM_STEM, self.A_LEAF_STEM_ARC, self.A_LEAF_STEM_LABEL,
            self.A_GROWTH_DIR, self.A_GROWTH_DIR_LABEL,
            self.A_PLANT_HEIGHT, self.A_PLANT_HEIGHT_LABEL,
            self.A_PLANT_CROWN, self.A_PLANT_CROWN_LABEL,
        ]:
            self._remove_actor(nm)

        self._update_measure_display()
        self.plotter.render()


    def _show_stem_only_scene(self):
        stem_pts = self._get_stem_points()
        if stem_pts is None or len(stem_pts) == 0:
            QMessageBox.information(self, self.tr("提示"), self.tr("没有可显示的茎点云，请检查茎语义选择。"))
            for btn in ["btn_toggle_stem_cyl", "btn_toggle_stem_path"]:
                if hasattr(self, btn):
                    w = getattr(self, btn)
                    w.blockSignals(True)
                    w.setChecked(False)
                    w.blockSignals(False)
            self._clear_stem_cylinders()
            self._clear_stem_length_paths()
            self._refresh_scene()
            return

        self._remove_actor(self.A_CLOUD_INST)
        self._actor_cloud_inst = None
        self._remove_actor(self.A_CLOUD_FULL)
        self._actor_cloud_full = self.plotter.add_points(
            pv.PolyData(stem_pts), color=(0.6, 0.4, 0.2),
            name=self.A_CLOUD_FULL, render_points_as_spheres=True, point_size=3
        )

        for nm in [
            self.A_BASE_SAVED, self.A_TIP_SAVED, self.A_CTRL_SAVED, self.A_WIDTH_SAVED, self.A_WIDTH_CTRL_SAVED,
            self.A_BASE_TEMP, self.A_TIP_TEMP, self.A_CTRL_TEMP, self.A_WIDTH_TEMP, self.A_WIDTH_CTRL_TEMP,
            self.A_LABELS_SAVED, self.A_LABELS_TEMP,
            self.A_LINE_CENTER_CUR, self.A_LINE_WIDTH_CUR,
            self.A_LINE_CENTER_CACHED, self.A_LINE_WIDTH_CACHED,
            self.A_MEASURE_LINE, self.A_MEASURE_P1, self.A_MEASURE_P2,
            self.A_VIEW_CENTER, self.A_AABB, self.A_OBB,
            self.A_LEAF_INCLINE_N, self.A_LEAF_INCLINE_Z, self.A_LEAF_INCLINE_ARC, self.A_LEAF_INCLINE_LABEL,
            self.A_LEAF_STEM_LEAF, self.A_LEAF_STEM_STEM, self.A_LEAF_STEM_ARC, self.A_LEAF_STEM_LABEL,
            self.A_GROWTH_DIR, self.A_GROWTH_DIR_LABEL,
            self.A_PLANT_HEIGHT, self.A_PLANT_HEIGHT_LABEL,
            self.A_PLANT_CROWN, self.A_PLANT_CROWN_LABEL,
        ]:
            self._remove_actor(nm)

        self._update_stem_cylinders()
        self._update_stem_length_paths()
        self.plotter.render()


    def _get_stem_points(self, inst_id: Optional[int] = None):
        if self.session.cloud is None:
            return None
        stem_label = self.session.semantic_map.get("stem")
        if stem_label is None:
            return None
        sem_map = self._get_instance_sem_map()
        if inst_id is not None:
            stem_insts = [int(inst_id)] if sem_map.get(int(inst_id)) == int(stem_label) else []
        else:
            stem_insts = [iid for iid, sem in sem_map.items() if int(sem) == int(stem_label)]
        if not stem_insts:
            return None
        inst = self.session.get_full_inst()
        mask = np.isin(inst, np.array(stem_insts, dtype=inst.dtype))
        return self.session.get_full_xyz()[mask]


    def _show_stem_instance_scene(self, inst_id: int):
        stem_pts = self._get_stem_points(inst_id)
        if stem_pts is None or len(stem_pts) == 0:
            return
        self._remove_actor(self.A_CLOUD_INST)
        self._actor_cloud_inst = None
        self._remove_actor(self.A_CLOUD_FULL)
        self._actor_cloud_full = self.plotter.add_points(
            pv.PolyData(stem_pts), color=(0.6, 0.4, 0.2),
            name=self.A_CLOUD_FULL, render_points_as_spheres=True, point_size=3
        )
        for nm in [
            self.A_BASE_SAVED, self.A_TIP_SAVED, self.A_CTRL_SAVED, self.A_WIDTH_SAVED, self.A_WIDTH_CTRL_SAVED,
            self.A_BASE_TEMP, self.A_TIP_TEMP, self.A_CTRL_TEMP, self.A_WIDTH_TEMP, self.A_WIDTH_CTRL_TEMP,
            self.A_LABELS_SAVED, self.A_LABELS_TEMP,
            self.A_LINE_CENTER_CUR, self.A_LINE_WIDTH_CUR,
            self.A_LINE_CENTER_CACHED, self.A_LINE_WIDTH_CACHED,
            self.A_MEASURE_LINE, self.A_MEASURE_P1, self.A_MEASURE_P2,
            self.A_VIEW_CENTER, self.A_AABB, self.A_OBB,
            self.A_OBB_DIM_L, self.A_OBB_DIM_W, self.A_OBB_DIM_H, self.A_OBB_DIM_LABELS,
            self.A_LEAF_INCLINE_N, self.A_LEAF_INCLINE_Z, self.A_LEAF_INCLINE_ARC, self.A_LEAF_INCLINE_LABEL,
            self.A_LEAF_STEM_LEAF, self.A_LEAF_STEM_STEM, self.A_LEAF_STEM_ARC, self.A_LEAF_STEM_LABEL,
            self.A_GROWTH_DIR, self.A_GROWTH_DIR_LABEL,
            self.A_PLANT_HEIGHT, self.A_PLANT_HEIGHT_LABEL,
            self.A_PLANT_CROWN, self.A_PLANT_CROWN_LABEL,
        ]:
            self._remove_actor(nm)
        self._update_stem_cylinders(inst_ids=[int(inst_id)])
        self._update_stem_length_paths(inst_ids=[int(inst_id)])
        self.plotter.render()


    def _show_obb_instance_scene(self, inst_id: int):
        pts = self.session.get_instance_points(inst_id)
        if pts is None or len(pts) == 0:
            return
        self._remove_actor(self.A_CLOUD_INST)
        self._actor_cloud_inst = None
        self._remove_actor(self.A_CLOUD_FULL)
        self._actor_cloud_full = self.plotter.add_points(
            pv.PolyData(pts), color=(0.7, 0.7, 0.7),
            name=self.A_CLOUD_FULL, render_points_as_spheres=True, point_size=3
        )
        for nm in [
            self.A_BASE_SAVED, self.A_TIP_SAVED, self.A_CTRL_SAVED, self.A_WIDTH_SAVED, self.A_WIDTH_CTRL_SAVED,
            self.A_BASE_TEMP, self.A_TIP_TEMP, self.A_CTRL_TEMP, self.A_WIDTH_TEMP, self.A_WIDTH_CTRL_TEMP,
            self.A_LABELS_SAVED, self.A_LABELS_TEMP,
            self.A_LINE_CENTER_CUR, self.A_LINE_WIDTH_CUR,
            self.A_LINE_CENTER_CACHED, self.A_LINE_WIDTH_CACHED,
            self.A_MEASURE_LINE, self.A_MEASURE_P1, self.A_MEASURE_P2,
            self.A_VIEW_CENTER, self.A_AABB, self.A_OBB,
            self.A_LEAF_INCLINE_N, self.A_LEAF_INCLINE_Z, self.A_LEAF_INCLINE_ARC, self.A_LEAF_INCLINE_LABEL,
            self.A_LEAF_STEM_LEAF, self.A_LEAF_STEM_STEM, self.A_LEAF_STEM_ARC, self.A_LEAF_STEM_LABEL,
            self.A_GROWTH_DIR, self.A_GROWTH_DIR_LABEL,
            self.A_PLANT_HEIGHT, self.A_PLANT_HEIGHT_LABEL,
            self.A_PLANT_CROWN, self.A_PLANT_CROWN_LABEL,
        ]:
            self._remove_actor(nm)
        self._update_obb_dimension_display()
        self.plotter.render()


    def _show_annotate_scene(self):
        if self.session.ds is None:
            return

        ds = self.session.get_ds_points()

        self._remove_actor(self.A_CLOUD_FULL)
        self._remove_actor(self.A_CLOUD_INST)
        self._actor_cloud_full = None

        poly = pv.PolyData(ds)
        if self._get_view_mode() == self.VIEW_LABEL and self.session.point_labels is not None:
            src_idx = self.session.ds.src_indices.astype(np.int64)
            if len(self.session.point_labels) >= np.max(src_idx) + 1:
                ds_labels = self.session.point_labels[src_idx]
                poly["rgb"] = colors_from_labels(ds_labels)

        self._actor_cloud_inst = self.plotter.add_mesh(
            poly, name=self.A_CLOUD_INST,
            scalars="rgb" if "rgb" in poly.array_names else None,
            rgb="rgb" in poly.array_names,
            color=(0.7, 0.7, 0.7),
            render_points_as_spheres=True, point_size=5
        )

        self._update_markers_saved()
        self._update_markers_temp()
        self._update_labels_saved()
        self._update_labels_temp()
        self._update_lines()
        self._update_measure_display()
        self.plotter.render()

    # ----------------------------
    # recommend width hook
    # ----------------------------

    def _update_markers_saved(self):
        if self.session.ds is None:
            return
        ds = self.session.get_ds_points()

        if self.session.base_idx is not None:
            p = ds[int(self.session.base_idx)][None, :]
            self._add_points_actor(self.A_BASE_SAVED, p, "red", 16)
        else:
            self._remove_actor(self.A_BASE_SAVED)

        if self.session.tip_idx is not None:
            p = ds[int(self.session.tip_idx)][None, :]
            self._add_points_actor(self.A_TIP_SAVED, p, "blue", 16)
        else:
            self._remove_actor(self.A_TIP_SAVED)

        if len(self.session.ctrl_indices) > 0:
            pts = ds[np.array(self.session.ctrl_indices, dtype=int)]
            self._add_points_actor(self.A_CTRL_SAVED, pts, "green", 14)
        else:
            self._remove_actor(self.A_CTRL_SAVED)

        w_pts = []
        if getattr(self.session, "width_w1_idx", None) is not None:
            w_pts.append(ds[int(self.session.width_w1_idx)])
        if getattr(self.session, "width_w2_idx", None) is not None:
            w_pts.append(ds[int(self.session.width_w2_idx)])
        if len(w_pts) > 0:
            self._add_points_actor(self.A_WIDTH_SAVED, np.asarray(w_pts), "orange", 14)
        else:
            self._remove_actor(self.A_WIDTH_SAVED)

        if len(self.session.width_ctrl_indices) > 0:
            pts = ds[np.array(self.session.width_ctrl_indices, dtype=int)]
            self._add_points_actor(self.A_WIDTH_CTRL_SAVED, pts, "cyan", 12)
        else:
            self._remove_actor(self.A_WIDTH_CTRL_SAVED)


    def _update_markers_temp(self):
        if self.session.ds is None:
            return
        ds = self.session.get_ds_points()

        if self.temp_base_idx is not None:
            p = ds[int(self.temp_base_idx)][None, :]
            self._add_points_actor(self.A_BASE_TEMP, p, "red", 18)
        else:
            self._remove_actor(self.A_BASE_TEMP)

        if self.temp_tip_idx is not None:
            p = ds[int(self.temp_tip_idx)][None, :]
            self._add_points_actor(self.A_TIP_TEMP, p, "blue", 18)
        else:
            self._remove_actor(self.A_TIP_TEMP)

        if len(self.temp_ctrl_indices) > 0:
            pts = ds[np.array(self.temp_ctrl_indices, dtype=int)]
            self._add_points_actor(self.A_CTRL_TEMP, pts, "green", 16)
        else:
            self._remove_actor(self.A_CTRL_TEMP)

        w_pts = []
        if self.temp_w1_idx is not None:
            w_pts.append(ds[int(self.temp_w1_idx)])
        if self.temp_w2_idx is not None:
            w_pts.append(ds[int(self.temp_w2_idx)])
        if len(w_pts) > 0:
            self._add_points_actor(self.A_WIDTH_TEMP, np.asarray(w_pts), "orange", 16)
        else:
            self._remove_actor(self.A_WIDTH_TEMP)

        if len(self.temp_width_ctrl_indices) > 0:
            pts = ds[np.array(self.temp_width_ctrl_indices, dtype=int)]
            self._add_points_actor(self.A_WIDTH_CTRL_TEMP, pts, "cyan", 12)
        else:
            self._remove_actor(self.A_WIDTH_CTRL_TEMP)


    def _update_labels_saved(self):
        if self.session.ds is None:
            self._remove_actor(self.A_LABELS_SAVED)
            return
        ds = self.session.get_ds_points()

        pts = []
        labels = []

        if self.session.base_idx is not None:
            pts.append(ds[int(self.session.base_idx)])
            labels.append("B1")

        if self.session.tip_idx is not None:
            pts.append(ds[int(self.session.tip_idx)])
            labels.append("T1")

        for idx, cid in zip(self.session.ctrl_indices, self.session.ctrl_ids):
            pts.append(ds[int(idx)])
            labels.append(f"C{cid}")

        if getattr(self.session, "width_w1_idx", None) is not None:
            pts.append(ds[int(self.session.width_w1_idx)])
            labels.append("W1")
        if getattr(self.session, "width_w2_idx", None) is not None:
            pts.append(ds[int(self.session.width_w2_idx)])
            labels.append("W2")

        for idx, cid in zip(self.session.width_ctrl_indices, self.session.width_ctrl_ids):
            pts.append(ds[int(idx)])
            labels.append(f"WC{cid}")

        if len(pts) > 0:
            self._add_labels_actor(self.A_LABELS_SAVED, np.asarray(pts), labels)
        else:
            self._remove_actor(self.A_LABELS_SAVED)


    def _update_labels_temp(self):
        if self.session.ds is None:
            self._remove_actor(self.A_LABELS_TEMP)
            return
        ds = self.session.get_ds_points()

        pts = []
        labels = []

        if self.temp_base_idx is not None:
            pts.append(ds[int(self.temp_base_idx)])
            labels.append("B*")
        if self.temp_tip_idx is not None:
            pts.append(ds[int(self.temp_tip_idx)])
            labels.append("T*")
        for i, idx in enumerate(self.temp_ctrl_indices, start=1):
            pts.append(ds[int(idx)])
            labels.append(f"C*{i}")
        if self.temp_w1_idx is not None:
            pts.append(ds[int(self.temp_w1_idx)])
            labels.append("W1*")
        if self.temp_w2_idx is not None:
            pts.append(ds[int(self.temp_w2_idx)])
            labels.append("W2*")

        for i, idx in enumerate(self.temp_width_ctrl_indices, start=1):
            pts.append(ds[int(idx)])
            labels.append(f"WC*{i}")

        if len(pts) > 0:
            self._add_labels_actor(self.A_LABELS_TEMP, np.asarray(pts), labels)
        else:
            self._remove_actor(self.A_LABELS_TEMP)


    def _update_measure_display(self):
        self._remove_actor(self.A_MEASURE_LINE)
        self._remove_actor(self.A_MEASURE_P1)
        self._remove_actor(self.A_MEASURE_P2)
        if self.temp_measure_p1 is None:
            return
        self._add_points_actor(self.A_MEASURE_P1, np.asarray([self.temp_measure_p1]), "yellow", 14)
        if self.temp_measure_p2 is None:
            return
        self._add_points_actor(self.A_MEASURE_P2, np.asarray([self.temp_measure_p2]), "yellow", 14)
        self._add_polyline_actor(
            self.A_MEASURE_LINE,
            np.asarray([self.temp_measure_p1, self.temp_measure_p2]),
            "yellow",
            3
        )
        dist = float(np.linalg.norm(self.temp_measure_p2 - self.temp_measure_p1))
        self._update_status(self.tr("测距结果：{dist:.6f}", dist=dist))

    # ----------------------------
    # growth direction / plant measurement
    # ----------------------------

    def _plant_scale(self) -> float:
        if self.session.cloud is None:
            return 0.1
        pts = self.session.get_full_xyz()
        if pts is None or len(pts) == 0:
            return 0.1
        diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        return max(diag * 0.25, 0.05)


    def _update_growth_direction_display(self):
        self._remove_actor(self.A_GROWTH_DIR)
        self._remove_actor(self.A_GROWTH_DIR_LABEL)
        if self.session.growth_direction is None:
            return
        if hasattr(self, "btn_toggle_growth_dir") and not self.btn_toggle_growth_dir.isChecked():
            return
        origin = self.session.growth_origin
        direction = np.asarray(self.session.growth_direction, dtype=np.float64)
        if origin is None:
            pts = self.session.get_full_xyz()
            if pts is None or len(pts) == 0:
                return
            proj = pts @ direction
            origin = pts[int(np.argmin(proj))]
        scale = self._plant_scale() * 1.6
        p0 = np.asarray(origin, dtype=np.float64)
        p1 = p0 + direction * scale
        self._add_polyline_actor(self.A_GROWTH_DIR, np.asarray([p0, p1]), "purple", 4)
        self._add_labels_actor(self.A_GROWTH_DIR_LABEL, np.asarray([p1]), [self.tr("Z轴")])


    def _local_to_world(self, origin: np.ndarray, basis: np.ndarray, xyz_local: np.ndarray) -> np.ndarray:
        return origin + xyz_local[0] * basis[:, 0] + xyz_local[1] * basis[:, 1] + xyz_local[2] * basis[:, 2]


    def _update_plant_measurement_display(self):
        self._remove_actor(self.A_PLANT_HEIGHT)
        self._remove_actor(self.A_PLANT_HEIGHT_LABEL)
        self._remove_actor(self.A_PLANT_CROWN)
        self._remove_actor(self.A_PLANT_CROWN_LABEL)
        if not hasattr(self.session, "plant_measurements"):
            return
        meas = self.session.plant_measurements or {}
        ext = meas.get("extents")
        if not isinstance(ext, dict):
            return
        show_height = True
        show_crown = True
        if hasattr(self, "btn_toggle_plant_height"):
            show_height = self.btn_toggle_plant_height.isChecked()
        if hasattr(self, "btn_toggle_plant_crown"):
            show_crown = self.btn_toggle_plant_crown.isChecked()
        if not show_height and not show_crown:
            return

        origin = meas.get("origin")
        basis = meas.get("basis")
        if origin is None:
            origin = self.session.growth_origin
        if basis is None:
            basis = self.session.growth_basis
        if origin is None or basis is None:
            return
        origin = np.asarray(origin, dtype=np.float64).reshape(3,)
        basis = np.asarray(basis, dtype=np.float64).reshape(3, 3)

        xmin = float(ext.get("x_min", 0.0))
        xmax = float(ext.get("x_max", 0.0))
        ymin = float(ext.get("y_min", 0.0))
        ymax = float(ext.get("y_max", 0.0))
        zmin = float(ext.get("z_min", 0.0))
        zmax = float(ext.get("z_max", 0.0))

        if show_height:
            p0 = self._local_to_world(origin, basis, np.array([0.0, 0.0, zmin], dtype=np.float64))
            p1 = self._local_to_world(origin, basis, np.array([0.0, 0.0, zmax], dtype=np.float64))
            self._add_polyline_actor(self.A_PLANT_HEIGHT, np.asarray([p0, p1]), "magenta", 4)
            height = meas.get("height", None)
            label = self.tr("株高={height:.3f}", height=float(height)) if height is not None else self.tr("株高")
            self._add_labels_actor(self.A_PLANT_HEIGHT_LABEL, np.asarray([(p0 + p1) / 2.0]), [label])

        if show_crown:
            zmid = 0.5 * (zmin + zmax)
            p0 = self._local_to_world(origin, basis, np.array([xmin, ymin, zmid], dtype=np.float64))
            p1 = self._local_to_world(origin, basis, np.array([xmax, ymax, zmid], dtype=np.float64))
            self._add_polyline_actor(self.A_PLANT_CROWN, np.asarray([p0, p1]), "teal", 4)
            crown = meas.get("crown_width", None)
            label = self.tr("冠幅={crown:.3f}", crown=float(crown)) if crown is not None else self.tr("冠幅")
            self._add_labels_actor(self.A_PLANT_CROWN_LABEL, np.asarray([(p0 + p1) / 2.0]), [label])

    # ----------------------------
    # lines
    # ----------------------------

    def _update_lines(self):
        if self.session.centerline_result is not None:
            cl = np.asarray(self.session.centerline_result.smooth_points, dtype=np.float64)
            self._add_polyline_actor(self.A_LINE_CENTER_CUR, cl, "red", 4)
            self._remove_actor(self.A_LINE_CENTER_CACHED)
        else:
            self._remove_actor(self.A_LINE_CENTER_CUR)

        width_path = getattr(self.session, "width_path_points", None)
        if width_path is not None and len(width_path) >= 2:
            self._add_polyline_actor(self.A_LINE_WIDTH_CUR, np.asarray(width_path), "green", 6)
            self._remove_actor(self.A_LINE_WIDTH_CACHED)
        else:
            self._remove_actor(self.A_LINE_WIDTH_CUR)

        if (self.session.centerline_result is None or getattr(self.session, "width_path_points", None) is None) and (self.session.current_inst_id is not None):
            cl_cached, width_cached = self.session.get_cached_display(self.session.current_inst_id)

            if self.session.centerline_result is None:
                if cl_cached is not None and len(cl_cached) >= 2:
                    self._add_polyline_actor(self.A_LINE_CENTER_CACHED, np.asarray(cl_cached), "red", 4)
                else:
                    self._remove_actor(self.A_LINE_CENTER_CACHED)
            else:
                self._remove_actor(self.A_LINE_CENTER_CACHED)

            if getattr(self.session, "width_path_points", None) is None:
                if width_cached is not None and len(width_cached) >= 2:
                    self._add_polyline_actor(self.A_LINE_WIDTH_CACHED, np.asarray(width_cached), "green", 6)
                else:
                    self._remove_actor(self.A_LINE_WIDTH_CACHED)
            else:
                self._remove_actor(self.A_LINE_WIDTH_CACHED)
        else:
            self._remove_actor(self.A_LINE_CENTER_CACHED)
            self._remove_actor(self.A_LINE_WIDTH_CACHED)
        self._update_leaf_angle_visuals()

    def _clear_leaf_angle_actors(self):
        for nm in [
            self.A_LEAF_INCLINE_N, self.A_LEAF_INCLINE_Z, self.A_LEAF_INCLINE_ARC, self.A_LEAF_INCLINE_LABEL,
            self.A_LEAF_STEM_LEAF, self.A_LEAF_STEM_STEM, self.A_LEAF_STEM_ARC, self.A_LEAF_STEM_LABEL,
        ]:
            self._remove_actor(nm)

    def _leaf_angle_scale(self) -> float:
        scale = None
        if self.session.centerline_result is not None:
            pts = getattr(self.session.centerline_result, "smooth_points", None)
            if pts is not None and len(pts) >= 2:
                seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
                length = float(np.sum(seg))
                if length > 0:
                    scale = 0.2 * length
        if scale is None and self.session.leaf_pts is not None and len(self.session.leaf_pts) > 0:
            pts = np.asarray(self.session.leaf_pts, dtype=np.float64)
            diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
            if diag > 0:
                scale = 0.2 * diag
        if scale is None:
            scale = 0.05
        return max(scale, 0.02)

    def _update_leaf_angle_visuals(self):
        if (not self.annotating) or self.annotate_semantic != "leaf":
            self._clear_leaf_angle_actors()
            return
        if self.session.current_inst_id is None:
            self._clear_leaf_angle_actors()
            return
        ann = self.session.annotations.get(int(self.session.current_inst_id))
        if ann is None:
            self._clear_leaf_angle_actors()
            return

        def _norm_vec(vec):
            if vec is None:
                return None
            v = np.asarray(vec, dtype=np.float64)
            if v.shape != (3,):
                return None
            n = float(np.linalg.norm(v))
            if n <= 1e-12:
                return None
            return v / n

        scale = self._leaf_angle_scale()

        inc = ann.get("leaf_inclination_viz")
        if inc is None:
            self._remove_actor(self.A_LEAF_INCLINE_N)
            self._remove_actor(self.A_LEAF_INCLINE_Z)
            self._remove_actor(self.A_LEAF_INCLINE_ARC)
            self._remove_actor(self.A_LEAF_INCLINE_LABEL)
        else:
            origin = inc.get("origin")
            origin = np.asarray(origin, dtype=np.float64) if origin is not None else None
            normal = _norm_vec(inc.get("normal"))
            z_axis = _norm_vec(inc.get("z_axis") or [0.0, 0.0, 1.0])
            if origin is None or origin.shape != (3,) or normal is None or z_axis is None:
                self._remove_actor(self.A_LEAF_INCLINE_N)
                self._remove_actor(self.A_LEAF_INCLINE_Z)
                self._remove_actor(self.A_LEAF_INCLINE_ARC)
                self._remove_actor(self.A_LEAF_INCLINE_LABEL)
            else:
                if float(np.dot(normal, z_axis)) < 0:
                    normal = -normal
                p_n = origin + normal * scale
                p_z = origin + z_axis * scale
                self._add_polyline_actor(self.A_LEAF_INCLINE_N, np.asarray([origin, p_n]), "yellow", 3)
                self._add_polyline_actor(self.A_LEAF_INCLINE_Z, np.asarray([origin, p_z]), "cyan", 3)
                dot = float(np.clip(np.dot(normal, z_axis), -1.0, 1.0))
                angle = float(np.arccos(dot))
                v = z_axis - dot * normal
                v_norm = float(np.linalg.norm(v))
                if angle > 1e-3 and v_norm > 1e-6:
                    v = v / v_norm
                    arc_pts = []
                    for t in np.linspace(0.0, angle, 21):
                        arc_pts.append(origin + (scale * 0.8) * (np.cos(t) * normal + np.sin(t) * v))
                    self._add_polyline_actor(self.A_LEAF_INCLINE_ARC, np.asarray(arc_pts), "orange", 3)
                    mid = angle / 2.0
                    label_pt = origin + (scale * 0.9) * (np.cos(mid) * normal + np.sin(mid) * v)
                    label_angle = inc.get("angle", np.degrees(angle))
                    self._add_labels_actor(self.A_LEAF_INCLINE_LABEL, np.asarray([label_pt]), [f"{float(label_angle):.1f}°"])
                else:
                    self._remove_actor(self.A_LEAF_INCLINE_ARC)
                    self._remove_actor(self.A_LEAF_INCLINE_LABEL)

        leaf_stem = ann.get("leaf_stem_angle_viz")
        if leaf_stem is None:
            self._remove_actor(self.A_LEAF_STEM_LEAF)
            self._remove_actor(self.A_LEAF_STEM_STEM)
            self._remove_actor(self.A_LEAF_STEM_ARC)
            self._remove_actor(self.A_LEAF_STEM_LABEL)
        else:
            origin = leaf_stem.get("origin")
            origin = np.asarray(origin, dtype=np.float64) if origin is not None else None
            leaf_dir = _norm_vec(leaf_stem.get("leaf_dir"))
            stem_dir = _norm_vec(leaf_stem.get("stem_dir"))
            if origin is None or origin.shape != (3,) or leaf_dir is None or stem_dir is None:
                self._remove_actor(self.A_LEAF_STEM_LEAF)
                self._remove_actor(self.A_LEAF_STEM_STEM)
                self._remove_actor(self.A_LEAF_STEM_ARC)
                self._remove_actor(self.A_LEAF_STEM_LABEL)
            else:
                self._add_polyline_actor(
                    self.A_LEAF_STEM_LEAF,
                    np.asarray([origin, origin + leaf_dir * scale]),
                    "green",
                    3
                )
                self._add_polyline_actor(
                    self.A_LEAF_STEM_STEM,
                    np.asarray([origin, origin + stem_dir * scale]),
                    "orange",
                    3
                )
                dot = float(np.clip(np.dot(stem_dir, leaf_dir), -1.0, 1.0))
                angle = float(np.arccos(dot))
                v = leaf_dir - dot * stem_dir
                v_norm = float(np.linalg.norm(v))
                if angle > 1e-3 and v_norm > 1e-6:
                    v = v / v_norm
                    arc_pts = []
                    for t in np.linspace(0.0, angle, 21):
                        arc_pts.append(origin + (scale * 0.8) * (np.cos(t) * stem_dir + np.sin(t) * v))
                    self._add_polyline_actor(self.A_LEAF_STEM_ARC, np.asarray(arc_pts), "orange", 3)
                    mid = angle / 2.0
                    label_pt = origin + (scale * 0.9) * (np.cos(mid) * stem_dir + np.sin(mid) * v)
                    label_angle = leaf_stem.get("angle", np.degrees(angle))
                    self._add_labels_actor(self.A_LEAF_STEM_LABEL, np.asarray([label_pt]), [f"{float(label_angle):.1f}°"])
                else:
                    self._remove_actor(self.A_LEAF_STEM_ARC)
                    self._remove_actor(self.A_LEAF_STEM_LABEL)

    # ----------------------------
    # point list UI
    # ----------------------------

    def _refresh_scene(self, mode: str = None):
        if mode is not None and self._get_view_mode() != mode:
            self.combo_view.blockSignals(True)
            self._set_combo_current_by_data(self.combo_view, mode)
            self.combo_view.blockSignals(False)
        if self.session.cloud is None:
            return
        stem_only = False
        if hasattr(self, "btn_toggle_stem_cyl") and self.btn_toggle_stem_cyl.isChecked():
            stem_only = True
        if hasattr(self, "btn_toggle_stem_path") and self.btn_toggle_stem_path.isChecked():
            stem_only = True
        if stem_only:
            self._show_stem_only_scene()
            self._update_growth_direction_display()
            self._update_plant_measurement_display()
            self.plotter.render()
            return
        if not self.annotating:
            self._show_browse_scene()
        else:
            self._show_annotate_scene()
        mode_text = self._get_view_mode()
        if mode_text == self.VIEW_RGB:
            self.act_view_rgb.setChecked(True)
        elif mode_text == self.VIEW_SEM:
            self.act_view_sem.setChecked(True)
        elif mode_text == self.VIEW_INST:
            self.act_view_inst.setChecked(True)
        elif mode_text == self.VIEW_LABEL:
            self.act_view_label.setChecked(True)
        self._refresh_bbox_actors()
        self._update_view_legend()
        self._update_growth_direction_display()
        self._update_plant_measurement_display()
        self.plotter.render()


    def _color_for_label(self, label_id: int) -> np.ndarray:
        if label_id == -1:
            return np.array([128, 128, 128], dtype=np.uint8)
        if label_id == 0:
            return np.array([255, 0, 0], dtype=np.uint8)
        if label_id == 1:
            return np.array([0, 255, 0], dtype=np.uint8)
        if label_id == 2:
            return np.array([0, 0, 255], dtype=np.uint8)
        if label_id == 3:
            return np.array([139, 69, 19], dtype=np.uint8)
        if label_id == 4:
            return np.array([128, 0, 128], dtype=np.uint8)
        if label_id == 5:
            return np.array([255, 165, 0], dtype=np.uint8)
        return stable_color_from_id(int(label_id))


    def _update_view_legend(self):
        if not hasattr(self, "view_legend_layout"):
            return
        if self.session.cloud is None:
            self._set_view_legend([])
            return

        mode = self._get_view_mode()
        labels = []
        if mode == self.VIEW_SEM:
            if not self.annotating:
                labels = np.unique(self.session.get_full_sem().astype(np.int64)).tolist()
            else:
                labels = []
        elif mode == self.VIEW_INST:
            labels = np.unique(self.session.get_full_inst().astype(np.int64)).tolist()
        elif mode == self.VIEW_LABEL:
            if self.annotating:
                if self.session.point_labels is not None:
                    labels = np.unique(self.session.point_labels.astype(np.int64)).tolist()
                else:
                    labels = []
            else:
                full_labels = self._get_full_phenotype_labels()
                if full_labels is not None:
                    labels = np.unique(full_labels.astype(np.int64)).tolist()
                else:
                    labels = []
        else:
            labels = []

        labels = [int(v) for v in labels if v >= -1]
        labels = sorted(set(labels))
        self._set_view_legend(labels)


    def _set_view_legend(self, labels):
        self._clear_layout(self.view_legend_layout)
        if not labels:
            empty = QtWidgets.QLabel(self.tr("无"))
            empty.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.view_legend_layout.addWidget(empty)
            return
        for label_id in labels:
            rgb = self._color_for_label(int(label_id))
            row = QtWidgets.QHBoxLayout()
            swatch = QtWidgets.QLabel()
            swatch.setFixedSize(LABEL_SWATCH_SIZE, LABEL_SWATCH_SIZE)
            swatch.setStyleSheet(f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]});")
            row.addWidget(swatch)
            row.addWidget(QtWidgets.QLabel(str(label_id)))
            row.addStretch(1)
            self.view_legend_layout.addLayout(row)


    def _get_bbox_points(self) -> np.ndarray:
        if self.session.cloud is None:
            return None
        if self.annotating and self.session.leaf_pts is not None and len(self.session.leaf_pts) > 0:
            return self.session.leaf_pts
        return self.session.get_full_xyz()


    def _refresh_bbox_actors(self):
        if not hasattr(self, "btn_toggle_aabb"):
            return
        if not self.btn_toggle_aabb.isChecked():
            self._remove_actor(self.A_AABB)
        else:
            self._update_aabb_actor()
        if not self.btn_toggle_obb.isChecked():
            self._remove_actor(self.A_OBB)
        else:
            self._update_obb_actor()
        self._update_bbox_info()
        self._update_stem_cylinders()


    def _update_aabb_actor(self):
        pts = self._get_bbox_points()
        if pts is None or len(pts) == 0:
            self._remove_actor(self.A_AABB)
            return
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        center = (mins + maxs) / 2.0
        lengths = (maxs - mins).tolist()
        cube = pv.Cube(center=center, x_length=lengths[0], y_length=lengths[1], z_length=lengths[2])
        self._remove_actor(self.A_AABB)
        self.plotter.add_mesh(cube, color="orange", style="wireframe", line_width=2, name=self.A_AABB)


    def _update_obb_actor(self):
        pts = self._get_bbox_points()
        if pts is None or len(pts) < 3:
            self._remove_actor(self.A_OBB)
            return
        mean = pts.mean(axis=0)
        centered = pts - mean
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        R = eigvecs[:, order]
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1
        local = centered @ R
        mins = local.min(axis=0)
        maxs = local.max(axis=0)
        lengths = (maxs - mins).tolist()
        center_local = (mins + maxs) / 2.0
        cube = pv.Cube(center=center_local, x_length=lengths[0], y_length=lengths[1], z_length=lengths[2])
        mat = np.eye(4)
        mat[:3, :3] = R
        mat[:3, 3] = mean
        cube = cube.transform(mat, inplace=False)
        self._remove_actor(self.A_OBB)
        self.plotter.add_mesh(cube, color="green", style="wireframe", line_width=2, name=self.A_OBB)


    def _clear_obb_dimension_display(self):
        for name in [self.A_OBB_DIM_L, self.A_OBB_DIM_W, self.A_OBB_DIM_H, self.A_OBB_DIM_LABELS]:
            self._remove_actor(name)


    def _update_obb_dimension_display(self):
        self._clear_obb_dimension_display()
        if not self.annotating or self.session.current_inst_id is None:
            return
        key = self._get_annotation_semantic_key()
        if key not in ["flower", "fruit"]:
            return
        ann = self.session.annotations.get(int(self.session.current_inst_id), {})
        obb_key = "flower_obb" if key == "flower" else "fruit_obb"
        obb = ann.get(obb_key)
        if obb is None:
            return
        center = np.asarray(obb.get("center", [0, 0, 0]), dtype=np.float64)
        lengths = obb.get("lengths")
        R = obb.get("rotation")
        if lengths is None or R is None:
            return
        lengths = np.asarray(lengths, dtype=np.float64)
        R = np.asarray(R, dtype=np.float64)
        axes = [R[:, 0], R[:, 1], R[:, 2]]
        order = np.argsort(lengths)[::-1]
        dims = [("L", lengths[order[0]], axes[order[0]]),
                ("W", lengths[order[1]], axes[order[1]]),
                ("H", lengths[order[2]], axes[order[2]])]
        colors = {"L": "red", "W": "green", "H": "blue"}
        label_pts = []
        label_texts = []
        for tag, L, axis in dims:
            p1 = center - axis * (L / 2.0)
            p2 = center + axis * (L / 2.0)
            name = {"L": self.A_OBB_DIM_L, "W": self.A_OBB_DIM_W, "H": self.A_OBB_DIM_H}[tag]
            self._add_polyline_actor(name, np.asarray([p1, p2]), colors[tag], 3)
            label_pts.append((p1 + p2) / 2.0)
            label_texts.append(f"{tag}={float(L):.3f}")
        if label_pts:
            self._add_labels_actor(self.A_OBB_DIM_LABELS, np.asarray(label_pts), label_texts)


    def _clear_stem_cylinders(self):
        for name in self._stem_cylinder_actors:
            self._remove_actor(name)
        self._stem_cylinder_actors = []


    def _clear_stem_length_paths(self):
        for name in self._stem_length_actors:
            self._remove_actor(name)
        self._stem_length_actors = []


    def _update_stem_cylinders(self, inst_ids: Optional[list] = None):
        if not hasattr(self, "btn_toggle_stem_cyl"):
            return
        if not self.btn_toggle_stem_cyl.isChecked():
            self._clear_stem_cylinders()
            return
        self._clear_stem_cylinders()
        if self.session.cloud is None:
            return
        items = self.session.annotations.items()
        if inst_ids is not None:
            items = [(inst_id, self.session.annotations.get(inst_id, {})) for inst_id in inst_ids]
        for inst_id, ann in items:
            cyl = ann.get("stem_cylinder")
            if cyl is None:
                continue
            center = np.asarray(cyl.get("center", [0, 0, 0]), dtype=np.float64)
            axis = np.asarray(cyl.get("axis", [0, 0, 1]), dtype=np.float64)
            height = float(cyl.get("height", 0.0))
            radius = float(cyl.get("radius", 0.0))
            if height <= 0.0 or radius <= 0.0:
                continue
            cyl_mesh = pv.Cylinder(center=center, direction=axis, radius=radius, height=height)
            name = f"stem_cyl_{int(inst_id)}"
            self.plotter.add_mesh(cyl_mesh, color="brown", style="wireframe", line_width=2, name=name)
            self._stem_cylinder_actors.append(name)


    def _update_stem_length_paths(self, inst_ids: Optional[list] = None):
        if not hasattr(self, "btn_toggle_stem_path"):
            return
        if not self.btn_toggle_stem_path.isChecked():
            self._clear_stem_length_paths()
            return
        self._clear_stem_length_paths()
        items = self.session.annotations.items()
        if inst_ids is not None:
            items = [(inst_id, self.session.annotations.get(inst_id, {})) for inst_id in inst_ids]
        for inst_id, ann in items:
            path = ann.get("stem_length_path")
            if path is None or len(path) < 2:
                continue
            pts = np.asarray(path, dtype=np.float64)
            name = f"{self.A_STEM_LENGTH}_{int(inst_id)}"
            self._add_polyline_actor(name, pts, "brown", 4)
            self._stem_length_actors.append(name)


    def _update_bbox_info(self):
        if not hasattr(self, "bbox_info"):
            return
        if self.session.cloud is None:
            self.bbox_info.setText(f"{self.tr('AABB：-')}\n{self.tr('OBB：-')}")
            return
        pts = self._get_bbox_points()
        if pts is None or len(pts) == 0:
            self.bbox_info.setText(f"{self.tr('AABB：-')}\n{self.tr('OBB：-')}")
            return

        aabb_text = self.tr("AABB：-")
        if self.btn_toggle_aabb.isChecked():
            mins = pts.min(axis=0)
            maxs = pts.max(axis=0)
            lengths = (maxs - mins).tolist()
            aabb_text = f"AABB：L={lengths[0]:.3f} W={lengths[1]:.3f} H={lengths[2]:.3f}"

        obb_text = self.tr("OBB：-")
        if self.btn_toggle_obb.isChecked() and len(pts) >= 3:
            mean = pts.mean(axis=0)
            centered = pts - mean
            cov = np.cov(centered, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            R = eigvecs[:, order]
            if np.linalg.det(R) < 0:
                R[:, 2] *= -1
            local = centered @ R
            mins = local.min(axis=0)
            maxs = local.max(axis=0)
            lengths = (maxs - mins).tolist()
            obb_text = f"OBB：L={lengths[0]:.3f} W={lengths[1]:.3f} H={lengths[2]:.3f}"

        self.bbox_info.setText(f"{aabb_text}\n{obb_text}")
