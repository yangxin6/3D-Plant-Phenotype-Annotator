# ui/main_window.py
import os
import numpy as np
import pyvista as pv
import vtk
from pyvistaqt import QtInteractor

from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QListWidgetItem

from core.annotation import LeafAnnotationSession, AnnotationParams
from core.schema import CloudSchema


def make_polyline_mesh(points: np.ndarray) -> pv.PolyData:
    n = len(points)
    poly = pv.PolyData(points)
    if n >= 2:
        lines = np.hstack([[n], np.arange(n)]).astype(np.int64)
        poly.lines = lines
    return poly


def stable_color_from_id(_id: int) -> np.ndarray:
    seed = (int(_id) * 2654435761) & 0xFFFFFFFF
    rs = np.random.RandomState(seed)
    return rs.randint(50, 256, size=3, dtype=np.uint8)


def colors_from_labels(labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(np.int64)
    uniq = np.unique(labels)
    lut = {int(k): stable_color_from_id(int(k)) for k in uniq}
    out = np.zeros((len(labels), 3), dtype=np.uint8)
    for k in uniq:
        out[labels == k] = lut[int(k)]
    return out


class LeafAnnotatorWindow(QtWidgets.QMainWindow):
    """
    - 左侧点列表：B1/T1/Ci/W1/W2，可删除并同步更新
    - 叶宽：用户可手选 W1/W2 做最短路径；并且当叶长点齐了后，可自动/手动推荐 W1/W2
    - 标注模式单视图增量更新：不 clear，不 reset_camera
    - 拾取：VTK PointPicker + Shift+左键（只拾取实例点云）
    """

    MODE_NONE = "NONE"
    MODE_BASE = "PICK_BASE"
    MODE_TIP = "PICK_TIP"
    MODE_CTRL = "PICK_CTRL"
    MODE_WIDTH = "PICK_WIDTH"

    # actor names
    A_CLOUD_FULL = "cloud_full"
    A_CLOUD_INST = "cloud_inst"

    A_BASE_SAVED = "mark_base_saved"
    A_TIP_SAVED = "mark_tip_saved"
    A_CTRL_SAVED = "mark_ctrl_saved"
    A_WIDTH_SAVED = "mark_width_saved"

    A_BASE_TEMP = "mark_base_temp"
    A_TIP_TEMP = "mark_tip_temp"
    A_CTRL_TEMP = "mark_ctrl_temp"
    A_WIDTH_TEMP = "mark_width_temp"

    A_LABELS_SAVED = "labels_saved"
    A_LABELS_TEMP = "labels_temp"

    A_LINE_CENTER_CUR = "line_center_current"
    A_LINE_WIDTH_CUR = "line_width_current"
    A_LINE_CENTER_CACHED = "line_center_cached"
    A_LINE_WIDTH_CACHED = "line_width_cached"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Corn Leaf 3D Annotation (PyVista)")

        schema = CloudSchema(xyz_slice=slice(0, 3), sem_col=-2, inst_col=-1, rgb_slice=slice(3, 6))
        self.session = LeafAnnotationSession(params=AnnotationParams(), schema=schema)

        self.annotating = False
        self.pick_mode = self.MODE_NONE

        # 临时点（开启期间）
        self.temp_base_idx = None
        self.temp_tip_idx = None
        self.temp_ctrl_indices = []

        # 叶宽两点：临时（开启期间）
        self.temp_w1_idx = None
        self.temp_w2_idx = None

        # VTK picker
        self._vtk_picker = vtk.vtkPointPicker()
        self._vtk_picker.SetTolerance(0.02)
        self._pick_observer_id = None

        # 记录当前实例点云 actor（picker 只拾取它）
        self._actor_cloud_inst = None

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QHBoxLayout(root)

        # ---------- left panel (scrollable) ----------
        panel_widget = QtWidgets.QWidget()
        panel = QtWidgets.QVBoxLayout(panel_widget)
        panel.setContentsMargins(0, 0, 0, 0)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setWidget(panel_widget)

        layout.addWidget(scroll, 0)

        self.btn_load = QtWidgets.QPushButton("加载整株点云")
        panel.addWidget(self.btn_load)

        panel.addSpacing(6)
        panel.addWidget(QtWidgets.QLabel("显示模式（浏览模式有效）"))
        self.combo_view = QtWidgets.QComboBox()
        self.combo_view.addItems(["RGB", "Semantic", "Instance"])
        panel.addWidget(self.combo_view)

        panel.addSpacing(6)
        panel.addWidget(QtWidgets.QLabel("实例 inst_id（用于标注）"))
        self.combo_inst = QtWidgets.QComboBox()
        panel.addWidget(self.combo_inst)

        self.btn_start_anno = QtWidgets.QPushButton("进入标注模式（只显示当前实例）")
        self.btn_back_browse = QtWidgets.QPushButton("返回浏览模式（显示整株）")
        panel.addWidget(self.btn_start_anno)
        panel.addWidget(self.btn_back_browse)

        panel.addSpacing(8)

        # Toggle buttons：开启/关闭
        self.btn_toggle_base = QtWidgets.QPushButton("叶基选择：关闭")
        self.btn_toggle_tip = QtWidgets.QPushButton("叶尖选择：关闭")
        self.btn_toggle_ctrl = QtWidgets.QPushButton("控制点选择：关闭")
        self.btn_toggle_width = QtWidgets.QPushButton("叶宽点选择：关闭 (W1/W2)")
        for b in [self.btn_toggle_base, self.btn_toggle_tip, self.btn_toggle_ctrl, self.btn_toggle_width]:
            b.setCheckable(True)

        panel.addWidget(self.btn_toggle_base)
        panel.addWidget(self.btn_toggle_tip)
        panel.addWidget(self.btn_toggle_ctrl)
        panel.addWidget(self.btn_toggle_width)

        # 叶长推荐/生成
        self.btn_recommend_length = QtWidgets.QPushButton("推荐叶长（B1→T1 最短路径）")
        self.btn_generate_length = QtWidgets.QPushButton("生成叶长（使用控制点）")
        panel.addWidget(self.btn_recommend_length)
        panel.addWidget(self.btn_generate_length)

        # ✅ 新增：推荐叶宽按钮（强制重新推荐并展示）
        self.btn_recommend_width = QtWidgets.QPushButton("推荐叶宽（最大叶宽）")
        panel.addWidget(self.btn_recommend_width)

        panel.addSpacing(8)

        # ------- point lists + delete -------
        panel.addWidget(QtWidgets.QLabel("点列表（选择编号后删除）"))

        self.list_base = QtWidgets.QListWidget()
        self.list_tip = QtWidgets.QListWidget()
        self.list_ctrl = QtWidgets.QListWidget()
        self.list_width = QtWidgets.QListWidget()

        self.list_base.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list_tip.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list_ctrl.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list_width.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        panel.addWidget(QtWidgets.QLabel("叶基 (B1)"))
        panel.addWidget(self.list_base)
        self.btn_del_base = QtWidgets.QPushButton("删除选中的叶基")
        panel.addWidget(self.btn_del_base)

        panel.addWidget(QtWidgets.QLabel("叶尖 (T1)"))
        panel.addWidget(self.list_tip)
        self.btn_del_tip = QtWidgets.QPushButton("删除选中的叶尖")
        panel.addWidget(self.btn_del_tip)

        panel.addWidget(QtWidgets.QLabel("控制点 (C1..Cn)"))
        panel.addWidget(self.list_ctrl)
        self.btn_del_ctrl = QtWidgets.QPushButton("删除选中的控制点")
        panel.addWidget(self.btn_del_ctrl)

        panel.addWidget(QtWidgets.QLabel("叶宽点 (W1/W2)"))
        panel.addWidget(self.list_width)
        self.btn_del_width = QtWidgets.QPushButton("删除选中的叶宽点")
        panel.addWidget(self.btn_del_width)

        panel.addSpacing(6)

        self.btn_clear_ctrl = QtWidgets.QPushButton("清空控制点（已保存）")
        self.btn_compute = QtWidgets.QPushButton("计算并保存（叶长 + 叶宽路径）")
        self.btn_export = QtWidgets.QPushButton("导出整株标注 JSON（所有已标注实例）")
        panel.addWidget(self.btn_clear_ctrl)
        panel.addWidget(self.btn_compute)
        panel.addWidget(self.btn_export)

        panel.addSpacing(10)
        self.info = QtWidgets.QLabel("状态：未加载")
        self.info.setWordWrap(True)
        panel.addWidget(self.info)
        panel.addStretch(1)

        # ---------- right: PyVista ----------
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor, 1)
        self.plotter.set_background("white")
        self.plotter.show_axes()

        # ---------- signals ----------
        self.btn_load.clicked.connect(self.on_load)
        self.combo_view.currentIndexChanged.connect(self.on_view_mode_changed)

        self.btn_start_anno.clicked.connect(self.on_start_annotation)
        self.btn_back_browse.clicked.connect(self.on_back_browse)
        self.combo_inst.currentIndexChanged.connect(self.on_inst_changed)

        self.btn_toggle_base.toggled.connect(self.on_toggle_base)
        self.btn_toggle_tip.toggled.connect(self.on_toggle_tip)
        self.btn_toggle_ctrl.toggled.connect(self.on_toggle_ctrl)
        self.btn_toggle_width.toggled.connect(self.on_toggle_width)

        self.btn_recommend_length.clicked.connect(self.on_recommend_length)
        self.btn_generate_length.clicked.connect(self.on_generate_length)

        self.btn_del_base.clicked.connect(self.on_delete_base)
        self.btn_del_tip.clicked.connect(self.on_delete_tip)
        self.btn_del_ctrl.clicked.connect(self.on_delete_ctrl)
        self.btn_del_width.clicked.connect(self.on_delete_width)

        self.btn_recommend_width.clicked.connect(self.on_recommend_width)  # ✅

        self.btn_clear_ctrl.clicked.connect(self.on_clear_ctrl)
        self.btn_compute.clicked.connect(self.on_compute)
        self.btn_export.clicked.connect(self.on_export_all)

        self._install_vtk_pick_observer()

        self._update_buttons()
        self._update_status("提示：标注模式用 Shift+左键 选点。")
        self._refresh_point_lists()

    # ----------------------------
    # status
    # ----------------------------
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

    def _update_buttons(self):
        in_anno = self.annotating
        for b in [
            self.btn_toggle_base, self.btn_toggle_tip, self.btn_toggle_ctrl, self.btn_toggle_width,
            self.btn_recommend_length, self.btn_generate_length,
            self.btn_recommend_width,  # ✅
            self.btn_del_base, self.btn_del_tip, self.btn_del_ctrl, self.btn_del_width,
            self.btn_clear_ctrl, self.btn_compute
        ]:
            b.setEnabled(in_anno)

        self.btn_export.setEnabled(self.session.cloud is not None and self.session.get_annotations_count() > 0)
        self.btn_back_browse.setEnabled(self.session.cloud is not None)
        self.btn_start_anno.setEnabled(self.session.cloud is not None and self.combo_inst.count() > 0)

    # ----------------------------
    # vtk picking (Shift+Left Click)
    # ----------------------------
    def _install_vtk_pick_observer(self):
        if self._pick_observer_id is not None:
            return

        iren = self.plotter.interactor

        def _on_left_press(obj, event):
            if (not self.annotating) or (self.pick_mode == self.MODE_NONE) or (self.session.ds is None):
                return
            if iren.GetShiftKey() == 0:
                return

            x, y = iren.GetEventPosition()
            renderer = self.plotter.renderer

            self._vtk_picker.PickFromListOn()
            self._vtk_picker.InitializePickList()
            if self._actor_cloud_inst is not None:
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
    def _remove_actor(self, name: str):
        try:
            self.plotter.remove_actor(name)
        except Exception:
            pass

    def _add_points_actor(self, name: str, pts: np.ndarray, color, point_size: int):
        self._remove_actor(name)
        if pts is None or len(pts) == 0:
            return None
        poly = pv.PolyData(pts)
        return self.plotter.add_mesh(
            poly, name=name, color=color,
            render_points_as_spheres=True, point_size=point_size
        )

    def _add_polyline_actor(self, name: str, pts: np.ndarray, color, line_width: int):
        self._remove_actor(name)
        if pts is None or len(pts) < 2:
            return None
        return self.plotter.add_mesh(
            make_polyline_mesh(np.asarray(pts, dtype=np.float64)),
            name=name, color=color, line_width=line_width
        )

    def _add_labels_actor(self, name: str, pts: np.ndarray, labels: list):
        self._remove_actor(name)
        if pts is None or len(pts) == 0:
            return None
        return self.plotter.add_point_labels(
            pts, labels,
            name=name,
            point_size=0,
            font_size=14,
            shape_opacity=0.25,
            always_visible=True
        )

    # ----------------------------
    # browsing polydata
    # ----------------------------
    def _get_full_cloud_polydata(self) -> pv.PolyData:
        xyz = self.session.get_full_xyz()
        poly = pv.PolyData(xyz)

        mode = self.combo_view.currentText()
        if mode == "RGB" and self.session.has_rgb():
            poly["rgb"] = self.session.get_full_rgb()
        elif mode == "Semantic":
            poly["rgb"] = colors_from_labels(self.session.get_full_sem())
        elif mode == "Instance":
            poly["rgb"] = colors_from_labels(self.session.get_full_inst())
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
        self.plotter.add_points(
            poly, scalars="rgb", rgb=True,
            name=self.A_CLOUD_FULL,
            render_points_as_spheres=True, point_size=3
        )

        for nm in [
            self.A_BASE_SAVED, self.A_TIP_SAVED, self.A_CTRL_SAVED, self.A_WIDTH_SAVED,
            self.A_BASE_TEMP, self.A_TIP_TEMP, self.A_CTRL_TEMP, self.A_WIDTH_TEMP,
            self.A_LABELS_SAVED, self.A_LABELS_TEMP,
            self.A_LINE_CENTER_CUR, self.A_LINE_WIDTH_CUR,
            self.A_LINE_CENTER_CACHED, self.A_LINE_WIDTH_CACHED
        ]:
            self._remove_actor(nm)

        self.plotter.render()

    def _show_annotate_scene(self):
        if self.session.ds is None:
            return

        ds = self.session.get_ds_points()

        self._remove_actor(self.A_CLOUD_FULL)
        self._remove_actor(self.A_CLOUD_INST)

        poly = pv.PolyData(ds)
        self._actor_cloud_inst = self.plotter.add_mesh(
            poly, name=self.A_CLOUD_INST,
            color=(0.7, 0.7, 0.7),
            render_points_as_spheres=True, point_size=5
        )

        self._update_markers_saved()
        self._update_markers_temp()
        self._update_labels_saved()
        self._update_labels_temp()
        self._update_lines()

        self.plotter.render()

    # ----------------------------
    # recommend width hook
    # ----------------------------
    def _maybe_recommend_width_and_refresh(self):
        if not self.annotating:
            return
        if self.session.base_idx is None or self.session.tip_idx is None:
            return
        try:
            changed = self.session.recommend_width_endpoints(overwrite=False)
        except Exception:
            changed = False

        if changed:
            self._update_markers_saved()
            self._update_labels_saved()
            self._refresh_point_lists()
            self.plotter.render()

    def on_recommend_width(self):
        """
        ✅ 新按钮：强制重新推荐叶宽点（overwrite=True），并展示：
        - 橙色 W1/W2 点
        - 绿色最短路径（如果可达）
        """
        if not self.annotating:
            return

        # 提交 temp，避免推荐时用到旧状态
        if self.pick_mode != self.MODE_NONE:
            self._exit_current_mode(commit=True)

        if self.session.base_idx is None or self.session.tip_idx is None:
            QMessageBox.information(self, "提示", "请先完成叶基(B1)和叶尖(T1)选择（可选控制点）。")
            return

        try:
            ok = self.session.recommend_width_endpoints(overwrite=True)
            if not ok:
                QMessageBox.information(self, "提示", "未能找到有效的最大叶宽推荐（可调大 radius/slab_half 或检查点云密度）。")
                # 仍刷新一下（有时推荐失败但已有旧点）
                self._update_markers_saved()
                self._update_labels_saved()
                self._refresh_point_lists()
                self.plotter.render()
                return

            # 为了“展示出叶宽的位置”，这里直接计算一次当前 width_path（不 commit）
            self.session.compute()

            self._update_markers_saved()
            self._update_labels_saved()
            self._update_lines()
            self._refresh_point_lists()
            self.plotter.render()

            self._update_status("已推荐叶宽点：W1/W2 已更新（橙色点），并显示最短路径（绿色线）。")
        except Exception as e:
            QMessageBox.critical(self, "推荐失败", str(e))

    def on_recommend_length(self):
        """
        推荐叶长：仅用 B1/T1，在 kNN 图上求最短路径并显示红线。
        """
        if not self.annotating:
            return

        if self.pick_mode != self.MODE_NONE:
            self._exit_current_mode(commit=True)

        if self.session.base_idx is None or self.session.tip_idx is None:
            QMessageBox.information(self, "提示", "请先完成叶基(B1)和叶尖(T1)选择。")
            return

        try:
            self.session.compute_centerline(use_ctrl=False)
            self._update_lines()
            self.plotter.render()
            L = self.session.centerline_result.length if self.session.centerline_result else None
            self._update_status(f"已推荐叶长（B1→T1 最短路径）：叶长={L:.6f}")
        except Exception as e:
            QMessageBox.critical(self, "推荐叶长失败", str(e))

    def on_generate_length(self):
        """
        生成叶长：用 B1/控制点/T1 在 kNN 图上求分段最短路径并显示红线。
        """
        if not self.annotating:
            return

        if self.pick_mode != self.MODE_NONE:
            self._exit_current_mode(commit=True)

        if self.session.base_idx is None or self.session.tip_idx is None:
            QMessageBox.information(self, "提示", "请先完成叶基(B1)和叶尖(T1)选择（可选控制点）。")
            return

        try:
            self.session.compute_centerline(use_ctrl=True)
            self._update_lines()
            self.plotter.render()
            L = self.session.centerline_result.length if self.session.centerline_result else None
            self._update_status(f"已生成叶长（使用控制点）：叶长={L:.6f}")
        except Exception as e:
            QMessageBox.critical(self, "生成叶长失败", str(e))

    # ----------------------------
    # markers + labels
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

        for i, idx in enumerate(self.session.ctrl_indices, start=1):
            pts.append(ds[int(idx)])
            labels.append(f"C{i}")

        if getattr(self.session, "width_w1_idx", None) is not None:
            pts.append(ds[int(self.session.width_w1_idx)])
            labels.append("W1")
        if getattr(self.session, "width_w2_idx", None) is not None:
            pts.append(ds[int(self.session.width_w2_idx)])
            labels.append("W2")

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

        if len(pts) > 0:
            self._add_labels_actor(self.A_LABELS_TEMP, np.asarray(pts), labels)
        else:
            self._remove_actor(self.A_LABELS_TEMP)

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

    # ----------------------------
    # point list UI
    # ----------------------------
    def _refresh_point_lists(self):
        self.list_base.clear()
        self.list_tip.clear()
        self.list_ctrl.clear()
        self.list_width.clear()

        if self.session.base_idx is not None:
            it = QListWidgetItem("B1")
            it.setData(0x0100, ("base", 0))
            self.list_base.addItem(it)

        if self.session.tip_idx is not None:
            it = QListWidgetItem("T1")
            it.setData(0x0100, ("tip", 0))
            self.list_tip.addItem(it)

        for i, _ in enumerate(self.session.ctrl_indices, start=1):
            it = QListWidgetItem(f"C{i}")
            it.setData(0x0100, ("ctrl", i - 1))
            self.list_ctrl.addItem(it)

        if getattr(self.session, "width_w1_idx", None) is not None:
            it = QListWidgetItem("W1")
            it.setData(0x0100, ("w", 1))
            self.list_width.addItem(it)
        if getattr(self.session, "width_w2_idx", None) is not None:
            it = QListWidgetItem("W2")
            it.setData(0x0100, ("w", 2))
            self.list_width.addItem(it)

    def _invalidate_results_after_point_change(self):
        self.session.centerline_result = None
        self.session.width_path_points = None
        if hasattr(self.session, "width_path_length"):
            self.session.width_path_length = None
        self._update_lines()

    # ----------------------------
    # pick state machine
    # ----------------------------
    def _enter_mode(self, mode: str):
        if self.pick_mode != self.MODE_NONE:
            self._exit_current_mode(commit=True)

        self.pick_mode = mode

        if mode == self.MODE_BASE:
            self.temp_base_idx = None
            self.btn_toggle_base.setText("叶基选择：开启")
        elif mode == self.MODE_TIP:
            self.temp_tip_idx = None
            self.btn_toggle_tip.setText("叶尖选择：开启")
        elif mode == self.MODE_CTRL:
            self.temp_ctrl_indices = []
            self.btn_toggle_ctrl.setText("控制点选择：开启")
        elif mode == self.MODE_WIDTH:
            self.temp_w1_idx = None
            self.temp_w2_idx = None
            self.btn_toggle_width.setText("叶宽点选择：开启 (W1/W2)")

        self._update_markers_temp()
        self._update_labels_temp()
        self.plotter.render()

        if mode == self.MODE_WIDTH:
            self._update_status("叶宽点选择：Shift+左键依次选择 W1/W2；关闭后保存。")
        else:
            self._update_status("提示：Shift+左键选点；再次点击按钮关闭并保存。")

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

        if self.pick_mode == self.MODE_BASE:
            self.temp_base_idx = None
        elif self.pick_mode == self.MODE_TIP:
            self.temp_tip_idx = None
        elif self.pick_mode == self.MODE_CTRL:
            self.temp_ctrl_indices = []
        elif self.pick_mode == self.MODE_WIDTH:
            self.temp_w1_idx = None
            self.temp_w2_idx = None

        self.pick_mode = self.MODE_NONE

        for b in [self.btn_toggle_base, self.btn_toggle_tip, self.btn_toggle_ctrl, self.btn_toggle_width]:
            b.blockSignals(True)
            b.setChecked(False)
            b.blockSignals(False)

        self.btn_toggle_base.setText("叶基选择：关闭")
        self.btn_toggle_tip.setText("叶尖选择：关闭")
        self.btn_toggle_ctrl.setText("控制点选择：关闭")
        self.btn_toggle_width.setText("叶宽点选择：关闭 (W1/W2)")

        self._update_markers_saved()
        self._update_markers_temp()
        self._update_labels_saved()
        self._update_labels_temp()
        self._update_lines()
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
                self._update_status("叶基选择关闭：叶基点已保存并保留显示。")

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
                self._update_status("叶尖选择关闭：叶尖点已保存并保留显示。")

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
                self._update_status("控制点选择关闭：控制点已保存并保留显示。")

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
                self._update_status("叶宽点选择关闭：W1/W2 已保存并保留显示。")

    def _close_other_toggles(self, except_mode: str):
        mapping = {
            self.MODE_BASE: self.btn_toggle_base,
            self.MODE_TIP: self.btn_toggle_tip,
            self.MODE_CTRL: self.btn_toggle_ctrl,
            self.MODE_WIDTH: self.btn_toggle_width,
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
        if self.session.ds is None:
            return

        p = np.array(point_xyz, dtype=np.float64)
        idx = self.session.snap_to_ds_index(p)

        if self.pick_mode == self.MODE_BASE:
            self.temp_base_idx = idx
        elif self.pick_mode == self.MODE_TIP:
            self.temp_tip_idx = idx
        elif self.pick_mode == self.MODE_CTRL:
            self.temp_ctrl_indices.append(idx)
        elif self.pick_mode == self.MODE_WIDTH:
            if self.temp_w1_idx is None:
                self.temp_w1_idx = idx
            elif self.temp_w2_idx is None:
                self.temp_w2_idx = idx
            else:
                self.temp_w2_idx = idx

        self._update_markers_temp()
        self._update_labels_temp()
        self.plotter.render()

    # ----------------------------
    # delete actions
    # ----------------------------
    def on_delete_base(self):
        if self.session.base_idx is None:
            return
        self.session.base_idx = None
        self._invalidate_results_after_point_change()
        self._update_markers_saved()
        self._update_labels_saved()
        self.plotter.render()
        self._refresh_point_lists()
        self._update_status("已删除叶基点。")

    def on_delete_tip(self):
        if self.session.tip_idx is None:
            return
        self.session.tip_idx = None
        self._invalidate_results_after_point_change()
        self._update_markers_saved()
        self._update_labels_saved()
        self.plotter.render()
        self._refresh_point_lists()
        self._update_status("已删除叶尖点。")

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

        self._invalidate_results_after_point_change()
        self._update_markers_saved()
        self._update_labels_saved()
        self.plotter.render()
        self._refresh_point_lists()
        self._maybe_recommend_width_and_refresh()
        self._update_status("已删除选中的控制点。")

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
        self._update_status("已删除选中的叶宽点。")

    # ----------------------------
    # view mode changed (browse only)
    # ----------------------------
    def on_view_mode_changed(self):
        if self.session.cloud is None:
            return
        if not self.annotating:
            self._show_browse_scene()

    # ----------------------------
    # UI actions
    # ----------------------------
    def on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择整株点云文件", "",
            "PointCloud (*.xyz *.txt *.csv *.npy *.ply *.pcd *.vtk *.vtp);;All (*.*)"
        )
        if not path:
            return

        try:
            self.session.load(path)
        except Exception as e:
            QMessageBox.critical(self, "读取失败", str(e))
            return

        self.combo_inst.blockSignals(True)
        self.combo_inst.clear()
        ids = self.session.list_instance_ids()
        for _id in ids.tolist():
            self.combo_inst.addItem(str(int(_id)))
        self.combo_inst.blockSignals(False)

        self.annotating = False
        self.pick_mode = self.MODE_NONE
        self.temp_base_idx = None
        self.temp_tip_idx = None
        self.temp_ctrl_indices = []
        self.temp_w1_idx = None
        self.temp_w2_idx = None

        self._update_buttons()
        self._show_browse_scene()
        self._refresh_point_lists()
        self._update_status("浏览模式：可切换 RGB / Semantic / Instance 显示整株。")

    def on_start_annotation(self):
        if self.combo_inst.count() == 0:
            QMessageBox.information(self, "提示", "没有可选实例（inst>=0）。")
            return
        try:
            inst_id = int(self.combo_inst.currentText())
            self.session.select_instance(inst_id)
        except Exception as e:
            QMessageBox.critical(self, "进入标注失败", str(e))
            return

        self.annotating = True

        self.pick_mode = self.MODE_NONE
        self.temp_base_idx = None
        self.temp_tip_idx = None
        self.temp_ctrl_indices = []
        self.temp_w1_idx = None
        self.temp_w2_idx = None

        self.session.centerline_result = None
        self.session.width_path_points = None

        self._update_buttons()
        self._show_annotate_scene()
        self._refresh_point_lists()

        extra = "该实例已标注：已恢复 base/tip/ctrl/W1/W2，并显示缓存中心线/宽线。" if self.session.is_current_annotated() else "该实例未标注。"
        self._update_status(f"已进入标注模式：inst_id={inst_id}\n{extra}\n提示：Shift+左键选点。")

        self._maybe_recommend_width_and_refresh()

    def on_back_browse(self):
        self.annotating = False
        if self.pick_mode != self.MODE_NONE:
            self._exit_current_mode(commit=True)

        self._update_buttons()
        self._show_browse_scene()
        self._refresh_point_lists()
        self._update_status("已返回浏览模式：显示整株点云。")

    def on_inst_changed(self):
        if not self.annotating:
            self._update_buttons()
            return
        if self.combo_inst.count() == 0:
            return
        try:
            if self.pick_mode != self.MODE_NONE:
                self._exit_current_mode(commit=True)

            inst_id = int(self.combo_inst.currentText())
            self.session.select_instance(inst_id)

            self.temp_base_idx = None
            self.temp_tip_idx = None
            self.temp_ctrl_indices = []
            self.temp_w1_idx = None
            self.temp_w2_idx = None

            self.session.centerline_result = None
            self.session.width_path_points = None

            self._update_buttons()
            self._show_annotate_scene()
            self._refresh_point_lists()

            extra = "该实例已标注：已恢复 base/tip/ctrl/W1/W2，并显示缓存中心线/宽线。" if self.session.is_current_annotated() else "该实例未标注。"
            self._update_status(f"切换实例：inst_id={inst_id}\n{extra}\n提示：Shift+左键选点。")

            self._maybe_recommend_width_and_refresh()

        except Exception as e:
            QMessageBox.critical(self, "切换实例失败", str(e))

    def on_clear_ctrl(self):
        self.session.clear_ctrl()
        self.temp_ctrl_indices = []
        self._invalidate_results_after_point_change()
        if self.annotating:
            self._update_markers_saved()
            self._update_markers_temp()
            self._update_labels_saved()
            self._update_labels_temp()
            self.plotter.render()
        self._refresh_point_lists()
        self._maybe_recommend_width_and_refresh()
        self._update_status("已清空控制点（已保存 + 临时）。")

    def on_compute(self):
        if self.pick_mode != self.MODE_NONE:
            self._exit_current_mode(commit=True)

        try:
            self.session.compute()
            self.session.commit_current(False)
        except Exception as e:
            QMessageBox.critical(self, "计算失败", str(e))
            return

        if self.annotating:
            self._update_markers_saved()
            self._update_labels_saved()
            self._update_lines()
            self.plotter.render()

        self._update_buttons()
        self._refresh_point_lists()

        L = self.session.centerline_result.length if self.session.centerline_result else None
        wlen = getattr(self.session, "width_path_length", None)
        if wlen is None:
            self._update_status(f"计算并保存完成：叶长={L:.6f}\n叶宽：未设置 W1/W2 或最短路径不可达。")
        else:
            self._update_status(f"计算并保存完成：叶长={L:.6f} | 叶宽(最短路径长度)={wlen:.6f}")

    def on_export_all(self):
        if self.session.get_annotations_count() == 0:
            QMessageBox.information(self, "提示", "当前文件还没有任何实例被标注。")
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self, "导出整株标注 JSON", "plant_annotations.json", "JSON (*.json)"
        )
        if not out_path:
            return

        try:
            self.session.export_all_json(out_path)
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))
            return

        QMessageBox.information(self, "完成", f"已导出：{out_path}")
        self._update_buttons()
        self._update_status(f"已导出整株标注：{os.path.basename(out_path)}")
