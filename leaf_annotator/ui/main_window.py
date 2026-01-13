# ui/main_window.py
import os
import numpy as np
import pyvista as pv
import vtk
from pyvistaqt import QtInteractor

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox

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
    ✅ 你要求的逻辑（保证）：
    1) 开启叶基 -> 选点 -> 关闭：点保留（保存到 session.base_idx，始终渲染）
    2) 开启叶尖 -> 选点 -> 关闭：叶基仍在，叶尖保留（session.tip_idx）
    3) 开启控制点 -> 选多个 -> 关闭：叶基/叶尖仍在，控制点多个保留（session.ctrl_indices）

    额外保证：
    - 标注模式共用一个视图：不 plotter.clear()，不 reset_camera()（只增量更新 actors）
    - 标注模式实例点云统一灰色
    - base 红、tip 蓝、ctrl 绿，点选的点（保存/临时）都可见
    - 切换 inst：若已标注，自动显示缓存中心线/宽线；并恢复保存的 base/tip/ctrl（由 session 负责）
    - 稳定拾取：使用 VTK vtkPointPicker + Shift+左键（不依赖 PyVista 版本的 enable_point_picking）
    """

    MODE_NONE = "NONE"
    MODE_BASE = "PICK_BASE"
    MODE_TIP = "PICK_TIP"
    MODE_CTRL = "PICK_CTRL"

    # actor names (增量更新用)
    A_CLOUD_FULL = "cloud_full"
    A_CLOUD_INST = "cloud_inst"

    A_BASE_SAVED = "mark_base_saved"
    A_TIP_SAVED = "mark_tip_saved"
    A_CTRL_SAVED = "mark_ctrl_saved"

    A_BASE_TEMP = "mark_base_temp"
    A_TIP_TEMP = "mark_tip_temp"
    A_CTRL_TEMP = "mark_ctrl_temp"

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

        # VTK picker
        self._vtk_picker = vtk.vtkPointPicker()
        self._vtk_picker.SetTolerance(0.02)  # 可调：越大越容易拾取到点
        self._pick_observer_id = None

        # 记录当前实例点云 actor，便于 PickFromList（防止拾取到线/marker）
        self._actor_cloud_inst = None

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QHBoxLayout(root)

        # ---------- left panel ----------
        panel = QtWidgets.QVBoxLayout()
        layout.addLayout(panel, 0)

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
        for b in [self.btn_toggle_base, self.btn_toggle_tip, self.btn_toggle_ctrl]:
            b.setCheckable(True)
        panel.addWidget(self.btn_toggle_base)
        panel.addWidget(self.btn_toggle_tip)
        panel.addWidget(self.btn_toggle_ctrl)

        self.btn_clear_ctrl = QtWidgets.QPushButton("清空控制点（已保存）")
        self.btn_compute = QtWidgets.QPushButton("计算并保存（叶长+叶宽）")
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

        self.btn_clear_ctrl.clicked.connect(self.on_clear_ctrl)
        self.btn_compute.clicked.connect(self.on_compute)
        self.btn_export.clicked.connect(self.on_export_all)

        # 安装一次：Shift+左键拾取（稳定，不依赖 pyvista picking）
        self._install_vtk_pick_observer()

        self._update_buttons()
        self._update_status("提示：标注模式下用 Shift+左键 选点。")

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
        for b in [self.btn_toggle_base, self.btn_toggle_tip, self.btn_toggle_ctrl,
                  self.btn_clear_ctrl, self.btn_compute]:
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

        iren = self.plotter.interactor  # vtkRenderWindowInteractor

        def _on_left_press(obj, event):
            # 仅标注模式 + 当前处于某个 pick 模式才拾取
            if (not self.annotating) or (self.pick_mode == self.MODE_NONE) or (self.session.ds is None):
                return

            # 只在 Shift 按下时拾取，避免影响平时旋转
            if iren.GetShiftKey() == 0:
                return

            x, y = iren.GetEventPosition()
            renderer = self.plotter.renderer

            # 只从实例点云 actor 拾取（避免拾取到线/marker）
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

            # 阻止相机旋转（仅在Shift拾取时）
            obj.AbortFlagOn()

        self._pick_observer_id = iren.AddObserver("LeftButtonPressEvent", _on_left_press)

    # ----------------------------
    # actor helpers (增量更新)
    # ----------------------------
    def _remove_actor(self, name: str):
        try:
            self.plotter.remove_actor(name)
        except Exception:
            # 某些版本 remove_actor 可能不支持 name，忽略即可
            pass

    def _add_points_actor(self, name: str, pts: np.ndarray, color, point_size: int):
        """一次性添加点 actor（pts: Nx3）。"""
        self._remove_actor(name)
        if pts is None or len(pts) == 0:
            return None
        poly = pv.PolyData(pts)
        actor = self.plotter.add_mesh(
            poly,
            name=name,
            color=color,
            render_points_as_spheres=True,
            point_size=point_size
        )
        return actor

    def _add_polyline_actor(self, name: str, pts: np.ndarray, color, line_width: int):
        self._remove_actor(name)
        if pts is None or len(pts) < 2:
            return None
        actor = self.plotter.add_mesh(
            make_polyline_mesh(np.asarray(pts, dtype=np.float64)),
            name=name,
            color=color,
            line_width=line_width
        )
        return actor

    # ----------------------------
    # build polydata for browsing
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
    # scene update (单视图增量更新)
    # ----------------------------
    def _show_browse_scene(self):
        """浏览模式：显示整株，隐藏标注层。"""
        # cloud_full
        poly = self._get_full_cloud_polydata()
        self._remove_actor(self.A_CLOUD_INST)
        self._actor_cloud_inst = None

        # 重新加 full cloud
        self._remove_actor(self.A_CLOUD_FULL)
        self.plotter.add_points(poly, scalars="rgb", rgb=True,
                                name=self.A_CLOUD_FULL,
                                render_points_as_spheres=True, point_size=3)

        # 移除所有标注层
        for nm in [
            self.A_BASE_SAVED, self.A_TIP_SAVED, self.A_CTRL_SAVED,
            self.A_BASE_TEMP, self.A_TIP_TEMP, self.A_CTRL_TEMP,
            self.A_LINE_CENTER_CUR, self.A_LINE_WIDTH_CUR,
            self.A_LINE_CENTER_CACHED, self.A_LINE_WIDTH_CACHED
        ]:
            self._remove_actor(nm)

        self.plotter.render()

    def _show_annotate_scene(self):
        """标注模式：显示当前实例（灰），叠加 markers/lines。"""
        if self.session.ds is None:
            return

        ds = self.session.get_ds_points()

        # 替换 cloud_inst
        self._remove_actor(self.A_CLOUD_FULL)
        self._remove_actor(self.A_CLOUD_INST)
        poly = pv.PolyData(ds)

        # add_mesh 返回 actor，保存以便 picker 只拾取它
        self._actor_cloud_inst = self.plotter.add_mesh(
            poly,
            name=self.A_CLOUD_INST,
            color=(0.7, 0.7, 0.7),
            render_points_as_spheres=True,
            point_size=5
        )

        # markers: saved + temp
        self._update_markers_saved()
        self._update_markers_temp()

        # lines
        self._update_lines()

        self.plotter.render()

    def _update_markers_saved(self):
        """保存点：始终显示（base红 tip蓝 ctrl绿）。"""
        if self.session.ds is None:
            return
        ds = self.session.get_ds_points()

        # base saved (single)
        if self.session.base_idx is not None:
            p = ds[int(self.session.base_idx)][None, :]
            self._add_points_actor(self.A_BASE_SAVED, p, "red", 16)
        else:
            self._remove_actor(self.A_BASE_SAVED)

        # tip saved (single)
        if self.session.tip_idx is not None:
            p = ds[int(self.session.tip_idx)][None, :]
            self._add_points_actor(self.A_TIP_SAVED, p, "blue", 16)
        else:
            self._remove_actor(self.A_TIP_SAVED)

        # ctrl saved (many) — 聚合成一个 actor
        if len(self.session.ctrl_indices) > 0:
            pts = ds[np.array(self.session.ctrl_indices, dtype=int)]
            self._add_points_actor(self.A_CTRL_SAVED, pts, "green", 14)
        else:
            self._remove_actor(self.A_CTRL_SAVED)

    def _update_markers_temp(self):
        """临时点：仅用于预览；关闭后清空（base红 tip蓝 ctrl绿）。"""
        if self.session.ds is None:
            return
        ds = self.session.get_ds_points()

        # base temp
        if self.temp_base_idx is not None:
            p = ds[int(self.temp_base_idx)][None, :]
            self._add_points_actor(self.A_BASE_TEMP, p, "red", 18)
        else:
            self._remove_actor(self.A_BASE_TEMP)

        # tip temp
        if self.temp_tip_idx is not None:
            p = ds[int(self.temp_tip_idx)][None, :]
            self._add_points_actor(self.A_TIP_TEMP, p, "blue", 18)
        else:
            self._remove_actor(self.A_TIP_TEMP)

        # ctrl temp
        if len(self.temp_ctrl_indices) > 0:
            pts = ds[np.array(self.temp_ctrl_indices, dtype=int)]
            self._add_points_actor(self.A_CTRL_TEMP, pts, "green", 16)
        else:
            self._remove_actor(self.A_CTRL_TEMP)

    def _update_lines(self):
        """线展示：当前结果优先，否则显示缓存。"""
        # current results
        if self.session.centerline_result is not None:
            cl = np.asarray(self.session.centerline_result.smooth_points, dtype=np.float64)
            self._add_polyline_actor(self.A_LINE_CENTER_CUR, cl, "red", 4)
            self._remove_actor(self.A_LINE_CENTER_CACHED)
        else:
            self._remove_actor(self.A_LINE_CENTER_CUR)

        if self.session.width_result is not None and self.session.width_result.max_item is not None:
            it = self.session.width_result.max_item
            seg = np.vstack([it.pL, it.pR])
            self._add_polyline_actor(self.A_LINE_WIDTH_CUR, seg, "green", 6)
            self._remove_actor(self.A_LINE_WIDTH_CACHED)
        else:
            self._remove_actor(self.A_LINE_WIDTH_CUR)

        # cached results (only if no current)
        if (self.session.centerline_result is None) and (self.session.current_inst_id is not None):
            cl, seg = self.session.get_cached_display(self.session.current_inst_id)
            if cl is not None and len(cl) >= 2:
                self._add_polyline_actor(self.A_LINE_CENTER_CACHED, np.asarray(cl, dtype=np.float64), "red", 4)
            else:
                self._remove_actor(self.A_LINE_CENTER_CACHED)

            if seg is not None:
                pL, pR = seg
                self._add_polyline_actor(self.A_LINE_WIDTH_CACHED, np.vstack([pL, pR]), "green", 6)
            else:
                self._remove_actor(self.A_LINE_WIDTH_CACHED)
        else:
            # 如果有 current，就隐藏 cached
            self._remove_actor(self.A_LINE_CENTER_CACHED)
            self._remove_actor(self.A_LINE_WIDTH_CACHED)

    # ----------------------------
    # view mode changed (browse only)
    # ----------------------------
    def on_view_mode_changed(self):
        if self.session.cloud is None:
            return
        if not self.annotating:
            self._show_browse_scene()

    # ----------------------------
    # pick state machine (开启/关闭 & commit)
    # ----------------------------
    def _enter_mode(self, mode: str):
        """进入某模式：先退出当前模式并提交当前模式 temp，再进入新模式并清该模式 temp。"""
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

        # 标注模式：不清保存点，只更新 temp layer
        self._update_markers_temp()
        self.plotter.render()
        self._update_status("提示：按住 Shift + 左键 选点；再次点击按钮关闭并保存。")

    def _exit_current_mode(self, commit: bool = True):
        """退出当前模式：可选提交 temp -> session，并清 temp；关闭按钮状态；更新渲染。"""
        if commit:
            if self.pick_mode == self.MODE_BASE and self.temp_base_idx is not None:
                self.session.set_base(self.temp_base_idx)
            if self.pick_mode == self.MODE_TIP and self.temp_tip_idx is not None:
                self.session.set_tip(self.temp_tip_idx)
            if self.pick_mode == self.MODE_CTRL and len(self.temp_ctrl_indices) > 0:
                self.session.extend_ctrl(self.temp_ctrl_indices)

        # 清 temp（只清当前模式对应 temp）
        if self.pick_mode == self.MODE_BASE:
            self.temp_base_idx = None
        elif self.pick_mode == self.MODE_TIP:
            self.temp_tip_idx = None
        elif self.pick_mode == self.MODE_CTRL:
            self.temp_ctrl_indices = []

        self.pick_mode = self.MODE_NONE

        # 关闭 toggle（避免递归）
        for b in [self.btn_toggle_base, self.btn_toggle_tip, self.btn_toggle_ctrl]:
            b.blockSignals(True)
            b.setChecked(False)
            b.blockSignals(False)

        self.btn_toggle_base.setText("叶基选择：关闭")
        self.btn_toggle_tip.setText("叶尖选择：关闭")
        self.btn_toggle_ctrl.setText("控制点选择：关闭")

        # 更新保存点（关闭后仍展示）+ 清 temp layer
        self._update_markers_saved()
        self._update_markers_temp()
        self.plotter.render()

    # ----------------------------
    # toggle slots
    # ----------------------------
    def on_toggle_base(self, checked: bool):
        if not self.annotating:
            return
        if checked:
            # 互斥关闭其它
            self.btn_toggle_tip.blockSignals(True);  self.btn_toggle_tip.setChecked(False);  self.btn_toggle_tip.blockSignals(False)
            self.btn_toggle_ctrl.blockSignals(True); self.btn_toggle_ctrl.setChecked(False); self.btn_toggle_ctrl.blockSignals(False)
            self._enter_mode(self.MODE_BASE)
        else:
            # 关闭：提交 base temp，保证红点保留
            if self.pick_mode == self.MODE_BASE:
                self._exit_current_mode(commit=True)
                self._update_status("叶基选择关闭：叶基点已保存并保留显示。")

    def on_toggle_tip(self, checked: bool):
        if not self.annotating:
            return
        if checked:
            self.btn_toggle_base.blockSignals(True); self.btn_toggle_base.setChecked(False); self.btn_toggle_base.blockSignals(False)
            self.btn_toggle_ctrl.blockSignals(True); self.btn_toggle_ctrl.setChecked(False); self.btn_toggle_ctrl.blockSignals(False)
            self._enter_mode(self.MODE_TIP)
        else:
            if self.pick_mode == self.MODE_TIP:
                self._exit_current_mode(commit=True)
                self._update_status("叶尖选择关闭：叶尖点已保存并保留显示。")

    def on_toggle_ctrl(self, checked: bool):
        if not self.annotating:
            return
        if checked:
            self.btn_toggle_base.blockSignals(True); self.btn_toggle_base.setChecked(False); self.btn_toggle_base.blockSignals(False)
            self.btn_toggle_tip.blockSignals(True);  self.btn_toggle_tip.setChecked(False);  self.btn_toggle_tip.blockSignals(False)
            self._enter_mode(self.MODE_CTRL)
        else:
            if self.pick_mode == self.MODE_CTRL:
                self._exit_current_mode(commit=True)
                self._update_status("控制点选择关闭：控制点已保存并保留显示。")

    # ----------------------------
    # pick callback (由 VTK picker 调用)
    # ----------------------------
    def on_picked_point(self, point_xyz):
        """point_xyz: 3D 坐标（世界坐标）。"""
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

        # 只更新 temp layer（保存点仍在）
        self._update_markers_temp()
        self.plotter.render()

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

        self._update_buttons()
        self._show_browse_scene()
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
        self._exit_current_mode(commit=False)  # 关闭所有toggle + 清temp（不提交）
        self._update_buttons()

        # 切 inst 后 session 可能恢复保存点（已标注时）
        self._show_annotate_scene()
        extra = "该实例已标注：已恢复 base/tip/ctrl，并显示缓存中心线/宽线。" if self.session.is_current_annotated() else "该实例未标注。"
        self._update_status(f"已进入标注模式：inst_id={inst_id}\n{extra}\n提示：Shift+左键选点。")

    def on_back_browse(self):
        self.annotating = False
        self._exit_current_mode(commit=True)  # 可选：退出前提交临时点
        self._update_buttons()
        self._show_browse_scene()
        self._update_status("已返回浏览模式：显示整株点云。")

    def on_inst_changed(self):
        if not self.annotating:
            self._update_buttons()
            return
        if self.combo_inst.count() == 0:
            return

        try:
            # 切换前：退出当前模式并提交 temp（避免丢）
            self._exit_current_mode(commit=True)

            inst_id = int(self.combo_inst.currentText())
            self.session.select_instance(inst_id)

            # 切换后清 temp
            self.temp_base_idx = None
            self.temp_tip_idx = None
            self.temp_ctrl_indices = []

            # 清当前计算结果显示（如果你希望切 inst 后不带着上一个 current）
            self.session.centerline_result = None
            self.session.width_result = None

            self._update_buttons()
            self._show_annotate_scene()

            extra = "该实例已标注：已恢复 base/tip/ctrl，并显示缓存中心线/宽线。" if self.session.is_current_annotated() else "该实例未标注。"
            self._update_status(f"切换实例：inst_id={inst_id}\n{extra}\n提示：Shift+左键选点。")
        except Exception as e:
            QMessageBox.critical(self, "切换实例失败", str(e))

    def on_clear_ctrl(self):
        # 清已保存控制点 + 清临时控制点
        self.session.clear_ctrl()
        self.temp_ctrl_indices = []
        if self.annotating:
            self._update_markers_saved()
            self._update_markers_temp()
            self.plotter.render()
        self._update_status("已清空控制点（已保存 + 临时）。")

    def on_compute(self):
        # 计算前：退出点选模式并提交 temp，确保 base/tip/ctrl 都已保存
        self._exit_current_mode(commit=True)

        try:
            self.session.compute()
            self.session.commit_current(include_width_profile=False)
        except Exception as e:
            QMessageBox.critical(self, "计算失败", str(e))
            return

        # 更新线层（current 优先）
        if self.annotating:
            self._update_lines()
            self.plotter.render()

        self._update_buttons()

        L = self.session.centerline_result.length if self.session.centerline_result else None
        if self.session.width_result is None or self.session.width_result.max_item is None:
            self._update_status(
                f"计算并保存完成：叶长={L:.6f}\n未找到有效宽度：可调大 slab_half/radius 或减小 step。"
            )
        else:
            wmax = self.session.width_result.max_item.width
            self._update_status(f"计算并保存完成：叶长={L:.6f} | 最大叶宽={wmax:.6f}")

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
