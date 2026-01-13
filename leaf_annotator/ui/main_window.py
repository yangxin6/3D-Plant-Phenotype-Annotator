import os
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from core.annotation import LeafAnnotationSession, AnnotationParams


def make_polyline_mesh(points: np.ndarray) -> pv.PolyData:
    n = len(points)
    poly = pv.PolyData(points)
    if n >= 2:
        lines = np.hstack([[n], np.arange(n)]).astype(np.int64)
        poly.lines = lines
    return poly


class LeafAnnotatorWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Corn Leaf 3D Annotation (PyVista, class-based)")

        self.session = LeafAnnotationSession(params=AnnotationParams())
        self.pick_mode = None  # "base" | "tip" | "ctrl"

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QHBoxLayout(root)

        # 左侧面板
        panel = QtWidgets.QVBoxLayout()
        layout.addLayout(panel, 0)

        self.btn_load = QtWidgets.QPushButton("加载点云")
        self.btn_pick_base = QtWidgets.QPushButton("点选：叶基")
        self.btn_pick_tip = QtWidgets.QPushButton("点选：叶尖")
        self.btn_add_ctrl = QtWidgets.QPushButton("点选：添加控制点(可选)")
        self.btn_clear_ctrl = QtWidgets.QPushButton("清空控制点")
        self.btn_compute = QtWidgets.QPushButton("计算：叶长+叶宽")
        self.btn_export = QtWidgets.QPushButton("导出 JSON")

        panel.addWidget(self.btn_load)
        panel.addWidget(self.btn_pick_base)
        panel.addWidget(self.btn_pick_tip)
        panel.addWidget(self.btn_add_ctrl)
        panel.addWidget(self.btn_clear_ctrl)
        panel.addWidget(self.btn_compute)
        panel.addWidget(self.btn_export)

        panel.addSpacing(10)
        self.info = QtWidgets.QLabel("状态：未加载")
        self.info.setWordWrap(True)
        panel.addWidget(self.info)
        panel.addStretch(1)

        # 右侧：PyVista 视图
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor, 1)
        self.plotter.set_background("white")
        self.plotter.show_axes()

        # connect
        self.btn_load.clicked.connect(self.on_load)
        self.btn_pick_base.clicked.connect(lambda: self.enable_pick("base"))
        self.btn_pick_tip.clicked.connect(lambda: self.enable_pick("tip"))
        self.btn_add_ctrl.clicked.connect(lambda: self.enable_pick("ctrl"))
        self.btn_clear_ctrl.clicked.connect(self.on_clear_ctrl)
        self.btn_compute.clicked.connect(self.on_compute)
        self.btn_export.clicked.connect(self.on_export)

    def set_status(self, text: str):
        self.info.setText(text)

    def redraw_scene(self):
        """重绘：点云 + base/tip/ctrl + centerline + maxwidth"""
        self.plotter.clear()

        ds = self.session.ds_pts
        if ds is None:
            self.plotter.render()
            return

        self.plotter.add_points(pv.PolyData(ds), render_points_as_spheres=True, point_size=4, color="gray")

        # picked points
        if self.session.base_idx is not None:
            self.plotter.add_points(pv.PolyData([ds[self.session.base_idx]]), color="blue",
                                    render_points_as_spheres=True, point_size=14, name="base")
        if self.session.tip_idx is not None:
            self.plotter.add_points(pv.PolyData([ds[self.session.tip_idx]]), color="orange",
                                    render_points_as_spheres=True, point_size=14, name="tip")
        for k, idx in enumerate(self.session.ctrl_indices, start=1):
            self.plotter.add_points(pv.PolyData([ds[idx]]), color="purple",
                                    render_points_as_spheres=True, point_size=12, name=f"ctrl_{k}")

        # results
        if self.session.centerline_result is not None:
            cl = self.session.centerline_result.smooth_points
            self.plotter.add_mesh(make_polyline_mesh(cl), color="red", line_width=4, name="centerline")

        if self.session.width_result is not None and self.session.width_result.max_item is not None:
            it = self.session.width_result.max_item
            seg = np.vstack([it.pL, it.pR])
            self.plotter.add_mesh(make_polyline_mesh(seg), color="green", line_width=6, name="maxwidth")

        self.plotter.reset_camera()
        self.plotter.render()

    def on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择单叶点云文件", "",
            "PointCloud (*.ply *.pcd *.xyz *.txt *.csv *.vtk *.vtp);;All (*.*)"
        )
        if not path:
            return

        try:
            self.session.load(path)
        except Exception as e:
            QMessageBox.critical(self, "读取失败", str(e))
            return

        self.redraw_scene()
        self.set_status(
            f"已加载：{os.path.basename(path)}\n"
            f"原始点数：{len(self.session.leaf_pts)} | 下采样点数：{len(self.session.ds_pts)}\n"
            f"下一步：点选叶基、叶尖（Shift+左键）。"
        )

    def enable_pick(self, mode: str):
        if self.session.ds_pts is None:
            QMessageBox.information(self, "提示", "请先加载点云。")
            return

        self.pick_mode = mode
        self.set_status(self.info.text() + f"\n点选模式：{mode}（Shift+左键选点）")

        try:
            self.plotter.disable_picking()
        except Exception:
            pass

        self.plotter.enable_point_picking(
            callback=self.on_picked_point,
            show_message=True,
            color="red",
            point_size=12,
            use_mesh=True
        )

    def on_picked_point(self, point):
        if point is None:
            return
        p = np.array(point, dtype=np.float64)
        idx = self.session.snap_to_ds_index(p)

        if self.pick_mode == "base":
            self.session.set_base(idx)
        elif self.pick_mode == "tip":
            self.session.set_tip(idx)
        elif self.pick_mode == "ctrl":
            self.session.add_ctrl(idx)

        self.redraw_scene()
        self.set_status(f"已选：base={self.session.base_idx} tip={self.session.tip_idx} 控制点数={len(self.session.ctrl_indices)}")

    def on_clear_ctrl(self):
        self.session.clear_ctrl()
        self.redraw_scene()
        self.set_status("已清空控制点。")

    def on_compute(self):
        try:
            self.session.compute()
        except Exception as e:
            QMessageBox.critical(self, "计算失败", str(e))
            return

        self.redraw_scene()

        L = self.session.centerline_result.length if self.session.centerline_result else None
        if self.session.width_result is None or self.session.width_result.max_item is None:
            self.set_status(
                f"完成：叶长={L:.6f}\n未找到有效宽度：可调大 slab_half/radius 或减小 step。"
            )
        else:
            wmax = self.session.width_result.max_item.width
            n = len(self.session.width_result.profile)
            self.set_status(f"完成：叶长={L:.6f} | 最大叶宽={wmax:.6f} | 宽度采样点={n}")

    def on_export(self):
        if self.session.centerline_result is None:
            QMessageBox.information(self, "提示", "请先计算叶长/叶宽。")
            return

        out_path, _ = QFileDialog.getSaveFileName(self, "保存标注 JSON", "leaf_annotation.json", "JSON (*.json)")
        if not out_path:
            return

        try:
            self.session.export_json(out_path)
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))
            return

        QMessageBox.information(self, "完成", f"已保存：{out_path}")
