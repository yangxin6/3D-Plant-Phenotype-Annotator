# ui/main_window_parts/annotation.py
from typing import Optional, Tuple

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox, QListWidgetItem


class AnnotationMixin:
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

        if not self.on_width_params_dialog():
            return

        # 提交 temp，避免推荐时用到旧状态
        if self.pick_mode != self.MODE_NONE:
            self._exit_current_mode(commit=True)

        if self.session.base_idx is None or self.session.tip_idx is None:
            QMessageBox.information(self, "提示", "请先完成叶基(B1)和叶尖(T1)选择（可选控制点）。")
            return

        dlg = None
        try:
            dlg = self._start_busy_dialog("正在推荐叶宽...")
            ok = self.session.recommend_width_endpoints(overwrite=True)
            if not ok:
                self._finish_busy_dialog(dlg)
                dlg = None
                QMessageBox.information(self, "提示", "未能找到有效的最大叶宽推荐（可调大 radius/slab_half 或检查点云密度）。")
                # 仍刷新一下（有时推荐失败但已有旧点）
                self._update_markers_saved()
                self._update_labels_saved()
                self._refresh_point_lists()
                self.plotter.render()
                return

            # 为了“展示出叶宽的位置”，这里直接计算一次当前 width_path（不 commit）
            self.session.compute()
            path_idx = getattr(self.session, "width_path_indices", None)
            if path_idx is not None and len(path_idx) > 0:
                w1 = getattr(self.session, "width_w1_idx", None)
                w2 = getattr(self.session, "width_w2_idx", None)
                ctrl_idx = []
                seen = set()
                for idx in path_idx:
                    i = int(idx)
                    if i == w1 or i == w2:
                        continue
                    if i in seen:
                        continue
                    seen.add(i)
                    ctrl_idx.append(i)
                self.session.width_ctrl_indices = ctrl_idx
                self.session.width_ctrl_ids = [10 * (i + 1) for i in range(len(ctrl_idx))]
                self.session._next_width_ctrl_id = 10 * (len(ctrl_idx) + 1) if len(ctrl_idx) > 0 else 10

            self._update_markers_saved()
            self._update_labels_saved()
            self._update_lines()
            self._refresh_point_lists()
            self.plotter.render()

            self._update_status("已推荐叶宽点：W1/W2 已更新（橙色点），并显示最短路径（绿色线）。")
        except Exception as e:
            QMessageBox.critical(self, "推荐失败", str(e))
        finally:
            self._finish_busy_dialog(dlg)


    def on_width_params_dialog(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("叶宽推荐参数")
        layout = QtWidgets.QVBoxLayout(dlg)

        form = QtWidgets.QFormLayout()
        spin_step = QtWidgets.QDoubleSpinBox()
        spin_step.setRange(0.0001, 1.0)
        spin_step.setDecimals(6)
        spin_step.setSingleStep(0.001)
        spin_step.setValue(float(self.session.params.step))

        spin_slab = QtWidgets.QDoubleSpinBox()
        spin_slab.setRange(0.0001, 1.0)
        spin_slab.setDecimals(6)
        spin_slab.setSingleStep(0.001)
        spin_slab.setValue(float(self.session.params.slab_half))

        spin_radius = QtWidgets.QDoubleSpinBox()
        spin_radius.setRange(0.0001, 1.0)
        spin_radius.setDecimals(6)
        spin_radius.setSingleStep(0.001)
        spin_radius.setValue(float(self.session.params.radius))

        spin_min_pts = QtWidgets.QSpinBox()
        spin_min_pts.setRange(1, 100000)
        spin_min_pts.setValue(int(self.session.params.min_slice_pts))

        form.addRow("步长 step：", spin_step)
        form.addRow("薄片半厚 slab_half：", spin_slab)
        form.addRow("半径 radius：", spin_radius)
        form.addRow("最小截面点数 min_slice_pts：", spin_min_pts)
        layout.addLayout(form)

        tips = QtWidgets.QLabel(
            "说明：step 越小越细致但更慢；slab_half/radius 越大越稳定但可能跨到邻近结构；"
            "min_slice_pts 太小易受噪声影响，太大可能找不到截面。"
        )
        tips.setWordWrap(True)
        layout.addWidget(tips)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return False

        self.session.params.step = float(spin_step.value())
        self.session.params.slab_half = float(spin_slab.value())
        self.session.params.radius = float(spin_radius.value())
        self.session.params.min_slice_pts = int(spin_min_pts.value())
        self._update_status("已更新叶宽推荐参数。")
        return True


    def on_generate_width(self):
        """
        生成叶宽：基于 W1/W2（可选控制点）生成最短路径并显示绿线。
        """
        if not self.annotating:
            return

        if self.pick_mode != self.MODE_NONE:
            self._exit_current_mode(commit=True)

        if getattr(self.session, "width_w1_idx", None) is None or getattr(self.session, "width_w2_idx", None) is None:
            QMessageBox.information(self, "提示", "请先选择叶宽端点 W1/W2。")
            return

        dlg = None
        try:
            dlg = self._start_busy_dialog("正在生成叶宽...")
            self.session.compute_width_path()
            path_idx = getattr(self.session, "width_path_indices", None)
            if path_idx is not None and len(path_idx) > 0:
                w1 = getattr(self.session, "width_w1_idx", None)
                w2 = getattr(self.session, "width_w2_idx", None)
                ctrl_idx = []
                seen = set()
                for idx in path_idx:
                    i = int(idx)
                    if i == w1 or i == w2:
                        continue
                    if i in seen:
                        continue
                    seen.add(i)
                    ctrl_idx.append(i)
                self.session.width_ctrl_indices = ctrl_idx
                self.session.width_ctrl_ids = [10 * (i + 1) for i in range(len(ctrl_idx))]
                self.session._next_width_ctrl_id = 10 * (len(ctrl_idx) + 1) if len(ctrl_idx) > 0 else 10
            self._update_lines()
            self.plotter.render()
            self._update_markers_saved()
            self._update_labels_saved()
            self._refresh_point_lists()
            wlen = getattr(self.session, "width_path_length", None)
            if wlen is None:
                self._update_status("叶宽路径生成完成。")
            else:
                self._update_status(f"叶宽路径生成完成：长度={wlen:.6f}")
        except Exception as e:
            QMessageBox.critical(self, "生成叶宽失败", str(e))
        finally:
            self._finish_busy_dialog(dlg)


    def on_compute_leaf_area(self):
        if (not self.annotating) or self.annotate_semantic != "leaf":
            QMessageBox.information(self, "提示", "请在叶子标注模式下计算叶面积。")
            return
        dlg = None
        try:
            dlg = self._start_busy_dialog("正在计算叶面积...")
            area = self.session.compute_leaf_area_instance()
            if area is None:
                QMessageBox.information(self, "提示", "当前无法计算叶面积，请先生成叶长并检查点云密度。")
                return
            self._update_phenotype_table()
            self._update_status(f"已计算叶面积：{area:.3f}")
        except Exception as e:
            QMessageBox.critical(self, "计算失败", str(e))
        finally:
            self._finish_busy_dialog(dlg)


    def on_compute_leaf_projected_area(self):
        if (not self.annotating) or self.annotate_semantic != "leaf":
            QMessageBox.information(self, "提示", "请在叶子标注模式下计算投影面积。")
            return
        dlg = None
        try:
            dlg = self._start_busy_dialog("正在计算投影面积...")
            area = self.session.compute_leaf_projected_area_instance()
            if area is None:
                QMessageBox.information(self, "提示", "当前无法计算投影面积，请检查点云数据。")
                return
            self._update_phenotype_table()
            self._update_status(f"已计算投影面积：{area:.3f}")
        except Exception as e:
            QMessageBox.critical(self, "计算失败", str(e))
        finally:
            self._finish_busy_dialog(dlg)


    def _open_leaf_inclination_params_dialog(self) -> Optional[Tuple[float, float]]:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("叶倾角参数")
        layout = QtWidgets.QVBoxLayout(dlg)

        form = QtWidgets.QFormLayout()
        spin_ratio = QtWidgets.QDoubleSpinBox()
        spin_ratio.setRange(0.0, 1.0)
        spin_ratio.setDecimals(3)
        spin_ratio.setSingleStep(0.05)
        spin_ratio.setValue(float(getattr(self.session.params, "leaf_inclination_ratio", 0.5)))

        spin_radius = QtWidgets.QDoubleSpinBox()
        spin_radius.setRange(0.0, 10.0)
        spin_radius.setDecimals(4)
        spin_radius.setSingleStep(0.001)
        default_radius = getattr(self.session.params, "leaf_inclination_radius", None)
        if default_radius is None:
            default_radius = max(
                float(getattr(self.session.params, "radius", 0.0)),
                float(getattr(self.session.params, "slab_half", 0.0)),
                float(getattr(self.session.params, "voxel", 0.0)),
            )
        spin_radius.setValue(float(default_radius))

        form.addRow("位置比例(0-1)：", spin_ratio)
        form.addRow("局部半径(0=整叶)：", spin_radius)
        layout.addLayout(form)

        tips = QtWidgets.QLabel("说明：位置比例沿叶长从基部到尖端；半径用于局部平面拟合。")
        tips.setWordWrap(True)
        layout.addWidget(tips)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return None

        return float(spin_ratio.value()), float(spin_radius.value())

    def _open_leaf_stem_angle_params_dialog(self) -> Optional[float]:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("叶夹角参数")
        layout = QtWidgets.QVBoxLayout(dlg)

        form = QtWidgets.QFormLayout()
        spin_ratio = QtWidgets.QDoubleSpinBox()
        spin_ratio.setRange(0.0, 1.0)
        spin_ratio.setDecimals(3)
        spin_ratio.setSingleStep(0.05)
        spin_ratio.setValue(float(getattr(self.session.params, "leaf_stem_ratio", 0.1)))

        form.addRow("起始比例(0-1)：", spin_ratio)
        layout.addLayout(form)

        tips = QtWidgets.QLabel("说明：起始比例用于定义叶长起始段方向。")
        tips.setWordWrap(True)
        layout.addWidget(tips)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return None

        return float(spin_ratio.value())

    def on_compute_leaf_inclination(self):
        if (not self.annotating) or self.annotate_semantic != "leaf":
            QMessageBox.information(self, "提示", "请在叶子标注模式下计算叶倾角。")
            return
        try:
            params = self._open_leaf_inclination_params_dialog()
            if params is None:
                return
            ratio, radius = params
            self.session.params.leaf_inclination_ratio = float(ratio)
            self.session.params.leaf_inclination_radius = float(radius)
            angle = self.session.compute_leaf_inclination_instance(ratio=ratio, radius=radius)
            if angle is None:
                QMessageBox.information(self, "提示", "当前无法计算叶倾角，请先生成叶长并检查点云。")
                return
            self._update_phenotype_table()
            self._update_status(f"已计算叶倾角：{angle:.1f}")
            self._update_lines()
            self.plotter.render()
        except Exception as e:
            QMessageBox.critical(self, "计算失败", str(e))

    def on_compute_leaf_stem_angle(self):
        if (not self.annotating) or self.annotate_semantic != "leaf":
            QMessageBox.information(self, "提示", "请在叶子标注模式下计算叶夹角。")
            return
        try:
            ratio = self._open_leaf_stem_angle_params_dialog()
            if ratio is None:
                return
            self.session.params.leaf_stem_ratio = float(ratio)
            angle = self.session.compute_leaf_stem_angle_instance(ratio=ratio)
            if angle is None:
                QMessageBox.information(self, "提示", "当前无法计算叶夹角，请先计算茎长并确保叶长有效。")
                return
            self._update_phenotype_table()
            self._update_status(f"已计算叶夹角：{angle:.1f}")
            self._update_lines()
            self.plotter.render()
        except Exception as e:
            QMessageBox.critical(self, "计算失败", str(e))

    def _open_smooth_params_dialog(self) -> Optional[int]:
        value = int(getattr(self.session.params, "smooth_win", 9))
        win, ok = QtWidgets.QInputDialog.getInt(
            self,
            "平滑曲线",
            "平滑窗口(奇数, >=3)：",
            value=value,
            min=3,
            max=9999,
            step=2
        )
        if not ok:
            return None
        return max(3, int(win) | 1)


    def on_smooth_leaf_paths(self):
        if (not self.annotating) or self.annotate_semantic != "leaf":
            QMessageBox.information(self, "提示", "请在叶子标注模式下使用平滑曲线功能。")
            return
        win = self._open_smooth_params_dialog()
        if win is None:
            return
        self.session.params.smooth_win = int(win)
        updated_len, updated_wid = self.session.smooth_leaf_paths(win)
        if not updated_len and not updated_wid:
            QMessageBox.information(self, "提示", "当前没有可平滑的叶长/叶宽路径，请先生成叶长和叶宽。")
            return
        if self.combo_view.currentText() == "表型标签":
            self._refresh_scene(mode="表型标签")
        else:
            self._update_lines()
            self.plotter.render()
        parts = []
        if updated_len:
            parts.append("叶长")
        if updated_wid:
            parts.append("叶宽")
        msg = f"已平滑：{'、'.join(parts)}，窗口={win}"
        self._update_status(msg)


    def on_recommend_length(self):
        """
        推荐叶长：仅用 B1/T1，在半径图上求最短路径并显示红线。
        """
        if not self.annotating:
            return

        if not self._open_length_params_dialog():
            return

        if self.pick_mode != self.MODE_NONE:
            self._exit_current_mode(commit=True)

        if self.session.base_idx is None or self.session.tip_idx is None:
            QMessageBox.information(self, "提示", "请先完成叶基(B1)和叶尖(T1)选择。")
            return

        dlg = None
        try:
            dlg = self._start_busy_dialog("正在推荐叶长...")
            result = self.session.compute_centerline(use_ctrl=False)
            path_idx = getattr(result, "path_indices", None)
            if path_idx is not None and len(path_idx) > 0:
                base = self.session.base_idx
                tip = self.session.tip_idx
                ctrl_idx = []
                seen = set()
                for idx in path_idx:
                    i = int(idx)
                    if i == base or i == tip:
                        continue
                    if i in seen:
                        continue
                    seen.add(i)
                    ctrl_idx.append(i)
                self.session.ctrl_indices = ctrl_idx
                self.session.ctrl_ids = [10 * (i + 1) for i in range(len(ctrl_idx))]
                self.session._next_ctrl_id = 10 * (len(ctrl_idx) + 1) if len(ctrl_idx) > 0 else 10
            self._update_lines()
            self.plotter.render()
            self._update_markers_saved()
            self._update_labels_saved()
            self._refresh_point_lists()
            L = self.session.centerline_result.length if self.session.centerline_result else None
            self._update_status(f"已推荐叶长（B1→T1 最短路径）：叶长={L:.6f}，并将 step 作为控制点。")
        except Exception as e:
            QMessageBox.critical(self, "推荐叶长失败", str(e))
        finally:
            self._finish_busy_dialog(dlg)


    def _open_length_params_dialog(self) -> bool:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("叶长推荐参数")
        layout = QtWidgets.QVBoxLayout(dlg)

        form = QtWidgets.QFormLayout()
        spin_step = QtWidgets.QDoubleSpinBox()
        spin_step.setRange(0.0001, 1.0)
        spin_step.setDecimals(6)
        spin_step.setSingleStep(0.001)
        spin_step.setValue(float(self.session.params.step))

        spin_radius = QtWidgets.QDoubleSpinBox()
        spin_radius.setRange(0.0001, 1.0)
        spin_radius.setDecimals(6)
        spin_radius.setSingleStep(0.001)
        spin_radius.setValue(float(self.session.params.radius))

        spin_min_pts = QtWidgets.QSpinBox()
        spin_min_pts.setRange(1, 100000)
        spin_min_pts.setValue(int(self.session.params.min_slice_pts))

        spin_graph_r = QtWidgets.QDoubleSpinBox()
        spin_graph_r.setRange(0.0001, 10.0)
        spin_graph_r.setDecimals(6)
        spin_graph_r.setSingleStep(0.001)
        spin_graph_r.setValue(float(self.session.params.graph_radius))

        form.addRow("步长 step：", spin_step)
        form.addRow("半径 radius：", spin_radius)
        form.addRow("最小截面点数 min_slice_pts：", spin_min_pts)
        form.addRow("连通半径 graph_radius：", spin_graph_r)
        layout.addLayout(form)

        tips = QtWidgets.QLabel(
            "说明：graph_radius 越大越连通但更慢；step 越小越细致但更慢；"
            "radius 越大越稳定但可能跨到邻近结构；min_slice_pts 太小易受噪声影响。"
        )
        tips.setWordWrap(True)
        layout.addWidget(tips)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return False

        self.session.params.step = float(spin_step.value())
        self.session.params.radius = float(spin_radius.value())
        self.session.params.min_slice_pts = int(spin_min_pts.value())
        self.session.params.graph_radius = float(spin_graph_r.value())
        self._update_status("已更新叶长推荐参数。")
        return True


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

        dlg = None
        try:
            dlg = self._start_busy_dialog("正在生成叶长...")
            self.session.compute_centerline_polyline(use_ctrl=True)
            self._update_lines()
            self.plotter.render()
            L = self.session.centerline_result.length if self.session.centerline_result else None
            self._update_status(f"已生成叶长（使用控制点）：叶长={L:.6f}")
            if self.session.current_inst_id is not None and self.session.centerline_result is not None:
                self.session.commit_current(False)
        except Exception as e:
            QMessageBox.critical(self, "生成叶长失败", str(e))
        finally:
            self._finish_busy_dialog(dlg)

    # ----------------------------
    # markers + labels
    # ----------------------------

    def _refresh_point_lists(self):
        self.list_base.clear()
        self.list_tip.clear()
        self.list_ctrl.clear()
        self.list_width.clear()
        self.list_width_ctrl.clear()

        if self.session.base_idx is not None:
            it = QListWidgetItem("B1")
            it.setData(0x0100, ("base", 0))
            self.list_base.addItem(it)

        if self.session.tip_idx is not None:
            it = QListWidgetItem("T1")
            it.setData(0x0100, ("tip", 0))
            self.list_tip.addItem(it)

        pairs = list(zip(self.session.ctrl_ids, self.session.ctrl_indices, range(len(self.session.ctrl_indices))))
        pairs.sort(key=lambda x: x[0])
        for cid, idx, orig_idx in pairs:
            it = QListWidgetItem(f"C{cid}")
            it.setData(0x0100, ("ctrl", orig_idx))
            self.list_ctrl.addItem(it)

        if getattr(self.session, "width_w1_idx", None) is not None:
            it = QListWidgetItem("W1")
            it.setData(0x0100, ("w", 1))
            self.list_width.addItem(it)
        if getattr(self.session, "width_w2_idx", None) is not None:
            it = QListWidgetItem("W2")
            it.setData(0x0100, ("w", 2))
            self.list_width.addItem(it)

        wpairs = list(zip(self.session.width_ctrl_ids, self.session.width_ctrl_indices, range(len(self.session.width_ctrl_indices))))
        wpairs.sort(key=lambda x: x[0])
        for cid, idx, orig_idx in wpairs:
            it = QListWidgetItem(f"WC{cid}")
            it.setData(0x0100, ("wctrl", orig_idx))
            self.list_width_ctrl.addItem(it)


    def _invalidate_results_after_point_change(self):
        self.session.centerline_result = None
        if hasattr(self.session, "centerline_source"):
            self.session.centerline_source = None
        self.session.width_path_points = None
        if hasattr(self.session, "width_path_length"):
            self.session.width_path_length = None
        self._update_lines()

    # ----------------------------
    # pick state machine
    # ----------------------------
