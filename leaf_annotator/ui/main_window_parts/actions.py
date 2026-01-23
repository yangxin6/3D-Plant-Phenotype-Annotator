# ui/main_window_parts/actions.py
import os

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox


class ActionsMixin:
    def on_select_instance_menu(self):
        if self.session.cloud is None or self.combo_inst.count() == 0:
            QMessageBox.information(self, "提示", "当前没有可选实例。")
            return
        items = [self.combo_inst.itemText(i) for i in range(self.combo_inst.count())]
        cur = self.combo_inst.currentText()
        cur_idx = items.index(cur) if cur in items else 0
        item, ok = QtWidgets.QInputDialog.getItem(self, "选择实例", "实例 id：", items, cur_idx, False)
        if not ok or not item:
            return
        self.combo_inst.setCurrentText(item)
        self.on_inst_changed()


    def on_choose_export_dir(self):
        export_dir = QFileDialog.getExistingDirectory(
            self, "选择标注导出目录", self._settings.value("export_dir", self._settings.value("last_dir", "", type=str), type=str)
        )
        if export_dir:
            self._settings.setValue("export_dir", export_dir)
            self._update_status(f"已设置导出目录：{export_dir}")

    # ----------------------------
    # UI actions
    # ----------------------------

    def on_load(self):
        last_dir = self._settings.value("last_dir", "", type=str)
        path, _ = QFileDialog.getOpenFileName(
            self, "选择整株点云文件", last_dir,
            "PointCloud (*.xyz *.txt *.csv *.npy *.ply *.pcd *.vtk *.vtp);;All (*.*)"
        )
        if not path:
            return
        self._settings.setValue("last_dir", os.path.dirname(path))

        dlg = None
        try:
            dlg = self._start_busy_dialog("正在读取点云...")
            self.session.load(path)
            if dlg is not None:
                dlg.setLabelText("正在导入标注/标签...")
                QtWidgets.QApplication.processEvents()

            base = os.path.splitext(os.path.basename(path))[0]
            data_dir = os.path.dirname(path)
            export_dir = self._settings.value("export_dir", "", type=str)
            cand_paths = [os.path.join(data_dir, f"{base}.json")]
            if export_dir:
                cand_paths.append(os.path.join(export_dir, f"{base}.json"))
            loaded_msgs = []
            for ap in cand_paths:
                if os.path.isfile(ap):
                    try:
                        self.session.load_annotations_json(ap)
                        loaded_msgs.append(f"已自动导入标注：{os.path.basename(ap)}")
                    except Exception:
                        pass
                    break
            self._set_plant_type(self.session.plant_type, update_status=False)
            loaded_msgs.append(f"植物类型：{self.session.plant_type}")

            label_paths = [os.path.join(data_dir, f"{base}_labels.txt")]
            if export_dir:
                label_paths.append(os.path.join(export_dir, f"{base}_labels.txt"))
            for lp in label_paths:
                if os.path.isfile(lp):
                    try:
                        full_labels = np.loadtxt(lp, dtype=np.int64)
                        if len(full_labels) == len(self.session.get_full_xyz()):
                            self.session.full_point_labels = full_labels
                            loaded_msgs.append(f"已自动导入标签：{os.path.basename(lp)}")
                    except Exception:
                        pass
                    break

            if dlg is not None:
                dlg.setLabelText("正在刷新界面...")
                QtWidgets.QApplication.processEvents()

            self._refresh_sem_filter_options()
            self._refresh_instance_list_for_annotation()

            self.annotating = False
            self.pick_mode = self.MODE_NONE
            self.temp_base_idx = None
            self.temp_tip_idx = None
            self.temp_ctrl_indices = []
            self.temp_w1_idx = None
            self.temp_w2_idx = None
            self.temp_width_ctrl_indices = []

            self._update_buttons()
            self._refresh_scene()
            self._refresh_point_lists()
            self._refresh_instance_meta_ui()
            self._update_instance_sem_label()
            self._update_view_legend()
            self._update_phenotype_table()
            loaded_msgs.append("浏览模式：可切换 RGB / 语义 / 实例 显示整株。")
            self._update_status("\n".join(loaded_msgs))
        except Exception as e:
            QMessageBox.critical(self, "读取失败", str(e))
            return
        finally:
            self._finish_busy_dialog(dlg)


    def on_start_annotation(self):
        key = self._get_annotation_semantic_key()
        sem_label = self.session.semantic_map.get(key)
        if sem_label is None:
            QMessageBox.information(self, "提示", "请先为当前标注语义选择语义标签。")
            return
        if self.combo_inst.count() == 0:
            QMessageBox.information(self, "提示", "当前没有可标注的实例（inst>=0）。")
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

        if key == "leaf":
            self._update_buttons()
            if self.session.point_labels is not None:
                self._refresh_scene(mode="表型标签")
            else:
                self._refresh_scene()
            self._refresh_point_lists()
            self._refresh_instance_meta_ui()
            self._update_instance_sem_label()
            self._update_view_legend()
            self._update_phenotype_table()

            extra = (
                "已有标注，可继续调整 base/tip/ctrl/W1/W2 或删除重标。"
                if self.session.is_current_annotated()
                else "请开始选择 base/tip/ctrl/W1/W2 标注点。"
            )
            self._update_status(
                f"进入标注模式：inst_id={inst_id}\n{extra}\n提示：Shift+左键选点。"
            )

            self._maybe_recommend_width_and_refresh()
        elif key == "stem":
            dlg = None
            try:
                dlg = self._start_busy_dialog("正在计算茎粗/茎长...")
                self.session.compute_stem_instance(inst_id)
            except Exception as e:
                QMessageBox.critical(self, "计算失败", str(e))
                return
            finally:
                self._finish_busy_dialog(dlg)
            self._update_buttons()
            self._refresh_scene()
            self._refresh_point_lists()
            self._refresh_instance_meta_ui()
            self._update_instance_sem_label()
            self._update_view_legend()
            self._update_phenotype_table()
            self._update_status(f"茎标注：inst_id={inst_id}，已计算茎粗/茎长。")
        elif key in ["flower", "fruit"]:
            dlg = None
            try:
                label = "花" if key == "flower" else "果"
                dlg = self._start_busy_dialog(f"正在计算{label}OBB...")
                self.session.compute_obb_instance(inst_id, key)
            except Exception as e:
                QMessageBox.critical(self, "计算失败", str(e))
                return
            finally:
                self._finish_busy_dialog(dlg)
            self._update_buttons()
            self._refresh_scene()
            self._refresh_point_lists()
            self._refresh_instance_meta_ui()
            self._update_instance_sem_label()
            self._update_view_legend()
            self._update_phenotype_table()
            self._update_status(f"{label}标注：inst_id={inst_id}，已计算{label}OBB。")


    def on_back_browse(self):
        self.annotating = False
        if self.pick_mode != self.MODE_NONE:
            self._exit_current_mode(commit=True)

        self._update_buttons()
        self._show_browse_scene()
        self._refresh_point_lists()
        self._refresh_instance_meta_ui()
        self._update_instance_sem_label()
        self._update_view_legend()
        self._update_phenotype_table()
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
            self.temp_width_ctrl_indices = []

            self.session.centerline_result = None
            self.session.width_path_points = None

            key = self._get_annotation_semantic_key()
            if key == "leaf":
                self._update_buttons()
                if self.session.point_labels is not None:
                    self._refresh_scene(mode="表型标签")
                else:
                    self._refresh_scene()
                self._refresh_point_lists()
                self._refresh_instance_meta_ui()
                self._update_instance_sem_label()
                self._update_view_legend()
                self._update_phenotype_table()

                extra = (
                    "已有标注，可继续调整 base/tip/ctrl/W1/W2 或删除重标。"
                    if self.session.is_current_annotated()
                    else "请开始选择 base/tip/ctrl/W1/W2 标注点。"
                )
                self._update_status(
                    f"切换标注实例：inst_id={inst_id}\n{extra}\n提示：Shift+左键选点。"
                )

                self._maybe_recommend_width_and_refresh()
            elif key == "stem":
                dlg = None
                try:
                    dlg = self._start_busy_dialog("正在计算茎粗/茎长...")
                    self.session.compute_stem_instance(inst_id)
                finally:
                    self._finish_busy_dialog(dlg)
                self._update_buttons()
                self._refresh_scene()
                self._refresh_point_lists()
                self._refresh_instance_meta_ui()
                self._update_instance_sem_label()
                self._update_view_legend()
                self._update_phenotype_table()
                self._update_status(f"茎标注：inst_id={inst_id}，已更新茎粗/茎长。")
            elif key in ["flower", "fruit"]:
                dlg = None
                label = "花" if key == "flower" else "果"
                try:
                    dlg = self._start_busy_dialog(f"正在计算{label}OBB...")
                    self.session.compute_obb_instance(inst_id, key)
                finally:
                    self._finish_busy_dialog(dlg)
                self._update_buttons()
                self._refresh_scene()
                self._refresh_point_lists()
                self._refresh_instance_meta_ui()
                self._update_instance_sem_label()
                self._update_view_legend()
                self._update_phenotype_table()
                self._update_status(f"{label}标注：inst_id={inst_id}，已更新{label}OBB。")

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


    def on_export_all(self):
        if self.session.get_annotations_count() == 0:
            QMessageBox.information(self, "提示", "当前文件还没有任何实例被标注。")
            return

        export_dir = self._settings.value("export_dir", "", type=str)
        if not export_dir:
            export_dir = QFileDialog.getExistingDirectory(self, "选择标注导出目录", self._settings.value("last_dir", "", type=str))
            if not export_dir:
                return
            self._settings.setValue("export_dir", export_dir)

        base = "plant_annotations"
        if self.session.file_path:
            base = os.path.splitext(os.path.basename(self.session.file_path))[0]
        out_path = os.path.join(export_dir, f"{base}.json")

        try:
            self.session.export_all_json(out_path)
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))
            return

        QMessageBox.information(self, "完成", f"已导出：{out_path}")
        self._update_buttons()
        self._update_status(f"已导出整株标注：{os.path.basename(out_path)}")

    def on_export_phenotype_csv(self):
        if self.session.cloud is None:
            return
        last_dir = self._settings.value("phenotype_export_dir", "", type=str)
        if not last_dir:
            last_dir = self._settings.value("last_dir", "", type=str)

        base = "plant_annotations"
        if self.session.file_path:
            base = os.path.splitext(os.path.basename(self.session.file_path))[0]
        default_name = f"{base}_phenotype.csv"
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出表型 CSV",
            os.path.join(last_dir, default_name),
            "CSV (*.csv);;All (*.*)"
        )
        if not out_path:
            return
        self._settings.setValue("phenotype_export_dir", os.path.dirname(out_path))

        try:
            self.session.export_phenotype_csv(out_path)
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))
            return

        QMessageBox.information(self, "完成", f"已导出：{out_path}")
        self._update_buttons()
        self._update_status(f"已导出表型数据：{os.path.basename(out_path)}")


    def on_export_labeled_cloud(self):
        radius, ok = QtWidgets.QInputDialog.getDouble(
            self, "标记半径", "输入 r（单位同点云）：", value=0.005,
            min=0.0, max=1000.0, decimals=6
        )
        if not ok:
            return
        self.session.params.label_radius = float(radius)
        if self.session.centerline_result is None:
            QMessageBox.information(self, "提示", "请先生成叶长或加载已有结果。")
            return
        if self.session.width_path_points is None:
            QMessageBox.information(self, "提示", "请先生成叶宽路径或加载已有结果。")
            return
        dlg = None
        try:
            dlg = self._start_busy_dialog("正在生成标注标签...")
            self.session.point_labels = self.session.compute_point_labels(self.session.params.label_radius)
        except Exception as e:
            QMessageBox.critical(self, "生成标签失败", str(e))
            return
        finally:
            self._finish_busy_dialog(dlg)
        if self.session.point_labels is None:
            QMessageBox.information(self, "提示", "当前无法生成标签，请先生成叶长和叶宽路径。")
            return
        if self.session.current_inst_id is not None:
            self.session.commit_current(False)
        self._refresh_scene(mode="表型标签")
        QMessageBox.information(self, "完成", "已生成标签并更新视图。")


    def on_compute_stem(self):
        if self.session.cloud is None:
            return
        if self.session.semantic_map.get("stem") is None:
            QMessageBox.information(self, "提示", "请先选择茎语义标签。")
            return
        if not self.on_stem_diameter_params_dialog():
            return
        dlg = None
        try:
            dlg = self._start_busy_dialog("正在计算茎粗...")
            count = self.session.compute_stem_diameter_structures()
        except Exception as e:
            QMessageBox.critical(self, "计算失败", str(e))
            return
        finally:
            self._finish_busy_dialog(dlg)
        if self.btn_toggle_stem_cyl.isChecked():
            self._update_stem_cylinders()
            self.plotter.render()
        self._update_phenotype_table()
        msg = f"已计算：茎粗 {count} 个。"
        QMessageBox.information(self, "完成", msg)
        self._update_status(msg)


    def on_compute_stem_length(self):
        if self.session.cloud is None:
            return
        if self.session.semantic_map.get("stem") is None:
            QMessageBox.information(self, "提示", "请先选择茎语义标签。")
            return
        if not self.on_stem_length_params_dialog():
            return
        dlg = None
        try:
            dlg = self._start_busy_dialog("正在计算茎长...")
            count = self.session.compute_stem_length_structures()
        except Exception as e:
            QMessageBox.critical(self, "计算失败", str(e))
            return
        finally:
            self._finish_busy_dialog(dlg)
        if hasattr(self, "btn_toggle_stem_path") and self.btn_toggle_stem_path.isChecked():
            self._update_stem_length_paths()
            self.plotter.render()
        self._update_phenotype_table()
        msg = f"已计算：茎长 {count} 个。"
        QMessageBox.information(self, "完成", msg)
        self._update_status(msg)


    def on_stem_diameter_params_dialog(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("茎粗参数")
        layout = QtWidgets.QVBoxLayout(dlg)

        form = QtWidgets.QFormLayout()
        spin_segments = QtWidgets.QSpinBox()
        spin_segments.setRange(1, 500)
        seg_val = int(getattr(self.session.params, "stem_diameter_segments", 0))
        if seg_val <= 0:
            seg_val = 20
        spin_segments.setValue(seg_val)

        spin_pct = QtWidgets.QDoubleSpinBox()
        spin_pct.setRange(50.0, 99.9)
        spin_pct.setDecimals(1)
        spin_pct.setSingleStep(0.5)
        spin_pct.setValue(float(getattr(self.session.params, "stem_diameter_percentile", 95.0)))

        form.addRow("分段数", spin_segments)
        form.addRow("半径百分位(%)", spin_pct)
        layout.addLayout(form)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return False

        self.session.params.stem_diameter_segments = int(spin_segments.value())
        self.session.params.stem_diameter_percentile = float(spin_pct.value())
        return True


    def on_stem_length_params_dialog(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("茎长参数")
        layout = QtWidgets.QVBoxLayout(dlg)

        form = QtWidgets.QFormLayout()
        spin_segments = QtWidgets.QSpinBox()
        spin_segments.setRange(1, 500)
        seg_val = int(getattr(self.session.params, "stem_length_segments", 0))
        if seg_val <= 0:
            seg_val = 20
        spin_segments.setValue(seg_val)

        spin_pct = QtWidgets.QDoubleSpinBox()
        spin_pct.setRange(50.0, 99.9)
        spin_pct.setDecimals(1)
        spin_pct.setSingleStep(0.5)
        spin_pct.setValue(float(getattr(self.session.params, "stem_length_percentile", 95.0)))

        form.addRow("分段数", spin_segments)
        form.addRow("半径百分位(%)", spin_pct)
        layout.addLayout(form)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return False

        self.session.params.stem_length_segments = int(spin_segments.value())
        self.session.params.stem_length_percentile = float(spin_pct.value())
        return True


    def on_compute_flower_fruit(self):
        if self.session.cloud is None:
            return
        if self.session.semantic_map.get("flower") is None and self.session.semantic_map.get("fruit") is None:
            QMessageBox.information(self, "提示", "请先选择花/果语义标签。")
            return
        dlg = None
        try:
            dlg = self._start_busy_dialog("正在计算花/果 OBB...")
            counts = self.session.compute_flower_fruit_obb()
        except Exception as e:
            QMessageBox.critical(self, "计算失败", str(e))
            return
        finally:
            self._finish_busy_dialog(dlg)
        self._update_phenotype_table()
        msg = f"已计算：花 {counts['flower']} 个，果 {counts['fruit']} 个。"
        QMessageBox.information(self, "完成", msg)
        self._update_status(msg)


    def on_save_annotations(self):
        has_meta = hasattr(self.session, "instance_meta") and len(self.session.instance_meta) > 0
        if self.session.get_annotations_count() == 0 and not has_meta:
            QMessageBox.information(self, "提示", "当前文件还没有任何实例被标注。")
            return
        if self.session.centerline_result is not None and self.session.width_path_points is not None:
            dlg = None
            try:
                dlg = self._start_busy_dialog("正在生成标注标签...")
                self.session.point_labels = self.session.compute_point_labels(self.session.params.label_radius)
            except Exception as e:
                QMessageBox.critical(self, "生成标签失败", str(e))
                return
            finally:
                self._finish_busy_dialog(dlg)
        if (
            self.session.current_inst_id is not None
            and (
                self.session.centerline_result is not None
                or self.session.width_path_points is not None
                or self.session.point_labels is not None
            )
        ):
            self.session.commit_current(False)
        export_dir = self._settings.value("export_dir", "", type=str)
        if not export_dir:
            export_dir = QFileDialog.getExistingDirectory(self, "选择标注导出目录", self._settings.value("last_dir", "", type=str))
            if not export_dir:
                return
            self._settings.setValue("export_dir", export_dir)

        base = "plant_annotations"
        if self.session.file_path:
            base = os.path.splitext(os.path.basename(self.session.file_path))[0]
        json_path = os.path.join(export_dir, f"{base}.json")
        label_path = os.path.join(export_dir, f"{base}_labels.txt")
        try:
            self.session.export_all_json(json_path)
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))
            return

        QMessageBox.information(self, "完成", f"已保存标注：{json_path}\n标签已包含在 JSON 中。")


    def on_show_help(self):
        text = (
            "<div style='font-size:12pt; line-height:1.5;'>"
            "<b>基础流程</b><br>"
            "读取点云 → 选择语义标签 → 选择实例 → 点击标注进入<br><br>"
            "<b>叶子标注</b><br>"
            "Shift+左键拾取 叶基/叶尖/叶宽端点/控制点<br>"
            "可用功能：生成叶长/叶宽/面积，推荐叶宽，平滑曲线，计算叶倾角/叶夹角<br><br>"
            "<b>茎/花/果</b><br>"
            "茎：计算茎粗/茎长，可显示圆柱与茎长路径<br>"
            "花/果：计算 OBB，并显示边长<br><br>"
            "<b>视图与显示</b><br>"
            "视图：正/侧/顶视图；显示 AABB/OBB；选择视图中心会显示黑色标记点<br>"
            "显示模式：RGB / 语义 / 实例 / 表型标签 视图切换<br><br>"
            "<b>导出与快捷键</b><br>"
            "保存标注：Ctrl + S；导出表型 CSV；导出标注点云<br>"
            "提示：先开启对应拾取模式再点击点云"
            "</div>"
        )
        dlg = QMessageBox(self)
        dlg.setWindowTitle("使用说明")
        dlg.setIcon(QMessageBox.Information)
        dlg.setTextFormat(Qt.RichText)
        dlg.setText(text)
        dlg.exec_()


    def on_show_about(self):
        QMessageBox.information(
            self,
            "关于",
            "版本：v0.1.0\n"
            "开发人：杨鑫,苗腾\n"
            "邮件：yangxinnc@163.com\n"
            "单位：沈阳农业大学 信息与电气工程学院",
        )
