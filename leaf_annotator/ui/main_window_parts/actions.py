# ui/main_window_parts/actions.py
import copy
import csv
import json
import os

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from core.io import PointCloudIO


class ActionsMixin:
    def _iter_point_cloud_files(self, folder: str):
        exts = {ext.lower() for ext in PointCloudIO.SUPPORTED}
        files = []
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                continue
            if os.path.splitext(name)[1].lower() in exts:
                files.append(path)
        return files


    def _resolve_batch_json_path(self, cloud_path: str, export_dir: str) -> str:
        base = os.path.splitext(os.path.basename(cloud_path))[0]
        cloud_json = os.path.join(os.path.dirname(cloud_path), f"{base}.json")
        export_json = os.path.join(export_dir, f"{base}.json")
        if os.path.isfile(cloud_json):
            return cloud_json
        if os.path.isfile(export_json):
            return export_json
        return export_json


    def _apply_semantic_fallback(self, session_obj):
        fallback = getattr(self.session, "semantic_map", {}) or {}
        for key in ["leaf", "stem", "flower", "fruit"]:
            if session_obj.semantic_map.get(key) is None and fallback.get(key) is not None:
                session_obj.semantic_map[key] = int(fallback[key])
        if not getattr(session_obj, "plant_type", None):
            session_obj.plant_type = self.session.plant_type


    def _rows_from_json_data(self, data: dict, file_name: str):
        rows = []
        plant_meas = data.get("plant_measurements") or {}
        height = plant_meas.get("height")
        absolute_height = plant_meas.get("absolute_height")
        crown = plant_meas.get("crown_width")
        if height is not None:
            rows.append([file_name, self.tr("整株"), "-", self.tr("株高"), f"{float(height):.3f}"])
        if absolute_height is not None:
            rows.append([file_name, self.tr("整株"), "-", self.tr("绝对株高"), f"{float(absolute_height):.3f}"])
        if crown is not None:
            rows.append([file_name, self.tr("整株"), "-", self.tr("冠幅"), f"{float(crown):.3f}"])

        for ann in data.get("annotations", []):
            inst_id = ann.get("inst_id", "")
            if ann.get("stem_diameter") is not None:
                rows.append([file_name, str(inst_id), self.tr("茎"), self.tr("茎粗"), f"{float(ann['stem_diameter']):.3f}"])
            if ann.get("stem_length") is not None:
                rows.append([file_name, str(inst_id), self.tr("茎"), self.tr("茎长"), f"{float(ann['stem_length']):.3f}"])
            if ann.get("length") is not None:
                rows.append([file_name, str(inst_id), self.tr("叶"), self.tr("叶长"), f"{float(ann['length']):.3f}"])
            if ann.get("width_path_length") is not None:
                rows.append([file_name, str(inst_id), self.tr("叶"), self.tr("叶宽"), f"{float(ann['width_path_length']):.3f}"])
            if ann.get("leaf_area") is not None:
                rows.append([file_name, str(inst_id), self.tr("叶"), self.tr("叶面积"), f"{float(ann['leaf_area']):.3f}"])
            if ann.get("leaf_projected_area") is not None:
                rows.append([file_name, str(inst_id), self.tr("叶"), self.tr("投影面积"), f"{float(ann['leaf_projected_area']):.3f}"])
            if ann.get("leaf_inclination") is not None:
                rows.append([file_name, str(inst_id), self.tr("叶"), self.tr("叶倾角"), f"{float(ann['leaf_inclination']):.1f}"])
            if ann.get("leaf_stem_angle") is not None:
                rows.append([file_name, str(inst_id), self.tr("叶"), self.tr("叶夹角"), f"{float(ann['leaf_stem_angle']):.1f}"])
            flower_obb = ann.get("flower_obb")
            if isinstance(flower_obb, dict):
                lengths = flower_obb.get("lengths")
                if lengths is not None and len(lengths) == 3:
                    rows.append([file_name, str(inst_id), self.tr("花"), self.tr("花OBB"), f"{lengths[0]:.3f},{lengths[1]:.3f},{lengths[2]:.3f}"])
            fruit_obb = ann.get("fruit_obb")
            if isinstance(fruit_obb, dict):
                lengths = fruit_obb.get("lengths")
                if lengths is not None and len(lengths) == 3:
                    rows.append([file_name, str(inst_id), self.tr("果"), self.tr("果OBB"), f"{lengths[0]:.3f},{lengths[1]:.3f},{lengths[2]:.3f}"])
        return rows


    def _compute_leaf_inclination_batch(self, session_obj):
        leaf_label = session_obj.semantic_map.get("leaf")
        if leaf_label is None:
            return 0
        sem_map = session_obj._get_instance_sem_map()
        count = 0
        for inst_id, sem_label in sem_map.items():
            if int(sem_label) != int(leaf_label):
                continue
            try:
                session_obj.select_instance(int(inst_id))
                angle = session_obj.compute_leaf_inclination_instance()
                if angle is not None:
                    count += 1
            except Exception:
                continue
        return count


    def _compute_leaf_stem_angle_batch(self, session_obj):
        leaf_label = session_obj.semantic_map.get("leaf")
        if leaf_label is None:
            return 0
        sem_map = session_obj._get_instance_sem_map()
        count = 0
        for inst_id, sem_label in sem_map.items():
            if int(sem_label) != int(leaf_label):
                continue
            try:
                session_obj.select_instance(int(inst_id))
                angle = session_obj.compute_leaf_stem_angle_instance()
                if angle is not None:
                    count += 1
            except Exception:
                continue
        return count


    def _open_batch_compute_dialog(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(self.tr("批量快速计算设置"))
        layout = QtWidgets.QVBoxLayout(dlg)

        form = QtWidgets.QFormLayout()
        cloud_edit = QtWidgets.QLineEdit(self._settings.value("last_dir", "", type=str))
        export_edit = QtWidgets.QLineEdit(self._settings.value("export_dir", "", type=str))

        cloud_row = QtWidgets.QHBoxLayout()
        cloud_row.addWidget(cloud_edit)
        btn_cloud = QtWidgets.QPushButton(self.tr("浏览"))
        cloud_row.addWidget(btn_cloud)

        export_row = QtWidgets.QHBoxLayout()
        export_row.addWidget(export_edit)
        btn_export = QtWidgets.QPushButton(self.tr("浏览"))
        export_row.addWidget(btn_export)

        form.addRow(self.tr("点云目录"), cloud_row)
        form.addRow(self.tr("JSON输出目录"), export_row)
        layout.addLayout(form)

        tips = QtWidgets.QLabel(self.tr("确认目录后，点击“开始计算”执行批量计算。"))
        tips.setWordWrap(True)
        layout.addWidget(tips)

        def choose_cloud():
            folder = QFileDialog.getExistingDirectory(
                self,
                self.tr("选择点云目录"),
                cloud_edit.text() or self._settings.value("last_dir", "", type=str),
            )
            if folder:
                cloud_edit.setText(folder)

        def choose_export():
            folder = QFileDialog.getExistingDirectory(
                self,
                self.tr("选择批量JSON导出目录"),
                export_edit.text() or cloud_edit.text() or self._settings.value("export_dir", "", type=str),
            )
            if folder:
                export_edit.setText(folder)

        btn_cloud.clicked.connect(choose_cloud)
        btn_export.clicked.connect(choose_export)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.button(QtWidgets.QDialogButtonBox.Ok).setText(self.tr("开始计算"))
        btns.button(QtWidgets.QDialogButtonBox.Cancel).setText(self.tr("取消"))
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return None

        cloud_dir = (cloud_edit.text() or "").strip()
        export_dir = (export_edit.text() or "").strip()
        if not cloud_dir or not os.path.isdir(cloud_dir):
            QMessageBox.information(self, self.tr("提示"), self.tr("请先选择有效的点云目录。"))
            return None
        if not export_dir:
            export_dir = cloud_dir
        if not os.path.isdir(export_dir):
            QMessageBox.information(self, self.tr("提示"), self.tr("请先选择有效的JSON输出目录。"))
            return None

        self._settings.setValue("last_dir", cloud_dir)
        self._settings.setValue("export_dir", export_dir)
        return cloud_dir, export_dir


    def _open_batch_export_dialog(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(self.tr("批量导出设置"))
        layout = QtWidgets.QVBoxLayout(dlg)

        form = QtWidgets.QFormLayout()
        json_dir_edit = QtWidgets.QLineEdit(
            self._settings.value("export_dir", self._settings.value("last_dir", "", type=str), type=str)
        )
        out_csv_edit = QtWidgets.QLineEdit()

        if json_dir_edit.text().strip():
            out_csv_edit.setText(os.path.join(json_dir_edit.text().strip(), "batch_phenotypes.csv"))

        json_row = QtWidgets.QHBoxLayout()
        json_row.addWidget(json_dir_edit)
        btn_json = QtWidgets.QPushButton(self.tr("浏览"))
        json_row.addWidget(btn_json)

        csv_row = QtWidgets.QHBoxLayout()
        csv_row.addWidget(out_csv_edit)
        btn_csv = QtWidgets.QPushButton(self.tr("浏览"))
        csv_row.addWidget(btn_csv)

        form.addRow(self.tr("JSON目录"), json_row)
        form.addRow(self.tr("CSV输出文件"), csv_row)
        layout.addLayout(form)

        tips = QtWidgets.QLabel(self.tr("确认参数后，点击“开始导出”执行批量导出。"))
        tips.setWordWrap(True)
        layout.addWidget(tips)

        def choose_json():
            folder = QFileDialog.getExistingDirectory(
                self,
                self.tr("选择JSON目录"),
                json_dir_edit.text() or self._settings.value("export_dir", "", type=str),
            )
            if folder:
                json_dir_edit.setText(folder)
                if not out_csv_edit.text().strip():
                    out_csv_edit.setText(os.path.join(folder, "batch_phenotypes.csv"))

        def choose_csv():
            base = out_csv_edit.text().strip()
            if not base:
                base = os.path.join(json_dir_edit.text().strip() or ".", "batch_phenotypes.csv")
            path, _ = QFileDialog.getSaveFileName(
                self,
                self.tr("选择CSV输出文件"),
                base,
                "CSV (*.csv);;All (*.*)",
            )
            if path:
                out_csv_edit.setText(path)

        btn_json.clicked.connect(choose_json)
        btn_csv.clicked.connect(choose_csv)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.button(QtWidgets.QDialogButtonBox.Ok).setText(self.tr("开始导出"))
        btns.button(QtWidgets.QDialogButtonBox.Cancel).setText(self.tr("取消"))
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return None

        json_dir = (json_dir_edit.text() or "").strip()
        out_path = (out_csv_edit.text() or "").strip()
        if not json_dir or not os.path.isdir(json_dir):
            QMessageBox.information(self, self.tr("提示"), self.tr("请先选择有效的JSON目录。"))
            return None
        if not out_path:
            QMessageBox.information(self, self.tr("提示"), self.tr("请先选择有效的CSV输出文件路径。"))
            return None
        self._settings.setValue("export_dir", json_dir)
        return json_dir, out_path


    def on_batch_compute_quick(self):
        selected = self._open_batch_compute_dialog()
        if not selected:
            return
        cloud_dir, export_dir = selected
        cloud_files = self._iter_point_cloud_files(cloud_dir)
        if not cloud_files:
            QMessageBox.information(self, self.tr("提示"), self.tr("所选目录中没有可处理的点云文件。"))
            return

        QMessageBox.information(
            self,
            self.tr("批量快速计算"),
            self.tr(
                "批量快速计算步骤：\n"
                "1. 逐个读取点云；\n"
                "2. 读取或创建同名 JSON；\n"
                "3. 自动计算生长方向；\n"
                "4. 计算茎粗、茎长、株高、冠幅；\n"
                "5. 计算叶倾角；\n"
                "6. 计算叶夹角；\n"
                "7. 保存回对应 JSON。"
            ),
        )

        dlg = None
        processed = 0
        skipped = []
        try:
            dlg = self._start_busy_dialog(self.tr("正在批量计算表型..."))
            session_cls = self.session.__class__
            for idx, cloud_path in enumerate(cloud_files, 1):
                if dlg is not None:
                    dlg.setLabelText(self.tr("正在处理：{name} ({index}/{total})", name=os.path.basename(cloud_path), index=idx, total=len(cloud_files)))
                    QtWidgets.QApplication.processEvents()
                self._update_status(self.tr("批量计算：准备处理 {name} ({index}/{total})", name=os.path.basename(cloud_path), index=idx, total=len(cloud_files)))
                batch_session = session_cls(params=copy.deepcopy(self.session.params), schema=copy.deepcopy(self.session.schema))
                if dlg is not None:
                    dlg.setLabelText(self.tr("步骤 1/7：读取点云 {name}", name=os.path.basename(cloud_path)))
                    QtWidgets.QApplication.processEvents()
                batch_session.load(cloud_path)
                json_path = self._resolve_batch_json_path(cloud_path, export_dir)
                if dlg is not None:
                    dlg.setLabelText(self.tr("步骤 2/7：读取或创建 JSON {name}", name=os.path.basename(json_path)))
                    QtWidgets.QApplication.processEvents()
                if os.path.isfile(json_path):
                    try:
                        batch_session.load_annotations_json(json_path)
                    except Exception:
                        pass
                self._apply_semantic_fallback(batch_session)
                if batch_session.semantic_map.get("stem") is None:
                    skipped.append(f"{os.path.basename(cloud_path)}: {self.tr('未设置茎语义标签')}")
                    continue
                try:
                    if dlg is not None:
                        dlg.setLabelText(self.tr("步骤 3/7：计算生长方向 {name}", name=os.path.basename(cloud_path)))
                        QtWidgets.QApplication.processEvents()
                    batch_session.compute_growth_direction_from_stem()
                    if dlg is not None:
                        dlg.setLabelText(self.tr("步骤 4/7：计算茎粗、茎长、株高、冠幅 {name}", name=os.path.basename(cloud_path)))
                        QtWidgets.QApplication.processEvents()
                    batch_session.compute_stem_diameter_structures()
                    batch_session.compute_stem_length_structures()
                    batch_session.compute_plant_height_crown()
                    if dlg is not None:
                        dlg.setLabelText(self.tr("步骤 5/7：计算叶倾角 {name}", name=os.path.basename(cloud_path)))
                        QtWidgets.QApplication.processEvents()
                    self._compute_leaf_inclination_batch(batch_session)
                    if dlg is not None:
                        dlg.setLabelText(self.tr("步骤 6/7：计算叶夹角 {name}", name=os.path.basename(cloud_path)))
                        QtWidgets.QApplication.processEvents()
                    self._compute_leaf_stem_angle_batch(batch_session)
                    if dlg is not None:
                        dlg.setLabelText(self.tr("步骤 7/7：保存 JSON {name}", name=os.path.basename(json_path)))
                        QtWidgets.QApplication.processEvents()
                    batch_session.export_all_json(json_path)
                    processed += 1
                    self._update_status(self.tr("批量计算：已完成 {name}", name=os.path.basename(cloud_path)))
                except Exception as e:
                    skipped.append(f"{os.path.basename(cloud_path)}: {e}")
        finally:
            self._finish_busy_dialog(dlg)

        msg = self.tr("批量计算完成：成功 {count} 个。", count=processed)
        if skipped:
            tail = "\n".join(skipped[:10])
            if len(skipped) > 10:
                tail += f"\n... (+{len(skipped)-10})"
            msg = f"{msg}\n{self.tr('跳过/失败：')}\n{tail}"
        QMessageBox.information(self, self.tr("完成"), msg)
        self._update_status(msg)


    def on_batch_export_phenotype_csv(self):
        selected = self._open_batch_export_dialog()
        if not selected:
            return
        json_dir, out_path = selected
        json_files = [
            os.path.join(json_dir, name)
            for name in sorted(os.listdir(json_dir))
            if os.path.isfile(os.path.join(json_dir, name)) and name.lower().endswith('.json')
        ]
        if not json_files:
            QMessageBox.information(self, self.tr("提示"), self.tr("所选目录中没有JSON文件。"))
            return

        QMessageBox.information(
            self,
            self.tr("批量导出表型CSV"),
            self.tr(
                "批量导出 CSV 步骤：\n"
                "1. 扫描目录中的 JSON；\n"
                "2. 逐个读取表型数据；\n"
                "3. 汇总为一个 CSV 文件。"
            ),
        )

        rows = []
        skipped = []
        dlg = None
        try:
            dlg = self._start_busy_dialog(self.tr("正在批量导出表型CSV..."))
            for idx, path in enumerate(json_files, 1):
                if dlg is not None:
                    dlg.setLabelText(self.tr("步骤 1/3：读取 JSON {name} ({index}/{total})", name=os.path.basename(path), index=idx, total=len(json_files)))
                    QtWidgets.QApplication.processEvents()
                self._update_status(self.tr("批量导出：读取 {name} ({index}/{total})", name=os.path.basename(path), index=idx, total=len(json_files)))
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if dlg is not None:
                        dlg.setLabelText(self.tr("步骤 2/3：提取表型数据 {name}", name=os.path.basename(path)))
                        QtWidgets.QApplication.processEvents()
                    rows.extend(self._rows_from_json_data(data, os.path.basename(path)))
                except Exception as e:
                    skipped.append(f"{os.path.basename(path)}: {e}")
            if not rows:
                QMessageBox.information(self, self.tr("提示"), self.tr("未找到可导出的表型数据。"))
                return

            if dlg is not None:
                dlg.setLabelText(self.tr("步骤 3/3：写出 CSV {name}", name=os.path.basename(out_path)))
                QtWidgets.QApplication.processEvents()
            with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([self.tr("文件"), self.tr("实例"), self.tr("语义"), self.tr("表型名字"), self.tr("表型值")])
                writer.writerows(rows)
        finally:
            self._finish_busy_dialog(dlg)

        msg = self.tr("已批量导出表型数据：{path}", path=out_path)
        if skipped:
            msg = f"{msg}\n{self.tr('跳过/失败：{count}', count=len(skipped))}"
        QMessageBox.information(self, self.tr("完成"), msg)
        self._update_status(msg)

    def on_scale_cloud(self):
        if self.session.cloud is None:
            return
        factor, ok = QtWidgets.QInputDialog.getDouble(
            self,
            self.tr("缩放点云"),
            self.tr("缩放倍数（例如 0.1 表示缩小 10 倍）："),
            value=1.0,
            min=0.000001,
            max=1000000.0,
            decimals=6,
        )
        if not ok:
            return
        factor = float(factor)
        if abs(factor - 1.0) < 1e-12:
            return
        confirm = QMessageBox.question(
            self,
            self.tr("缩放点云"),
            self.tr("将点云缩放为原来的 {factor} 倍，当前标注和测量将被清空，是否继续？", factor=factor),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        try:
            self.session.scale_cloud(factor)
        except Exception as e:
            QMessageBox.critical(self, self.tr("计算失败"), self.tr(str(e)))
            return

        self.annotating = False
        self.pick_mode = self.MODE_NONE
        self.temp_base_idx = None
        self.temp_tip_idx = None
        self.temp_ctrl_indices = []
        self.temp_w1_idx = None
        self.temp_w2_idx = None
        self.temp_width_ctrl_indices = []
        self.temp_measure_p1 = None
        self.temp_measure_p2 = None

        self._update_buttons()
        self._refresh_scene()
        self._refresh_point_lists()
        self._refresh_instance_meta_ui()
        self._update_instance_sem_label()
        self._update_view_legend()
        self._update_phenotype_table()
        self._update_status(self.tr("已缩放点云：{factor} 倍。", factor=factor))

    def on_select_instance_menu(self):
        if self.session.cloud is None or self.combo_inst.count() == 0:
            QMessageBox.information(self, self.tr("提示"), self.tr("当前没有可选实例。"))
            return
        items = [self.combo_inst.itemText(i) for i in range(self.combo_inst.count())]
        cur = self.combo_inst.currentText()
        cur_idx = items.index(cur) if cur in items else 0
        item, ok = QtWidgets.QInputDialog.getItem(
            self, self.tr("选择实例"), self.tr("实例 id："), items, cur_idx, False
        )
        if not ok or not item:
            return
        self.combo_inst.setCurrentText(item)
        self.on_inst_changed()


    def on_choose_export_dir(self):
        export_dir = QFileDialog.getExistingDirectory(
            self, self.tr("选择标注导出目录"),
            self._settings.value("export_dir", self._settings.value("last_dir", "", type=str), type=str)
        )
        if export_dir:
            self._settings.setValue("export_dir", export_dir)
            self._update_status(self.tr("已设置导出目录：{path}", path=export_dir))

    # ----------------------------
    # UI actions
    # ----------------------------

    def on_load(self):
        last_dir = self._settings.value("last_dir", "", type=str)
        path, _ = QFileDialog.getOpenFileName(
            self, self.tr("选择整株点云文件"), last_dir,
            "PointCloud (*.xyz *.txt *.csv *.npy *.ply *.pcd *.vtk *.vtp);;All (*.*)"
        )
        if not path:
            return
        self._settings.setValue("last_dir", os.path.dirname(path))

        dlg = None
        try:
            dlg = self._start_busy_dialog(self.tr("正在读取点云..."))
            self.session.load(path)
            if dlg is not None:
                dlg.setLabelText(self.tr("正在导入标注/标签..."))
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
                        loaded_msgs.append(self.tr("已自动导入标注：{name}", name=os.path.basename(ap)))
                    except Exception:
                        pass
                    break
            self._set_plant_type(self.session.plant_type, update_status=False)
            loaded_msgs.append(self.tr("植物类型：{plant_type}", plant_type=self._plant_type_display(self.session.plant_type)))

            label_paths = [os.path.join(data_dir, f"{base}_labels.txt")]
            if export_dir:
                label_paths.append(os.path.join(export_dir, f"{base}_labels.txt"))
            for lp in label_paths:
                if os.path.isfile(lp):
                    try:
                        full_labels = np.loadtxt(lp, dtype=np.int64)
                        if len(full_labels) == len(self.session.get_full_xyz()):
                            self.session.full_point_labels = full_labels
                            loaded_msgs.append(self.tr("已自动导入标签：{name}", name=os.path.basename(lp)))
                    except Exception:
                        pass
                    break

            if dlg is not None:
                dlg.setLabelText(self.tr("正在刷新界面..."))
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
            loaded_msgs.append(self.tr("浏览模式：可切换 RGB / 语义 / 实例 显示整株。"))
            self._update_status("\n".join(loaded_msgs))
        except Exception as e:
            QMessageBox.critical(self, self.tr("读取失败"), self.tr(str(e)))
            return
        finally:
            self._finish_busy_dialog(dlg)


    def on_start_annotation(self):
        key = self._get_annotation_semantic_key()
        sem_label = self.session.semantic_map.get(key)
        if sem_label is None:
            QMessageBox.information(self, self.tr("提示"), self.tr("请先为当前标注语义选择语义标签。"))
            return
        if self.combo_inst.count() == 0:
            QMessageBox.information(self, self.tr("提示"), self.tr("当前没有可标注的实例（inst>=0）。"))
            return
        try:
            inst_id = int(self.combo_inst.currentText())
            self.session.select_instance(inst_id)
        except Exception as e:
            QMessageBox.critical(self, self.tr("进入标注失败"), self.tr(str(e)))
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
                self._refresh_scene(mode=self.VIEW_LABEL)
            else:
                self._refresh_scene()
            self._refresh_point_lists()
            self._refresh_instance_meta_ui()
            self._update_instance_sem_label()
            self._update_view_legend()
            self._update_phenotype_table()

            extra = (
                self.tr("已有标注，可继续调整 base/tip/ctrl/W1/W2 或删除重标。")
                if self.session.is_current_annotated()
                else self.tr("请开始选择 base/tip/ctrl/W1/W2 标注点。")
            )
            self._update_status(self.tr(
                "进入标注模式：inst_id={inst_id}\n{extra}\n提示：Shift+左键选点。",
                inst_id=inst_id,
                extra=extra,
            ))

            self._maybe_recommend_width_and_refresh()
        elif key == "stem":
            dlg = None
            try:
                dlg = self._start_busy_dialog(self.tr("正在计算茎粗/茎长..."))
                self.session.compute_stem_instance(inst_id)
            except Exception as e:
                QMessageBox.critical(self, self.tr("计算失败"), self.tr(str(e)))
                return
            finally:
                self._finish_busy_dialog(dlg)
            if hasattr(self, "btn_toggle_stem_cyl"):
                self.btn_toggle_stem_cyl.blockSignals(True)
                self.btn_toggle_stem_cyl.setChecked(True)
                self.btn_toggle_stem_cyl.blockSignals(False)
            self._update_buttons()
            self._refresh_scene()
            self._refresh_point_lists()
            self._refresh_instance_meta_ui()
            self._update_instance_sem_label()
            self._update_view_legend()
            self._update_phenotype_table()
            self._update_status(self.tr("茎标注：inst_id={inst_id}，已计算茎粗/茎长。", inst_id=inst_id))
        elif key in ["flower", "fruit"]:
            dlg = None
            try:
                label = self.tr("花") if key == "flower" else self.tr("果")
                dlg = self._start_busy_dialog(self.tr("正在计算{label}OBB...", label=label))
                self.session.compute_obb_instance(inst_id, key)
            except Exception as e:
                QMessageBox.critical(self, self.tr("计算失败"), self.tr(str(e)))
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
            self._update_status(self.tr("{label}标注：inst_id={inst_id}，已计算{label}OBB。", label=label, inst_id=inst_id))


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
        self._update_status(self.tr("已返回浏览模式：显示整株点云。"))


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
                    self._refresh_scene(mode=self.VIEW_LABEL)
                else:
                    self._refresh_scene()
                self._refresh_point_lists()
                self._refresh_instance_meta_ui()
                self._update_instance_sem_label()
                self._update_view_legend()
                self._update_phenotype_table()

                extra = (
                    self.tr("已有标注，可继续调整 base/tip/ctrl/W1/W2 或删除重标。")
                    if self.session.is_current_annotated()
                    else self.tr("请开始选择 base/tip/ctrl/W1/W2 标注点。")
                )
                self._update_status(self.tr(
                    "切换标注实例：inst_id={inst_id}\n{extra}\n提示：Shift+左键选点。",
                    inst_id=inst_id,
                    extra=extra,
                ))

                self._maybe_recommend_width_and_refresh()
            elif key == "stem":
                dlg = None
                try:
                    dlg = self._start_busy_dialog(self.tr("正在计算茎粗/茎长..."))
                    self.session.compute_stem_instance(inst_id)
                finally:
                    self._finish_busy_dialog(dlg)
                if hasattr(self, "btn_toggle_stem_cyl"):
                    self.btn_toggle_stem_cyl.blockSignals(True)
                    self.btn_toggle_stem_cyl.setChecked(True)
                    self.btn_toggle_stem_cyl.blockSignals(False)
                self._update_buttons()
                self._refresh_scene()
                self._refresh_point_lists()
                self._refresh_instance_meta_ui()
                self._update_instance_sem_label()
                self._update_view_legend()
                self._update_phenotype_table()
                self._update_status(self.tr("茎标注：inst_id={inst_id}，已更新茎粗/茎长。", inst_id=inst_id))
            elif key in ["flower", "fruit"]:
                dlg = None
                label = self.tr("花") if key == "flower" else self.tr("果")
                try:
                    dlg = self._start_busy_dialog(self.tr("正在计算{label}OBB...", label=label))
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
                self._update_status(self.tr("{label}标注：inst_id={inst_id}，已更新{label}OBB。", label=label, inst_id=inst_id))

        except Exception as e:
            QMessageBox.critical(self, self.tr("切换实例失败"), self.tr(str(e)))


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
        self._update_status(self.tr("已清空控制点（已保存 + 临时）。"))


    def on_export_all(self):
        if self.session.get_annotations_count() == 0:
            QMessageBox.information(self, self.tr("提示"), self.tr("当前文件还没有任何实例被标注。"))
            return

        export_dir = self._settings.value("export_dir", "", type=str)
        if not export_dir:
            export_dir = QFileDialog.getExistingDirectory(self, self.tr("选择标注导出目录"), self._settings.value("last_dir", "", type=str))
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
            QMessageBox.critical(self, self.tr("导出失败"), self.tr(str(e)))
            return

        QMessageBox.information(self, self.tr("完成"), self.tr("已导出：{path}", path=out_path))
        self._update_buttons()
        self._update_status(self.tr("已导出整株标注：{name}", name=os.path.basename(out_path)))

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
            self.tr("导出表型 CSV"),
            os.path.join(last_dir, default_name),
            "CSV (*.csv);;All (*.*)"
        )
        if not out_path:
            return
        self._settings.setValue("phenotype_export_dir", os.path.dirname(out_path))

        try:
            self.session.export_phenotype_csv(out_path)
        except Exception as e:
            QMessageBox.critical(self, self.tr("导出失败"), self.tr(str(e)))
            return

        QMessageBox.information(self, self.tr("完成"), self.tr("已导出：{path}", path=out_path))
        self._update_buttons()
        self._update_status(self.tr("已导出表型数据：{name}", name=os.path.basename(out_path)))


    def on_export_labeled_cloud(self):
        radius, ok = QtWidgets.QInputDialog.getDouble(
            self, self.tr("标记半径"), self.tr("输入 r（单位同点云）："), value=0.005,
            min=0.0, max=1000.0, decimals=6
        )
        if not ok:
            return
        self.session.params.label_radius = float(radius)
        if self.session.centerline_result is None:
            QMessageBox.information(self, self.tr("提示"), self.tr("请先生成叶长或加载已有结果。"))
            return
        if self.session.width_path_points is None:
            QMessageBox.information(self, self.tr("提示"), self.tr("请先生成叶宽路径或加载已有结果。"))
            return
        dlg = None
        try:
            dlg = self._start_busy_dialog(self.tr("正在生成标注标签..."))
            self.session.point_labels = self.session.compute_point_labels(self.session.params.label_radius)
        except Exception as e:
            QMessageBox.critical(self, self.tr("生成标签失败"), self.tr(str(e)))
            return
        finally:
            self._finish_busy_dialog(dlg)
        if self.session.point_labels is None:
            QMessageBox.information(self, self.tr("提示"), self.tr("当前无法生成标签，请先生成叶长和叶宽路径。"))
            return
        if self.session.current_inst_id is not None:
            self.session.commit_current(False)
        self._refresh_scene(mode=self.VIEW_LABEL)
        QMessageBox.information(self, self.tr("完成"), self.tr("已生成标签并更新视图。"))


    def on_compute_stem(self):
        if self.session.cloud is None:
            return
        if self.session.semantic_map.get("stem") is None:
            QMessageBox.information(self, self.tr("提示"), self.tr("请先选择茎语义标签。"))
            return
        if not self.on_stem_diameter_params_dialog():
            return
        dlg = None
        try:
            dlg = self._start_busy_dialog(self.tr("正在计算茎粗..."))
            count = self.session.compute_stem_diameter_structures()
        except Exception as e:
            QMessageBox.critical(self, self.tr("计算失败"), self.tr(str(e)))
            return
        finally:
            self._finish_busy_dialog(dlg)
        if hasattr(self, "btn_toggle_stem_cyl"):
            self.btn_toggle_stem_cyl.blockSignals(True)
            self.btn_toggle_stem_cyl.setChecked(True)
            self.btn_toggle_stem_cyl.blockSignals(False)
            self._update_stem_cylinders()
            self.plotter.render()
        self._update_phenotype_table()
        msg = self.tr("已计算：茎粗 {count} 个。", count=count)
        QMessageBox.information(self, self.tr("完成"), msg)
        self._update_status(msg)


    def on_compute_stem_length(self):
        if self.session.cloud is None:
            return
        if self.session.semantic_map.get("stem") is None:
            QMessageBox.information(self, self.tr("提示"), self.tr("请先选择茎语义标签。"))
            return
        if not self.on_stem_length_params_dialog():
            return
        dlg = None
        try:
            dlg = self._start_busy_dialog(self.tr("正在计算茎长..."))
            count = self.session.compute_stem_length_structures()
        except Exception as e:
            QMessageBox.critical(self, self.tr("计算失败"), self.tr(str(e)))
            return
        finally:
            self._finish_busy_dialog(dlg)
        if hasattr(self, "btn_toggle_stem_path") and self.btn_toggle_stem_path.isChecked():
            self._update_stem_length_paths()
            self.plotter.render()
        self._update_phenotype_table()
        msg = self.tr("已计算：茎长 {count} 个。", count=count)
        QMessageBox.information(self, self.tr("完成"), msg)
        self._update_status(msg)


    def on_stem_diameter_params_dialog(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(self.tr("茎粗去噪参数"))
        layout = QtWidgets.QVBoxLayout(dlg)

        form = QtWidgets.QFormLayout()
        spin_knn_k = QtWidgets.QSpinBox()
        spin_knn_k.setRange(2, 64)
        spin_knn_k.setValue(int(getattr(self.session.params, "stem_denoise_knn_k", 8)))

        spin_knn_sigma = QtWidgets.QDoubleSpinBox()
        spin_knn_sigma.setRange(0.5, 10.0)
        spin_knn_sigma.setDecimals(2)
        spin_knn_sigma.setSingleStep(0.1)
        spin_knn_sigma.setValue(float(getattr(self.session.params, "stem_denoise_knn_sigma", 2.5)))

        spin_radial_sigma = QtWidgets.QDoubleSpinBox()
        spin_radial_sigma.setRange(0.5, 10.0)
        spin_radial_sigma.setDecimals(2)
        spin_radial_sigma.setSingleStep(0.1)
        spin_radial_sigma.setValue(float(getattr(self.session.params, "stem_denoise_radial_sigma", 2.5)))

        form.addRow(self.tr("邻域点数 k"), spin_knn_k)
        form.addRow(self.tr("邻域距离阈值系数"), spin_knn_sigma)
        form.addRow(self.tr("径向阈值系数"), spin_radial_sigma)
        layout.addLayout(form)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return False

        self.session.params.stem_denoise_knn_k = int(spin_knn_k.value())
        self.session.params.stem_denoise_knn_sigma = float(spin_knn_sigma.value())
        self.session.params.stem_denoise_radial_sigma = float(spin_radial_sigma.value())
        return True


    def on_stem_length_params_dialog(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(self.tr("茎长参数"))
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

        form.addRow(self.tr("分段数"), spin_segments)
        form.addRow(self.tr("半径百分位(%)"), spin_pct)
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
            QMessageBox.information(self, self.tr("提示"), self.tr("请先选择花/果语义标签。"))
            return
        dlg = None
        try:
            dlg = self._start_busy_dialog(self.tr("正在计算花/果 OBB..."))
            counts = self.session.compute_flower_fruit_obb()
        except Exception as e:
            QMessageBox.critical(self, self.tr("计算失败"), self.tr(str(e)))
            return
        finally:
            self._finish_busy_dialog(dlg)
        self._update_phenotype_table()
        msg = self.tr("已计算：花 {flower} 个，果 {fruit} 个。", flower=counts["flower"], fruit=counts["fruit"])
        QMessageBox.information(self, self.tr("完成"), msg)
        self._update_status(msg)


    def on_growth_manual(self):
        if self.session.cloud is None:
            return
        if self.annotating:
            QMessageBox.information(self, self.tr("提示"), self.tr("请先返回浏览模式再选择最低点。"))
            return
        if self.pick_mode != self.MODE_NONE:
            self._exit_current_mode(commit=True)
        pts = self.session.get_full_xyz()
        if pts is None or len(pts) == 0:
            return
        min_idx = int(np.argmin(pts[:, 2]))
        origin = pts[min_idx]
        self.session.set_growth_direction(origin=origin, direction=[0.0, 0.0, 1.0], method="manual")
        if hasattr(self, "btn_toggle_growth_dir"):
            self.btn_toggle_growth_dir.blockSignals(True)
            self.btn_toggle_growth_dir.setChecked(True)
            self.btn_toggle_growth_dir.blockSignals(False)
        self._update_growth_direction_display()
        self._update_buttons()
        self.pick_mode = self.MODE_GROWTH_BASE
        self.plotter.render()
        self._update_status(self.tr("生长方向：已默认最低点，可 Shift+左键重新选择最低点；再选择旋转轴进行对齐。"))


    def on_growth_stem(self):
        if self.session.cloud is None:
            return
        try:
            res = self.session.compute_growth_direction_from_stem()
        except Exception as e:
            QMessageBox.critical(self, self.tr("计算失败"), self.tr(str(e)))
            return
        if res is None:
            QMessageBox.information(self, self.tr("提示"), self.tr("未能从茎点云计算生长方向，请检查茎语义。"))
            return
        if hasattr(self, "btn_toggle_growth_dir"):
            self.btn_toggle_growth_dir.blockSignals(True)
            self.btn_toggle_growth_dir.setChecked(True)
            self.btn_toggle_growth_dir.blockSignals(False)
        self._update_growth_direction_display()
        self._update_plant_measurement_display()
        self._update_phenotype_table()
        self._update_buttons()
        self.plotter.render()
        self._update_status(self.tr("已根据茎下半部分计算生长方向。"))


    def on_measure_plant(self):
        if self.session.cloud is None:
            return
        try:
            result = self.session.compute_plant_height_crown()
        except Exception as e:
            QMessageBox.critical(self, self.tr("测量失败"), self.tr(str(e)))
            return
        if result is None:
            QMessageBox.information(self, self.tr("提示"), self.tr("无法测量株高/冠幅，请检查点云。"))
            return
        height, crown = result
        if hasattr(self, "btn_toggle_growth_dir"):
            self.btn_toggle_growth_dir.blockSignals(True)
            self.btn_toggle_growth_dir.setChecked(True)
            self.btn_toggle_growth_dir.blockSignals(False)
        if hasattr(self, "btn_toggle_plant_height"):
            self.btn_toggle_plant_height.blockSignals(True)
            self.btn_toggle_plant_height.setChecked(True)
            self.btn_toggle_plant_height.blockSignals(False)
        if hasattr(self, "btn_toggle_plant_crown"):
            self.btn_toggle_plant_crown.blockSignals(True)
            self.btn_toggle_plant_crown.setChecked(True)
            self.btn_toggle_plant_crown.blockSignals(False)
        self._update_growth_direction_display()
        self._update_plant_measurement_display()
        self._update_phenotype_table()
        self._update_buttons()
        self.plotter.render()
        self._update_status(self.tr("已测量：株高={height:.3f}，冠幅={crown:.3f}", height=height, crown=crown))


    def on_save_annotations(self):
        has_meta = hasattr(self.session, "instance_meta") and len(self.session.instance_meta) > 0
        has_plant_data = hasattr(self.session, "has_plant_data") and self.session.has_plant_data()
        if self.session.get_annotations_count() == 0 and not has_meta and not has_plant_data:
            QMessageBox.information(self, self.tr("提示"), self.tr("当前文件还没有任何实例被标注。"))
            return
        if self.session.centerline_result is not None and self.session.width_path_points is not None:
            dlg = None
            try:
                dlg = self._start_busy_dialog(self.tr("正在生成标注标签..."))
                self.session.point_labels = self.session.compute_point_labels(self.session.params.label_radius)
            except Exception as e:
                QMessageBox.critical(self, self.tr("生成标签失败"), self.tr(str(e)))
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
            export_dir = QFileDialog.getExistingDirectory(self, self.tr("选择标注导出目录"), self._settings.value("last_dir", "", type=str))
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
            QMessageBox.critical(self, self.tr("导出失败"), self.tr(str(e)))
            return

        QMessageBox.information(
            self,
            self.tr("完成"),
            self.tr("已保存标注：{path}\n标签已包含在 JSON 中。", path=json_path),
        )


    def on_show_help(self):
        text = (
            "<div style='font-size:12pt; line-height:1.5;'>"
            f"<b>{self.tr('基础流程')}</b><br>"
            f"{self.tr('读取点云 → 选择语义标签 → 选择实例 → 点击标注进入')}<br><br>"
            f"<b>{self.tr('叶子标注')}</b><br>"
            f"{self.tr('Shift+左键拾取 叶基/叶尖/叶宽端点/控制点')}<br>"
            f"{self.tr('可用功能：生成叶长/叶宽/面积，推荐叶宽，平滑曲线，计算叶倾角/叶夹角')}<br><br>"
            f"<b>{self.tr('茎/花/果')}</b><br>"
            f"{self.tr('茎：计算茎粗/茎长，可显示圆柱与茎长路径')}<br>"
            f"{self.tr('花/果：计算 OBB，并显示边长')}<br><br>"
            f"<b>{self.tr('视图与显示')}</b><br>"
            f"{self.tr('视图：正/侧/顶视图；显示 AABB/OBB；选择视图中心会显示黑色标记点')}<br>"
            f"{self.tr('显示模式：RGB / 语义 / 实例 / 表型标签 视图切换')}<br><br>"
            f"<b>{self.tr('导出与快捷键')}</b><br>"
            f"{self.tr('保存标注：Ctrl + S；导出表型 CSV；导出标注点云')}<br>"
            f"{self.tr('提示：先开启对应拾取模式再点击点云')}"
            "</div>"
        )
        dlg = QMessageBox(self)
        dlg.setWindowTitle(self.tr("使用说明"))
        dlg.setIcon(QMessageBox.Information)
        dlg.setTextFormat(Qt.RichText)
        dlg.setText(text)
        dlg.exec_()


    def on_show_about(self):
        QMessageBox.information(
            self,
            self.tr("关于"),
            self.tr(
                "版本：v0.1.1\n"
                "开发人：杨鑫,苗腾\n"
                "邮件：yangxinnc@163.com\n"
                "单位：沈阳农业大学 信息与电气工程学院"
            ),
        )
