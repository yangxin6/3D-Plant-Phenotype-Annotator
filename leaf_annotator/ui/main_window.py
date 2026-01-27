# ui/main_window.py
import os
from typing import Optional
import numpy as np
import pyvista as pv
import vtk
from pyvistaqt import QtInteractor

from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QListWidgetItem

from core.annotation import LeafAnnotationSession, AnnotationParams
from core.schema import CloudSchema
from i18n import I18n

from .main_window_parts.constants import (
    PANEL_WIDTH_DEFAULT,
    PANEL_WIDTH_LEAF_ANNOTATE,
    PANEL_INNER_PADDING,
    PANEL_ROW_LABEL_SPACE,
    PANEL_COMBO_MIN_WIDTH,
    PANEL_LIST_MIN_WIDTH,
    PANEL_BUTTON_MIN_WIDTH,
    TEXT_REMARK_LINES,
    TEXT_REMARK_PADDING,
    LIST_BASE_MAX_HEIGHT,
    LIST_TIP_MAX_HEIGHT,
    LIST_WIDTH_MAX_HEIGHT,
    LIST_CTRL_MAX_HEIGHT,
    LIST_WIDTH_CTRL_MAX_HEIGHT,
    VIEW_LEGEND_HEIGHT,
    LABEL_SWATCH_SIZE,
)
from .main_window_parts.utils import make_polyline_mesh, stable_color_from_id, colors_from_labels
from .main_window_parts.ui_state import UiStateMixin
from .main_window_parts.semantics import SemanticMixin
from .main_window_parts.scene import SceneMixin
from .main_window_parts.annotation import AnnotationMixin
from .main_window_parts.interaction import InteractionMixin
from .main_window_parts.view_controls import ViewControlsMixin
from .main_window_parts.actions import ActionsMixin


class LeafAnnotatorWindow(QtWidgets.QMainWindow, UiStateMixin, SemanticMixin, SceneMixin, AnnotationMixin, InteractionMixin, ViewControlsMixin, ActionsMixin):
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
    MODE_WIDTH_CTRL = "PICK_WIDTH_CTRL"
    MODE_MEASURE = "MEASURE"
    MODE_GROWTH_BASE = "PICK_GROWTH_BASE"

    VIEW_RGB = "RGB"
    VIEW_SEM = "SEM"
    VIEW_INST = "INST"
    VIEW_LABEL = "PHENO"

    # actor names
    A_CLOUD_FULL = "cloud_full"
    A_CLOUD_INST = "cloud_inst"

    A_BASE_SAVED = "mark_base_saved"
    A_TIP_SAVED = "mark_tip_saved"
    A_CTRL_SAVED = "mark_ctrl_saved"
    A_WIDTH_SAVED = "mark_width_saved"
    A_WIDTH_CTRL_SAVED = "mark_width_ctrl_saved"

    A_BASE_TEMP = "mark_base_temp"
    A_TIP_TEMP = "mark_tip_temp"
    A_CTRL_TEMP = "mark_ctrl_temp"
    A_WIDTH_TEMP = "mark_width_temp"
    A_WIDTH_CTRL_TEMP = "mark_width_ctrl_temp"

    A_LABELS_SAVED = "labels_saved"
    A_LABELS_TEMP = "labels_temp"

    A_LINE_CENTER_CUR = "line_center_current"
    A_LINE_WIDTH_CUR = "line_width_current"
    A_LINE_CENTER_CACHED = "line_center_cached"
    A_LINE_WIDTH_CACHED = "line_width_cached"
    A_MEASURE_LINE = "measure_line"
    A_MEASURE_P1 = "measure_p1"
    A_MEASURE_P2 = "measure_p2"
    A_AABB = "aabb_box"
    A_OBB = "obb_box"
    A_VIEW_CENTER = "view_center"
    A_STEM_LENGTH = "stem_length_path"
    A_OBB_DIM_L = "obb_dim_l"
    A_OBB_DIM_W = "obb_dim_w"
    A_OBB_DIM_H = "obb_dim_h"
    A_OBB_DIM_LABELS = "obb_dim_labels"
    A_LEAF_INCLINE_N = "leaf_incline_normal"
    A_LEAF_INCLINE_Z = "leaf_incline_z"
    A_LEAF_INCLINE_ARC = "leaf_incline_arc"
    A_LEAF_INCLINE_LABEL = "leaf_incline_label"
    A_LEAF_STEM_LEAF = "leaf_stem_leaf"
    A_LEAF_STEM_STEM = "leaf_stem_stem"
    A_LEAF_STEM_ARC = "leaf_stem_arc"
    A_LEAF_STEM_LABEL = "leaf_stem_label"
    A_GROWTH_DIR = "growth_direction"
    A_GROWTH_DIR_LABEL = "growth_direction_label"
    A_PLANT_HEIGHT = "plant_height_line"
    A_PLANT_HEIGHT_LABEL = "plant_height_label"
    A_PLANT_CROWN = "plant_crown_line"
    A_PLANT_CROWN_LABEL = "plant_crown_label"

    def __init__(self):
        super().__init__()
        self._settings = QtCore.QSettings("LeafAnnotator", "3d_plant_phenotype")
        self._i18n = I18n.instance()
        lang = self._settings.value("lang", "zh", type=str)
        if lang not in ["zh", "en"]:
            lang = "zh"
        self._i18n.set_language(lang)
        self.tr = self._i18n.tr
        self.setWindowTitle(self.tr("植物表型标注工具"))

        menu = self.menuBar()
        self.menu_file = menu.addMenu(self.tr("文件"))
        self.act_load = self.menu_file.addAction(self.tr("加载点云"))
        self.act_export_dir = self.menu_file.addAction(self.tr("导出目录"))
        self.act_export_phenotype = self.menu_file.addAction(self.tr("导出表型CSV"))
        self.menu_file.addSeparator()
        self.act_quit = self.menu_file.addAction(self.tr("退出"))

        self.menu_view = menu.addMenu(self.tr("显示"))
        self.act_view_rgb = self.menu_view.addAction(self.tr("RGB"))
        self.act_view_sem = self.menu_view.addAction(self.tr("语义"))
        self.act_view_inst = self.menu_view.addAction(self.tr("实例"))
        self.act_view_label = self.menu_view.addAction(self.tr("表型标签"))
        for a in [self.act_view_rgb, self.act_view_sem, self.act_view_inst, self.act_view_label]:
            a.setCheckable(True)
        self.view_group = QtWidgets.QActionGroup(self)
        self.view_group.setExclusive(True)
        self.view_group.addAction(self.act_view_rgb)
        self.view_group.addAction(self.act_view_sem)
        self.view_group.addAction(self.act_view_inst)
        self.view_group.addAction(self.act_view_label)
        self.act_view_rgb.setChecked(True)

        self.menu_anno = menu.addMenu(self.tr("标注"))
        self.act_start_anno = self.menu_anno.addAction(self.tr("进入标注模式"))
        self.act_back_browse = self.menu_anno.addAction(self.tr("返回浏览模式"))

        self.menu_calc = menu.addMenu(self.tr("计算"))
        self.act_recommend_length = self.menu_calc.addAction(self.tr("推荐叶长"))
        self.act_generate_length = self.menu_calc.addAction(self.tr("生成叶长"))
        self.act_recommend_width = self.menu_calc.addAction(self.tr("推荐叶宽"))
        self.act_generate_width = self.menu_calc.addAction(self.tr("生成叶宽"))
        self.act_smooth_paths = self.menu_calc.addAction(self.tr("平滑叶长/叶宽"))
        self.act_compute_leaf_area = self.menu_calc.addAction(self.tr("计算叶面积"))
        self.act_compute_leaf_projected_area = self.menu_calc.addAction(self.tr("计算投影面积"))
        self.act_compute_leaf_inclination = self.menu_calc.addAction(self.tr("计算叶倾角"))
        self.act_compute_leaf_stem_angle = self.menu_calc.addAction(self.tr("计算叶夹角"))
        self.menu_calc.addSeparator()
        self.act_export_labeled = self.menu_calc.addAction(self.tr("生成标记点云"))
        self.act_save_labels = menu.addAction(self.tr("保存标注"))
        self.act_save_labels.setShortcut("Ctrl+S")
        self.act_compute_stem = self.menu_calc.addAction(self.tr("计算茎粗"))
        self.act_compute_stem_length = self.menu_calc.addAction(self.tr("计算茎长"))
        self.act_compute_flower_fruit = self.menu_calc.addAction(self.tr("计算花果OBB"))
        self.menu_calc.addSeparator()
        self.act_growth_manual = self.menu_calc.addAction(self.tr("生长方向：手动最低点"))
        self.act_growth_stem = self.menu_calc.addAction(self.tr("生长方向：茎下半部"))
        self.act_measure_plant = self.menu_calc.addAction(self.tr("测量株高/冠幅"))

        self.menu_plant = menu.addMenu(self.tr("植物类型"))
        self.plant_types = ["corn", "strawberry", "rice", "wheat"]
        self.plant_type_group = QtWidgets.QActionGroup(self)
        self.plant_type_group.setExclusive(True)
        self.plant_type_actions = {}
        for t in self.plant_types:
            act = self.menu_plant.addAction(self._plant_type_display(t))
            act.setCheckable(True)
            act.setData(t)
            self.plant_type_group.addAction(act)
            self.plant_type_actions[t] = act

        self.act_help = menu.addAction(self.tr("帮助"))
        self.act_about = menu.addAction(self.tr("关于"))

        self.menu_lang = menu.addMenu(self.tr("语言"))
        self.act_lang_zh = self.menu_lang.addAction(self.tr("中文"))
        self.act_lang_en = self.menu_lang.addAction(self.tr("英文"))
        for act in [self.act_lang_zh, self.act_lang_en]:
            act.setCheckable(True)
        self.act_lang_zh.setData("zh")
        self.act_lang_en.setData("en")
        self.lang_group = QtWidgets.QActionGroup(self)
        self.lang_group.setExclusive(True)
        self.lang_group.addAction(self.act_lang_zh)
        self.lang_group.addAction(self.act_lang_en)
        if lang == "en":
            self.act_lang_en.setChecked(True)
        else:
            self.act_lang_zh.setChecked(True)

        schema = CloudSchema(xyz_slice=slice(0, 3), sem_col=-2, inst_col=-1, rgb_slice=slice(3, 6))
        self.session = LeafAnnotationSession(params=AnnotationParams(), schema=schema)
        self.annotate_semantic = "leaf"

        self.annotating = False
        self.pick_mode = self.MODE_NONE

        # 临时点（开启期间）
        self.temp_base_idx = None
        self.temp_tip_idx = None
        self.temp_ctrl_indices = []
        self.temp_width_ctrl_indices = []

        # 叶宽两点：临时（开启期间）
        self.temp_w1_idx = None
        self.temp_w2_idx = None
        self.temp_measure_p1 = None
        self.temp_measure_p2 = None

        # VTK picker
        self._vtk_picker = vtk.vtkPointPicker()
        self._vtk_picker.SetTolerance(0.02)
        self._pick_observer_id = None
        self._pick_view_center = False
        self._view_center_point = None
        self._stem_cylinder_actors = []
        self._stem_length_actors = []
        self._rotation_active = False

        # 记录当前实例点云 actor（picker 只拾取它）
        self._actor_cloud_inst = None
        self._actor_cloud_full = None

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QHBoxLayout(root)

        # ---------- left panel (scrollable) ----------
        panel_widget = QtWidgets.QWidget()
        self.panel_widget = panel_widget
        panel = QtWidgets.QVBoxLayout(panel_widget)
        panel.setContentsMargins(0, 0, 0, 0)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setWidget(panel_widget)

        layout.addWidget(scroll, 0)
        self.panel_widget.setFixedWidth(PANEL_WIDTH_DEFAULT)

        # ---------- hidden controls (menu driven) ----------
        self.btn_load = QtWidgets.QPushButton(self.tr("加载点云"))
        self.combo_view = QtWidgets.QComboBox()
        self._set_view_combo_items()
        self.combo_inst = QtWidgets.QComboBox()
        self.btn_start_anno = QtWidgets.QPushButton(self.tr("进入标注模式（仅显示当前实例）"))
        self.btn_back_browse = QtWidgets.QPushButton(self.tr("返回浏览模式（显示整株）"))

        self.btn_toggle_base = QtWidgets.QPushButton(self.tr("叶基选择：关闭"))
        self.btn_toggle_tip = QtWidgets.QPushButton(self.tr("叶尖选择：关闭"))
        self.btn_toggle_ctrl = QtWidgets.QPushButton(self.tr("叶长控制点选择：关闭"))
        self.btn_toggle_width = QtWidgets.QPushButton(self.tr("叶宽端点选择：关闭 (W1/W2)"))
        self.btn_toggle_width_ctrl = QtWidgets.QPushButton(self.tr("叶宽控制点选择：关闭"))
        self.btn_toggle_measure = QtWidgets.QPushButton(self.tr("测距：关闭"))
        for b in [
            self.btn_toggle_base, self.btn_toggle_tip, self.btn_toggle_ctrl,
            self.btn_toggle_width, self.btn_toggle_width_ctrl, self.btn_toggle_measure
        ]:
            b.setCheckable(True)

        self.btn_recommend_length = QtWidgets.QPushButton(self.tr("推荐叶长（B1→T1）"))
        self.btn_generate_length = QtWidgets.QPushButton(self.tr("生成叶长（B1 + 控制点 + T1）"))
        self.btn_recommend_width = QtWidgets.QPushButton(self.tr("推荐叶宽（沿当前叶长）"))
        self.btn_generate_width = QtWidgets.QPushButton(self.tr("生成叶宽（W1→W2）"))
        self.btn_compute_leaf_area = QtWidgets.QPushButton(self.tr("计算叶面积"))
        self.btn_compute_leaf_projected_area = QtWidgets.QPushButton(self.tr("计算投影面积"))
        self.btn_compute_leaf_inclination = QtWidgets.QPushButton(self.tr("计算叶倾角"))
        self.btn_compute_leaf_stem_angle = QtWidgets.QPushButton(self.tr("计算叶夹角"))

        self.btn_delete = QtWidgets.QPushButton(self.tr("删除选中标记"))
        self.btn_rename_ctrl = QtWidgets.QPushButton(self.tr("重命名叶长控制点顺序 (C#)"))
        self.btn_rename_width_ctrl = QtWidgets.QPushButton(self.tr("重命名叶宽控制点顺序 (WC#)"))

        self.btn_export = QtWidgets.QPushButton(self.tr("保存标注（JSON+Label）"))

        # ---------- status ----------
        self.info = QtWidgets.QLabel(self.tr("状态：未加载"))
        self.info.setWordWrap(True)
        status_group = QtWidgets.QGroupBox(self.tr("状态信息"))
        self.status_group = status_group
        status_layout = QtWidgets.QVBoxLayout(status_group)
        status_layout.addWidget(self.info)
        panel.addWidget(status_group)

        # ---------- semantic labels ----------
        sem_group = QtWidgets.QGroupBox(self.tr("语义标签"))
        self.sem_group = sem_group
        sem_layout = QtWidgets.QVBoxLayout(sem_group)
        self.combo_anno_semantic = QtWidgets.QComboBox()
        self._set_anno_semantic_items()

        sem_row = QtWidgets.QHBoxLayout()
        self.lbl_sem_leaf = QtWidgets.QLabel(self.tr("叶语义："))
        sem_row.addWidget(self.lbl_sem_leaf)
        self.combo_sem_filter = QtWidgets.QComboBox()
        sem_row.addWidget(self.combo_sem_filter)
        sem_layout.addLayout(sem_row)

        stem_row = QtWidgets.QHBoxLayout()
        self.lbl_sem_stem = QtWidgets.QLabel(self.tr("茎语义："))
        stem_row.addWidget(self.lbl_sem_stem)
        self.combo_stem_filter = QtWidgets.QComboBox()
        stem_row.addWidget(self.combo_stem_filter)
        sem_layout.addLayout(stem_row)

        flower_row = QtWidgets.QHBoxLayout()
        self.lbl_sem_flower = QtWidgets.QLabel(self.tr("花语义："))
        flower_row.addWidget(self.lbl_sem_flower)
        self.combo_flower_filter = QtWidgets.QComboBox()
        flower_row.addWidget(self.combo_flower_filter)
        sem_layout.addLayout(flower_row)

        fruit_row = QtWidgets.QHBoxLayout()
        self.lbl_sem_fruit = QtWidgets.QLabel(self.tr("果语义："))
        fruit_row.addWidget(self.lbl_sem_fruit)
        self.combo_fruit_filter = QtWidgets.QComboBox()
        fruit_row.addWidget(self.combo_fruit_filter)
        sem_layout.addLayout(fruit_row)

        self.lbl_sem_note = QtWidgets.QLabel(self.tr("注：-1 表示没有该语义"))
        sem_layout.addWidget(self.lbl_sem_note)
        panel.addWidget(sem_group)
        for combo in [self.combo_sem_filter, self.combo_stem_filter, self.combo_flower_filter, self.combo_fruit_filter]:
            combo.addItem("-1")
            combo.setCurrentText("-1")

        # ---------- instance selector ----------
        inst_group = QtWidgets.QGroupBox(self.tr("实例"))
        self.inst_group = inst_group
        inst_layout = QtWidgets.QVBoxLayout(inst_group)
        anno_row = QtWidgets.QHBoxLayout()
        self.lbl_anno_sem = QtWidgets.QLabel(self.tr("标注语义："))
        anno_row.addWidget(self.lbl_anno_sem)
        anno_row.addWidget(self.combo_anno_semantic)
        inst_layout.addLayout(anno_row)
        inst_row = QtWidgets.QHBoxLayout()
        self.lbl_inst_label = QtWidgets.QLabel(self.tr("实例标签："))
        inst_row.addWidget(self.lbl_inst_label)
        inst_row.addWidget(self.combo_inst)
        inst_layout.addLayout(inst_row)
        self.lbl_inst_sem = QtWidgets.QLabel(self.tr("语义标签：-"))
        self.lbl_inst_sem.setWordWrap(True)
        inst_layout.addWidget(self.lbl_inst_sem)
        panel.addWidget(inst_group)

        # ---------- bbox info ----------
        bbox_group = QtWidgets.QGroupBox(self.tr("包围盒"))
        self.bbox_group = bbox_group
        bbox_layout = QtWidgets.QVBoxLayout(bbox_group)
        self.bbox_info = QtWidgets.QLabel(f"{self.tr('AABB：-')}\n{self.tr('OBB：-')}")
        self.bbox_info.setWordWrap(True)
        bbox_layout.addWidget(self.bbox_info)
        panel.addWidget(bbox_group)

        # ---------- instance meta ----------
        self.meta_group = QtWidgets.QGroupBox(self.tr("实例备注"))
        meta_layout = QtWidgets.QVBoxLayout(self.meta_group)
        self.lbl_meta_info = QtWidgets.QLabel(self.tr("附加信息"))
        meta_layout.addWidget(self.lbl_meta_info)
        self.combo_label_desc = QtWidgets.QComboBox()
        self._set_label_desc_items()
        meta_layout.addWidget(self.combo_label_desc)
        self.lbl_meta_remark = QtWidgets.QLabel(self.tr("实例备注"))
        meta_layout.addWidget(self.lbl_meta_remark)
        self.text_remark = QtWidgets.QTextEdit()
        self.text_remark.setFixedHeight(
            self.text_remark.fontMetrics().lineSpacing() * TEXT_REMARK_LINES + TEXT_REMARK_PADDING
        )
        meta_layout.addWidget(self.text_remark)
        panel.addWidget(self.meta_group)
        self.meta_group.setEnabled(False)

        # ---------- marker lists ----------
        self.points_group = QtWidgets.QGroupBox(self.tr("标记信息"))
        points_layout = QtWidgets.QVBoxLayout(self.points_group)
        self.list_base = QtWidgets.QListWidget()
        self.list_tip = QtWidgets.QListWidget()
        self.list_ctrl = QtWidgets.QListWidget()
        self.list_width = QtWidgets.QListWidget()
        self.list_width_ctrl = QtWidgets.QListWidget()
        self.list_base.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list_tip.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list_ctrl.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list_width.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list_width_ctrl.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list_base.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.list_tip.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.list_width.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.list_ctrl.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.list_width_ctrl.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.list_base.setMaximumHeight(LIST_BASE_MAX_HEIGHT)
        self.list_tip.setMaximumHeight(LIST_TIP_MAX_HEIGHT)
        self.list_width.setMaximumHeight(LIST_WIDTH_MAX_HEIGHT)
        self.list_ctrl.setMaximumHeight(LIST_CTRL_MAX_HEIGHT)
        self.list_width_ctrl.setMaximumHeight(LIST_WIDTH_CTRL_MAX_HEIGHT)
        self.lbl_points_base = QtWidgets.QLabel(self.tr("叶基 (B1)"))
        points_layout.addWidget(self.lbl_points_base)
        points_layout.addWidget(self.list_base)
        points_layout.addWidget(self.btn_toggle_base)
        self.lbl_points_tip = QtWidgets.QLabel(self.tr("叶尖 (T1)"))
        points_layout.addWidget(self.lbl_points_tip)
        points_layout.addWidget(self.list_tip)
        points_layout.addWidget(self.btn_toggle_tip)
        self.lbl_points_ctrl = QtWidgets.QLabel(self.tr("叶长控制点 (C1..Cn)"))
        points_layout.addWidget(self.lbl_points_ctrl)
        points_layout.addWidget(self.list_ctrl)
        points_layout.addWidget(self.btn_toggle_ctrl)
        self.lbl_points_width = QtWidgets.QLabel(self.tr("叶宽端点 (W1/W2)"))
        points_layout.addWidget(self.lbl_points_width)
        points_layout.addWidget(self.list_width)
        points_layout.addWidget(self.btn_toggle_width)
        self.lbl_points_width_ctrl = QtWidgets.QLabel(self.tr("叶宽控制点 (WC1..WCn)"))
        points_layout.addWidget(self.lbl_points_width_ctrl)
        points_layout.addWidget(self.list_width_ctrl)
        points_layout.addWidget(self.btn_toggle_width_ctrl)
        panel.addWidget(self.points_group)

        # ---------- leaf area ----------
        self.leaf_area_group = QtWidgets.QGroupBox(self.tr("叶面积"))
        leaf_area_layout = QtWidgets.QVBoxLayout(self.leaf_area_group)
        leaf_area_layout.addWidget(self.btn_compute_leaf_area)
        leaf_area_layout.addWidget(self.btn_compute_leaf_projected_area)
        panel.addWidget(self.leaf_area_group)

        # ---------- leaf angles ----------
        self.leaf_angle_group = QtWidgets.QGroupBox(self.tr("叶角度"))
        leaf_angle_layout = QtWidgets.QVBoxLayout(self.leaf_angle_group)
        leaf_angle_layout.addWidget(self.btn_compute_leaf_inclination)
        leaf_angle_layout.addWidget(self.btn_compute_leaf_stem_angle)
        panel.addWidget(self.leaf_angle_group)

        # ---------- phenotype table ----------
        self.phenotype_group = QtWidgets.QGroupBox(self.tr("表型信息"))
        phenotype_layout = QtWidgets.QVBoxLayout(self.phenotype_group)
        self.table_phenotype = QtWidgets.QTableWidget(0, 4)
        self.table_phenotype.setHorizontalHeaderLabels([
            self.tr("实例"), self.tr("语义"), self.tr("名字"), self.tr("值")
        ])
        header = self.table_phenotype.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        self.table_phenotype.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table_phenotype.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.table_phenotype.setAlternatingRowColors(True)
        self.table_phenotype.setWordWrap(False)
        self.table_phenotype.setTextElideMode(QtCore.Qt.ElideRight)
        phenotype_layout.addWidget(self.table_phenotype)
        self.btn_export_phenotype = QtWidgets.QPushButton(self.tr("导出表型CSV"))
        phenotype_layout.addWidget(self.btn_export_phenotype)
        panel.addWidget(self.phenotype_group)

        panel.addStretch(1)

        # ---------- right: PyVista ----------
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor, 1)
        self.plotter.set_background("white")
        self.plotter.show_axes()

        # ---------- right panel (view tools) ----------
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        view_tools_group = QtWidgets.QGroupBox(self.tr("视图操作"))
        self.view_tools_group = view_tools_group
        view_tools_layout = QtWidgets.QVBoxLayout(view_tools_group)
        self.btn_view_front = QtWidgets.QPushButton(self.tr("正视图"))
        self.btn_view_side = QtWidgets.QPushButton(self.tr("侧视图"))
        self.btn_view_top = QtWidgets.QPushButton(self.tr("顶视图"))
        self.btn_pick_view_center = QtWidgets.QPushButton(self.tr("选择视图中心"))
        self.btn_toggle_aabb = QtWidgets.QPushButton(self.tr("显示AABB盒"))
        self.btn_toggle_obb = QtWidgets.QPushButton(self.tr("显示OBB盒"))
        self.btn_toggle_stem_cyl = QtWidgets.QPushButton(self.tr("显示茎圆柱"))
        self.btn_toggle_stem_path = QtWidgets.QPushButton(self.tr("显示茎长路径"))
        self.btn_toggle_aabb.setCheckable(True)
        self.btn_toggle_obb.setCheckable(True)
        self.btn_toggle_stem_cyl.setCheckable(True)
        self.btn_toggle_stem_path.setCheckable(True)
        view_tools_layout.addWidget(self.btn_view_front)
        view_tools_layout.addWidget(self.btn_view_side)
        view_tools_layout.addWidget(self.btn_view_top)
        view_tools_layout.addWidget(self.btn_pick_view_center)
        view_tools_layout.addWidget(self.btn_toggle_aabb)
        view_tools_layout.addWidget(self.btn_toggle_obb)
        view_tools_layout.addWidget(self.btn_toggle_stem_cyl)
        view_tools_layout.addWidget(self.btn_toggle_stem_path)

        measure_group = QtWidgets.QGroupBox(self.tr("测量"))
        self.measure_group = measure_group
        measure_layout = QtWidgets.QVBoxLayout(measure_group)
        measure_layout.addWidget(self.btn_toggle_measure)

        plant_measure_group = QtWidgets.QGroupBox(self.tr("整株测量"))
        self.plant_measure_group = plant_measure_group
        plant_measure_layout = QtWidgets.QVBoxLayout(plant_measure_group)
        self.btn_toggle_growth_dir = QtWidgets.QPushButton(self.tr("显示生长方向"))
        self.btn_toggle_plant_height = QtWidgets.QPushButton(self.tr("显示株高"))
        self.btn_toggle_plant_crown = QtWidgets.QPushButton(self.tr("显示冠幅"))
        self.btn_toggle_growth_dir.setCheckable(True)
        self.btn_toggle_plant_height.setCheckable(True)
        self.btn_toggle_plant_crown.setCheckable(True)
        plant_measure_layout.addWidget(self.btn_toggle_growth_dir)
        plant_measure_layout.addWidget(self.btn_toggle_plant_height)
        plant_measure_layout.addWidget(self.btn_toggle_plant_crown)
        self.lbl_rotate_axis = QtWidgets.QLabel(self.tr("旋转轴"))
        plant_measure_layout.addWidget(self.lbl_rotate_axis)
        self.combo_rotate_axis = QtWidgets.QComboBox()
        self.combo_rotate_axis.addItems(["X", "Y", "Z"])
        plant_measure_layout.addWidget(self.combo_rotate_axis)
        rotate_step_row = QtWidgets.QHBoxLayout()
        self.lbl_rotate_step = QtWidgets.QLabel(self.tr("步长(°)"))
        rotate_step_row.addWidget(self.lbl_rotate_step)
        self.spin_rotate_step = QtWidgets.QDoubleSpinBox()
        self.spin_rotate_step.setRange(0.1, 180.0)
        self.spin_rotate_step.setDecimals(1)
        self.spin_rotate_step.setSingleStep(1.0)
        self.spin_rotate_step.setValue(5.0)
        rotate_step_row.addWidget(self.spin_rotate_step)
        plant_measure_layout.addLayout(rotate_step_row)
        rotate_btn_row = QtWidgets.QHBoxLayout()
        self.btn_rotate_minus = QtWidgets.QPushButton(self.tr("旋转 -"))
        self.btn_rotate_plus = QtWidgets.QPushButton(self.tr("旋转 +"))
        rotate_btn_row.addWidget(self.btn_rotate_minus)
        rotate_btn_row.addWidget(self.btn_rotate_plus)
        plant_measure_layout.addLayout(rotate_btn_row)
        self.btn_rotate_start = QtWidgets.QPushButton(self.tr("开始旋转"))
        self.btn_rotate_finish = QtWidgets.QPushButton(self.tr("完成旋转"))
        rotate_state_row = QtWidgets.QHBoxLayout()
        rotate_state_row.addWidget(self.btn_rotate_start)
        rotate_state_row.addWidget(self.btn_rotate_finish)
        plant_measure_layout.addLayout(rotate_state_row)
        self.btn_rotate_reset = QtWidgets.QPushButton(self.tr("重置旋转"))
        plant_measure_layout.addWidget(self.btn_rotate_reset)

        right_layout.addWidget(view_tools_group)
        right_layout.addWidget(measure_group)
        right_layout.addWidget(plant_measure_group)
        right_layout.addStretch(1)

        self.view_legend_group = QtWidgets.QGroupBox(self.tr("当前视图标签"))
        self.view_legend_group.setFixedHeight(VIEW_LEGEND_HEIGHT)
        self.view_legend_scroll = QtWidgets.QScrollArea()
        self.view_legend_scroll.setWidgetResizable(True)
        self.view_legend_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.view_legend_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.view_legend_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.view_legend_widget = QtWidgets.QWidget()
        self.view_legend_layout = QtWidgets.QVBoxLayout(self.view_legend_widget)
        self.view_legend_layout.setContentsMargins(0, 0, 0, 0)
        self.view_legend_scroll.setWidget(self.view_legend_widget)
        view_legend_box = QtWidgets.QVBoxLayout(self.view_legend_group)
        view_legend_box.setContentsMargins(0, 0, 0, 0)
        view_legend_box.addWidget(self.view_legend_scroll)
        right_layout.addWidget(self.view_legend_group)
        layout.addWidget(right_panel, 0)

        anno_default = self._settings.value("anno_semantic", "leaf", type=str)
        legacy_map = {"叶": "leaf", "茎": "stem", "花": "flower", "果": "fruit"}
        if anno_default in legacy_map:
            anno_default = legacy_map[anno_default]
        if anno_default in ["leaf", "stem", "flower", "fruit"]:
            self._set_combo_current_by_data(self.combo_anno_semantic, anno_default)
            self.annotate_semantic = self._get_annotation_semantic_key()
        plant_default = self._settings.value("plant_type", "corn", type=str)
        plant_default = self._normalize_plant_type(plant_default)
        self._set_plant_type(plant_default, update_status=False, save_settings=False)

        # ---------- signals ----------
        self.btn_load.clicked.connect(self.on_load)
        self.combo_view.currentIndexChanged.connect(self.on_view_mode_changed)

        self.btn_start_anno.clicked.connect(self.on_start_annotation)
        self.btn_back_browse.clicked.connect(self.on_back_browse)
        self.combo_inst.currentIndexChanged.connect(self.on_inst_changed)
        self.combo_anno_semantic.currentIndexChanged.connect(self.on_anno_semantic_changed)
        self.combo_sem_filter.currentIndexChanged.connect(self.on_sem_filter_changed)
        self.combo_stem_filter.currentIndexChanged.connect(self.on_stem_sem_changed)
        self.combo_flower_filter.currentIndexChanged.connect(self.on_flower_sem_changed)
        self.combo_fruit_filter.currentIndexChanged.connect(self.on_fruit_sem_changed)

        self.list_ctrl.customContextMenuRequested.connect(self.on_ctrl_context_menu)
        self.list_width_ctrl.customContextMenuRequested.connect(self.on_width_ctrl_context_menu)
        self.list_base.customContextMenuRequested.connect(self.on_base_context_menu)
        self.list_tip.customContextMenuRequested.connect(self.on_tip_context_menu)
        self.list_width.customContextMenuRequested.connect(self.on_width_context_menu)

        self.btn_toggle_base.toggled.connect(self.on_toggle_base)
        self.btn_toggle_tip.toggled.connect(self.on_toggle_tip)
        self.btn_toggle_ctrl.toggled.connect(self.on_toggle_ctrl)
        self.btn_toggle_width.toggled.connect(self.on_toggle_width)
        self.btn_toggle_width_ctrl.toggled.connect(self.on_toggle_width_ctrl)
        self.btn_toggle_measure.toggled.connect(self.on_toggle_measure)

        self.btn_recommend_length.clicked.connect(self.on_recommend_length)
        self.btn_generate_length.clicked.connect(self.on_generate_length)

        self.btn_delete.clicked.connect(self.on_delete_selected)
        self.btn_rename_ctrl.clicked.connect(self.on_rename_ctrl)
        self.btn_rename_width_ctrl.clicked.connect(self.on_rename_width_ctrl)

        self.btn_recommend_width.clicked.connect(self.on_recommend_width)  # ✅
        self.btn_generate_width.clicked.connect(self.on_generate_width)
        self.btn_compute_leaf_area.clicked.connect(self.on_compute_leaf_area)
        self.btn_compute_leaf_projected_area.clicked.connect(self.on_compute_leaf_projected_area)
        self.btn_compute_leaf_inclination.clicked.connect(self.on_compute_leaf_inclination)
        self.btn_compute_leaf_stem_angle.clicked.connect(self.on_compute_leaf_stem_angle)

        self.btn_export.clicked.connect(self.on_save_annotations)
        self.btn_export_phenotype.clicked.connect(self.on_export_phenotype_csv)

        self.text_remark.textChanged.connect(self.on_instance_meta_changed)
        self.combo_label_desc.currentIndexChanged.connect(self.on_instance_meta_changed)
        self.act_load.triggered.connect(self.on_load)
        self.act_export_dir.triggered.connect(self.on_choose_export_dir)
        self.act_export_phenotype.triggered.connect(self.on_export_phenotype_csv)
        self.act_quit.triggered.connect(self.close)
        self.act_help.triggered.connect(self.on_show_help)
        self.act_about.triggered.connect(self.on_show_about)

        self.act_view_rgb.triggered.connect(lambda: self._set_view_mode(self.VIEW_RGB))
        self.act_view_sem.triggered.connect(lambda: self._set_view_mode(self.VIEW_SEM))
        self.act_view_inst.triggered.connect(lambda: self._set_view_mode(self.VIEW_INST))
        self.act_view_label.triggered.connect(lambda: self._set_view_mode(self.VIEW_LABEL))
        self.act_start_anno.triggered.connect(self.on_start_annotation)
        self.act_back_browse.triggered.connect(self.on_back_browse)

        self.act_recommend_length.triggered.connect(self.on_recommend_length)
        self.act_generate_length.triggered.connect(self.on_generate_length)
        self.act_recommend_width.triggered.connect(self.on_recommend_width)
        self.act_generate_width.triggered.connect(self.on_generate_width)
        self.act_compute_leaf_area.triggered.connect(self.on_compute_leaf_area)
        self.act_compute_leaf_projected_area.triggered.connect(self.on_compute_leaf_projected_area)
        self.act_compute_leaf_inclination.triggered.connect(self.on_compute_leaf_inclination)
        self.act_compute_leaf_stem_angle.triggered.connect(self.on_compute_leaf_stem_angle)
        self.act_smooth_paths.triggered.connect(self.on_smooth_leaf_paths)
        self.act_export_labeled.triggered.connect(self.on_export_labeled_cloud)
        self.act_save_labels.triggered.connect(self.on_save_annotations)
        self.act_compute_stem.triggered.connect(self.on_compute_stem)
        self.act_compute_stem_length.triggered.connect(self.on_compute_stem_length)
        self.act_compute_flower_fruit.triggered.connect(self.on_compute_flower_fruit)
        self.act_growth_manual.triggered.connect(self.on_growth_manual)
        self.act_growth_stem.triggered.connect(self.on_growth_stem)
        self.act_measure_plant.triggered.connect(self.on_measure_plant)
        self.btn_view_front.clicked.connect(self.on_view_front)
        self.btn_view_side.clicked.connect(self.on_view_side)
        self.btn_view_top.clicked.connect(self.on_view_top)
        self.btn_pick_view_center.clicked.connect(self.on_pick_view_center)
        self.btn_toggle_aabb.toggled.connect(self.on_toggle_aabb)
        self.btn_toggle_obb.toggled.connect(self.on_toggle_obb)
        self.btn_toggle_stem_cyl.toggled.connect(self.on_toggle_stem_cyl)
        self.btn_toggle_stem_path.toggled.connect(self.on_toggle_stem_path)
        self.btn_toggle_growth_dir.toggled.connect(self.on_toggle_growth_direction)
        self.btn_toggle_plant_height.toggled.connect(self.on_toggle_plant_height)
        self.btn_toggle_plant_crown.toggled.connect(self.on_toggle_plant_crown)
        self.btn_rotate_start.clicked.connect(self.on_start_growth_rotation)
        self.btn_rotate_finish.clicked.connect(self.on_finish_growth_rotation)
        self.btn_rotate_minus.clicked.connect(lambda: self.on_rotate_growth_step(-1))
        self.btn_rotate_plus.clicked.connect(lambda: self.on_rotate_growth_step(1))
        self.btn_rotate_reset.clicked.connect(self.on_reset_growth_rotation)
        for act in self.plant_type_actions.values():
            act.triggered.connect(self.on_plant_type_selected)
        self.act_lang_zh.triggered.connect(self.on_language_selected)
        self.act_lang_en.triggered.connect(self.on_language_selected)

        self._install_vtk_pick_observer()

        self._update_buttons()
        self._apply_language()
        self._update_status(self.tr("提示：标注模式用 Shift+左键 选点。"))
        self._refresh_point_lists()
        self._refresh_instance_meta_ui()

    def _set_combo_current_by_data(self, combo: QtWidgets.QComboBox, data_value):
        for i in range(combo.count()):
            if combo.itemData(i) == data_value:
                combo.setCurrentIndex(i)
                return True
        return False


    def _set_view_combo_items(self):
        current = self.combo_view.currentData()
        self.combo_view.blockSignals(True)
        self.combo_view.clear()
        self.combo_view.addItem(self.tr("RGB"), self.VIEW_RGB)
        self.combo_view.addItem(self.tr("语义"), self.VIEW_SEM)
        self.combo_view.addItem(self.tr("实例"), self.VIEW_INST)
        self.combo_view.addItem(self.tr("表型标签"), self.VIEW_LABEL)
        if current:
            self._set_combo_current_by_data(self.combo_view, current)
        self.combo_view.blockSignals(False)


    def _set_anno_semantic_items(self):
        current = self.combo_anno_semantic.currentData()
        self.combo_anno_semantic.blockSignals(True)
        self.combo_anno_semantic.clear()
        for key, text in [
            ("leaf", self.tr("叶")),
            ("stem", self.tr("茎")),
            ("flower", self.tr("花")),
            ("fruit", self.tr("果")),
        ]:
            self.combo_anno_semantic.addItem(text, key)
        if current:
            self._set_combo_current_by_data(self.combo_anno_semantic, current)
        self.combo_anno_semantic.blockSignals(False)


    def _set_label_desc_items(self):
        current = self.combo_label_desc.currentData()
        self.combo_label_desc.blockSignals(True)
        self.combo_label_desc.clear()
        for key, text in [
            ("unselected", "未选择"),
            ("complete", "完整"),
            ("broken", "折断"),
            ("missing", "缺失"),
            ("noise", "噪声"),
        ]:
            self.combo_label_desc.addItem(self.tr(text), key)
        if current:
            self._set_combo_current_by_data(self.combo_label_desc, current)
        self.combo_label_desc.blockSignals(False)


    def _get_view_mode(self) -> str:
        data = self.combo_view.currentData()
        if data:
            return str(data)
        return self.combo_view.currentText()


    def on_language_selected(self):
        act = self.sender()
        if act is None:
            return
        lang = act.data()
        if lang not in ["zh", "en"]:
            return
        self._i18n.set_language(lang)
        self._settings.setValue("lang", lang)
        self._apply_language()


    def _apply_language(self):
        self.setWindowTitle(self.tr("植物表型标注工具"))
        self.menu_file.setTitle(self.tr("文件"))
        self.menu_view.setTitle(self.tr("显示"))
        self.menu_anno.setTitle(self.tr("标注"))
        self.menu_calc.setTitle(self.tr("计算"))
        self.menu_plant.setTitle(self.tr("植物类型"))
        self.menu_lang.setTitle(self.tr("语言"))
        self.act_load.setText(self.tr("加载点云"))
        self.act_export_dir.setText(self.tr("导出目录"))
        self.act_export_phenotype.setText(self.tr("导出表型CSV"))
        self.act_quit.setText(self.tr("退出"))
        self.act_view_rgb.setText(self.tr("RGB"))
        self.act_view_sem.setText(self.tr("语义"))
        self.act_view_inst.setText(self.tr("实例"))
        self.act_view_label.setText(self.tr("表型标签"))
        self.act_start_anno.setText(self.tr("进入标注模式"))
        self.act_back_browse.setText(self.tr("返回浏览模式"))
        self.act_recommend_length.setText(self.tr("推荐叶长"))
        self.act_generate_length.setText(self.tr("生成叶长"))
        self.act_recommend_width.setText(self.tr("推荐叶宽"))
        self.act_generate_width.setText(self.tr("生成叶宽"))
        self.act_smooth_paths.setText(self.tr("平滑叶长/叶宽"))
        self.act_compute_leaf_area.setText(self.tr("计算叶面积"))
        self.act_compute_leaf_projected_area.setText(self.tr("计算投影面积"))
        self.act_compute_leaf_inclination.setText(self.tr("计算叶倾角"))
        self.act_compute_leaf_stem_angle.setText(self.tr("计算叶夹角"))
        self.act_export_labeled.setText(self.tr("生成标记点云"))
        self.act_save_labels.setText(self.tr("保存标注"))
        self.act_compute_stem.setText(self.tr("计算茎粗"))
        self.act_compute_stem_length.setText(self.tr("计算茎长"))
        self.act_compute_flower_fruit.setText(self.tr("计算花果OBB"))
        self.act_growth_manual.setText(self.tr("生长方向：手动最低点"))
        self.act_growth_stem.setText(self.tr("生长方向：茎下半部"))
        self.act_measure_plant.setText(self.tr("测量株高/冠幅"))
        self.act_help.setText(self.tr("帮助"))
        self.act_about.setText(self.tr("关于"))
        self.act_lang_zh.setText(self.tr("中文"))
        self.act_lang_en.setText(self.tr("英文"))

        self.btn_load.setText(self.tr("加载点云"))
        self.btn_start_anno.setText(self.tr("进入标注模式（仅显示当前实例）"))
        self.btn_back_browse.setText(self.tr("返回浏览模式（显示整株）"))
        self.btn_recommend_length.setText(self.tr("推荐叶长（B1→T1）"))
        self.btn_generate_length.setText(self.tr("生成叶长（B1 + 控制点 + T1）"))
        self.btn_recommend_width.setText(self.tr("推荐叶宽（沿当前叶长）"))
        self.btn_generate_width.setText(self.tr("生成叶宽（W1→W2）"))
        self.btn_compute_leaf_area.setText(self.tr("计算叶面积"))
        self.btn_compute_leaf_projected_area.setText(self.tr("计算投影面积"))
        self.btn_compute_leaf_inclination.setText(self.tr("计算叶倾角"))
        self.btn_compute_leaf_stem_angle.setText(self.tr("计算叶夹角"))
        self.btn_delete.setText(self.tr("删除选中标记"))
        self.btn_rename_ctrl.setText(self.tr("重命名叶长控制点顺序 (C#)"))
        self.btn_rename_width_ctrl.setText(self.tr("重命名叶宽控制点顺序 (WC#)"))
        self.btn_export.setText(self.tr("保存标注（JSON+Label）"))
        self.btn_export_phenotype.setText(self.tr("导出表型CSV"))

        self.btn_view_front.setText(self.tr("正视图"))
        self.btn_view_side.setText(self.tr("侧视图"))
        self.btn_view_top.setText(self.tr("顶视图"))
        self.btn_pick_view_center.setText(self.tr("选择视图中心"))
        self.btn_toggle_aabb.setText(self.tr("显示AABB盒"))
        self.btn_toggle_obb.setText(self.tr("显示OBB盒"))
        self.btn_toggle_stem_cyl.setText(self.tr("显示茎圆柱"))
        self.btn_toggle_stem_path.setText(self.tr("显示茎长路径"))
        self.btn_toggle_growth_dir.setText(self.tr("显示生长方向"))
        self.btn_toggle_plant_height.setText(self.tr("显示株高"))
        self.btn_toggle_plant_crown.setText(self.tr("显示冠幅"))
        self.btn_rotate_minus.setText(self.tr("旋转 -"))
        self.btn_rotate_plus.setText(self.tr("旋转 +"))
        self.btn_rotate_start.setText(self.tr("开始旋转"))
        self.btn_rotate_finish.setText(self.tr("完成旋转"))
        self.btn_rotate_reset.setText(self.tr("重置旋转"))

        self._set_view_combo_items()
        self._set_anno_semantic_items()
        self._set_label_desc_items()
        self._update_instance_sem_label()
        self._refresh_sem_filter_options()
        self.table_phenotype.setHorizontalHeaderLabels([
            self.tr("实例"), self.tr("语义"), self.tr("名字"), self.tr("值")
        ])
        self._update_phenotype_table()
        self._update_view_legend()

        for t, act in self.plant_type_actions.items():
            act.setText(self._plant_type_display(t))

        self.status_group.setTitle(self.tr("状态信息"))
        self.sem_group.setTitle(self.tr("语义标签"))
        self.inst_group.setTitle(self.tr("实例"))
        self.bbox_group.setTitle(self.tr("包围盒"))
        self.meta_group.setTitle(self.tr("实例备注"))
        self.points_group.setTitle(self.tr("标记信息"))
        self.leaf_area_group.setTitle(self.tr("叶面积"))
        self.leaf_angle_group.setTitle(self.tr("叶角度"))
        self.phenotype_group.setTitle(self.tr("表型信息"))
        self.view_tools_group.setTitle(self.tr("视图操作"))
        self.measure_group.setTitle(self.tr("测量"))
        self.plant_measure_group.setTitle(self.tr("整株测量"))
        self.view_legend_group.setTitle(self.tr("当前视图标签"))

        self.lbl_sem_leaf.setText(self.tr("叶语义："))
        self.lbl_sem_stem.setText(self.tr("茎语义："))
        self.lbl_sem_flower.setText(self.tr("花语义："))
        self.lbl_sem_fruit.setText(self.tr("果语义："))
        self.lbl_sem_note.setText(self.tr("注：-1 表示没有该语义"))
        self.lbl_anno_sem.setText(self.tr("标注语义："))
        self.lbl_inst_label.setText(self.tr("实例标签："))
        self.lbl_meta_info.setText(self.tr("附加信息"))
        self.lbl_meta_remark.setText(self.tr("实例备注"))
        self.lbl_points_base.setText(self.tr("叶基 (B1)"))
        self.lbl_points_tip.setText(self.tr("叶尖 (T1)"))
        self.lbl_points_ctrl.setText(self.tr("叶长控制点 (C1..Cn)"))
        self.lbl_points_width.setText(self.tr("叶宽端点 (W1/W2)"))
        self.lbl_points_width_ctrl.setText(self.tr("叶宽控制点 (WC1..WCn)"))
        self.lbl_rotate_axis.setText(self.tr("旋转轴"))
        self.lbl_rotate_step.setText(self.tr("步长(°)"))

        # refresh static labels by recreating group titles
        self.info.setText(self.tr("状态：未加载") if self.session.cloud is None else self.info.text())
        self._refresh_toggle_texts()


    def _refresh_toggle_texts(self):
        base_on = self.btn_toggle_base.isChecked()
        tip_on = self.btn_toggle_tip.isChecked()
        ctrl_on = self.btn_toggle_ctrl.isChecked()
        width_on = self.btn_toggle_width.isChecked()
        wctrl_on = self.btn_toggle_width_ctrl.isChecked()
        measure_on = self.btn_toggle_measure.isChecked()

        self.btn_toggle_base.setText(self.tr("叶基选择：开启") if base_on else self.tr("叶基选择：关闭"))
        self.btn_toggle_tip.setText(self.tr("叶尖选择：开启") if tip_on else self.tr("叶尖选择：关闭"))
        self.btn_toggle_ctrl.setText(self.tr("控制点选择：开启") if ctrl_on else self.tr("叶长控制点选择：关闭"))
        self.btn_toggle_width.setText(self.tr("叶宽点选择：开启 (W1/W2)") if width_on else self.tr("叶宽端点选择：关闭 (W1/W2)"))
        self.btn_toggle_width_ctrl.setText(self.tr("叶宽控制点选择：开启") if wctrl_on else self.tr("叶宽控制点选择：关闭"))
        self.btn_toggle_measure.setText(self.tr("测距：开启") if measure_on else self.tr("测距：关闭"))
    # ----------------------------
    # status
    # ----------------------------
