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

    def __init__(self):
        super().__init__()
        self.setWindowTitle("植物表型标注工具")
        self._settings = QtCore.QSettings("LeafAnnotator", "3d_plant_phenotype")

        menu = self.menuBar()
        file_menu = menu.addMenu("文件")
        self.act_load = file_menu.addAction("加载点云")
        self.act_export_dir = file_menu.addAction("导出目录")
        self.act_export_phenotype = file_menu.addAction("导出表型CSV")
        file_menu.addSeparator()
        self.act_quit = file_menu.addAction("退出")

        view_menu = menu.addMenu("显示")
        self.act_view_rgb = view_menu.addAction("RGB")
        self.act_view_sem = view_menu.addAction("语义")
        self.act_view_inst = view_menu.addAction("实例")
        self.act_view_label = view_menu.addAction("表型标签")
        for a in [self.act_view_rgb, self.act_view_sem, self.act_view_inst, self.act_view_label]:
            a.setCheckable(True)
        self.view_group = QtWidgets.QActionGroup(self)
        self.view_group.setExclusive(True)
        self.view_group.addAction(self.act_view_rgb)
        self.view_group.addAction(self.act_view_sem)
        self.view_group.addAction(self.act_view_inst)
        self.view_group.addAction(self.act_view_label)
        self.act_view_rgb.setChecked(True)

        anno_menu = menu.addMenu("标注")
        self.act_start_anno = anno_menu.addAction("进入标注模式")
        self.act_back_browse = anno_menu.addAction("返回浏览模式")

        calc_menu = menu.addMenu("计算")
        self.act_recommend_length = calc_menu.addAction("推荐叶长")
        self.act_generate_length = calc_menu.addAction("生成叶长")
        self.act_recommend_width = calc_menu.addAction("推荐叶宽")
        self.act_generate_width = calc_menu.addAction("生成叶宽")
        self.act_smooth_paths = calc_menu.addAction("平滑叶长/叶宽")
        self.act_compute_leaf_area = calc_menu.addAction("计算叶面积")
        self.act_compute_leaf_projected_area = calc_menu.addAction("计算投影面积")
        self.act_compute_leaf_inclination = calc_menu.addAction("计算叶倾角")
        self.act_compute_leaf_stem_angle = calc_menu.addAction("计算叶夹角")
        calc_menu.addSeparator()
        self.act_export_labeled = calc_menu.addAction("生成标记点云")
        self.act_save_labels = menu.addAction("保存标注")
        self.act_save_labels.setShortcut("Ctrl+S")
        self.act_compute_stem = calc_menu.addAction("计算茎粗")
        self.act_compute_stem_length = calc_menu.addAction("计算茎长")
        self.act_compute_flower_fruit = calc_menu.addAction("计算花果OBB")

        plant_menu = menu.addMenu("植物类型")
        self.plant_types = ["玉米", "草莓", "水稻", "小麦"]
        self.plant_type_group = QtWidgets.QActionGroup(self)
        self.plant_type_group.setExclusive(True)
        self.plant_type_actions = {}
        for t in self.plant_types:
            act = plant_menu.addAction(t)
            act.setCheckable(True)
            self.plant_type_group.addAction(act)
            self.plant_type_actions[t] = act

        self.act_help = menu.addAction("帮助")
        self.act_about = menu.addAction("关于")

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
        self.btn_load = QtWidgets.QPushButton("加载点云")
        self.combo_view = QtWidgets.QComboBox()
        self.combo_view.addItems(["RGB", "语义", "实例", "表型标签"])
        self.combo_inst = QtWidgets.QComboBox()
        self.btn_start_anno = QtWidgets.QPushButton("进入标注模式（仅显示当前实例）")
        self.btn_back_browse = QtWidgets.QPushButton("返回浏览模式（显示整株）")

        self.btn_toggle_base = QtWidgets.QPushButton("叶基选择：关闭")
        self.btn_toggle_tip = QtWidgets.QPushButton("叶尖选择：关闭")
        self.btn_toggle_ctrl = QtWidgets.QPushButton("叶长控制点选择：关闭")
        self.btn_toggle_width = QtWidgets.QPushButton("叶宽端点选择：关闭 (W1/W2)")
        self.btn_toggle_width_ctrl = QtWidgets.QPushButton("叶宽控制点选择：关闭")
        self.btn_toggle_measure = QtWidgets.QPushButton("测距：关闭")
        for b in [
            self.btn_toggle_base, self.btn_toggle_tip, self.btn_toggle_ctrl,
            self.btn_toggle_width, self.btn_toggle_width_ctrl, self.btn_toggle_measure
        ]:
            b.setCheckable(True)

        self.btn_recommend_length = QtWidgets.QPushButton("推荐叶长（B1→T1）")
        self.btn_generate_length = QtWidgets.QPushButton("生成叶长（B1 + 控制点 + T1）")
        self.btn_recommend_width = QtWidgets.QPushButton("推荐叶宽（沿当前叶长）")
        self.btn_generate_width = QtWidgets.QPushButton("生成叶宽（W1→W2）")
        self.btn_compute_leaf_area = QtWidgets.QPushButton("计算叶面积")
        self.btn_compute_leaf_projected_area = QtWidgets.QPushButton("计算投影面积")
        self.btn_compute_leaf_inclination = QtWidgets.QPushButton("计算叶倾角")
        self.btn_compute_leaf_stem_angle = QtWidgets.QPushButton("计算叶夹角")

        self.btn_delete = QtWidgets.QPushButton("删除选中标记")
        self.btn_rename_ctrl = QtWidgets.QPushButton("重命名叶长控制点顺序 (C#)")
        self.btn_rename_width_ctrl = QtWidgets.QPushButton("重命名叶宽控制点顺序 (WC#)")

        self.btn_export = QtWidgets.QPushButton("保存标注（JSON+Label）")

        # ---------- status ----------
        self.info = QtWidgets.QLabel("状态：未加载")
        self.info.setWordWrap(True)
        status_group = QtWidgets.QGroupBox("状态信息")
        status_layout = QtWidgets.QVBoxLayout(status_group)
        status_layout.addWidget(self.info)
        panel.addWidget(status_group)

        # ---------- semantic labels ----------
        sem_group = QtWidgets.QGroupBox("语义标签")
        sem_layout = QtWidgets.QVBoxLayout(sem_group)
        self.combo_anno_semantic = QtWidgets.QComboBox()
        self.combo_anno_semantic.addItems(["叶", "茎", "花", "果"])

        sem_row = QtWidgets.QHBoxLayout()
        sem_row.addWidget(QtWidgets.QLabel("叶语义："))
        self.combo_sem_filter = QtWidgets.QComboBox()
        sem_row.addWidget(self.combo_sem_filter)
        sem_layout.addLayout(sem_row)

        stem_row = QtWidgets.QHBoxLayout()
        stem_row.addWidget(QtWidgets.QLabel("茎语义："))
        self.combo_stem_filter = QtWidgets.QComboBox()
        stem_row.addWidget(self.combo_stem_filter)
        sem_layout.addLayout(stem_row)

        flower_row = QtWidgets.QHBoxLayout()
        flower_row.addWidget(QtWidgets.QLabel("花语义："))
        self.combo_flower_filter = QtWidgets.QComboBox()
        flower_row.addWidget(self.combo_flower_filter)
        sem_layout.addLayout(flower_row)

        fruit_row = QtWidgets.QHBoxLayout()
        fruit_row.addWidget(QtWidgets.QLabel("果语义："))
        self.combo_fruit_filter = QtWidgets.QComboBox()
        fruit_row.addWidget(self.combo_fruit_filter)
        sem_layout.addLayout(fruit_row)

        sem_layout.addWidget(QtWidgets.QLabel("注：-1 表示没有该语义"))
        panel.addWidget(sem_group)
        for combo in [self.combo_sem_filter, self.combo_stem_filter, self.combo_flower_filter, self.combo_fruit_filter]:
            combo.addItem("-1")
            combo.setCurrentText("-1")

        # ---------- instance selector ----------
        inst_group = QtWidgets.QGroupBox("实例")
        inst_layout = QtWidgets.QVBoxLayout(inst_group)
        anno_row = QtWidgets.QHBoxLayout()
        anno_row.addWidget(QtWidgets.QLabel("标注语义："))
        anno_row.addWidget(self.combo_anno_semantic)
        inst_layout.addLayout(anno_row)
        inst_row = QtWidgets.QHBoxLayout()
        inst_row.addWidget(QtWidgets.QLabel("实例标签："))
        inst_row.addWidget(self.combo_inst)
        inst_layout.addLayout(inst_row)
        self.lbl_inst_sem = QtWidgets.QLabel("语义标签：-")
        self.lbl_inst_sem.setWordWrap(True)
        inst_layout.addWidget(self.lbl_inst_sem)
        panel.addWidget(inst_group)

        # ---------- bbox info ----------
        bbox_group = QtWidgets.QGroupBox("包围盒")
        bbox_layout = QtWidgets.QVBoxLayout(bbox_group)
        self.bbox_info = QtWidgets.QLabel("AABB：-\nOBB：-")
        self.bbox_info.setWordWrap(True)
        bbox_layout.addWidget(self.bbox_info)
        panel.addWidget(bbox_group)

        # ---------- instance meta ----------
        self.meta_group = QtWidgets.QGroupBox("实例备注")
        meta_layout = QtWidgets.QVBoxLayout(self.meta_group)
        meta_layout.addWidget(QtWidgets.QLabel("附加信息"))
        self.combo_label_desc = QtWidgets.QComboBox()
        self.combo_label_desc.addItems(["未选择", "完整", "折断", "缺失", "噪声"])
        meta_layout.addWidget(self.combo_label_desc)
        meta_layout.addWidget(QtWidgets.QLabel("实例备注"))
        self.text_remark = QtWidgets.QTextEdit()
        self.text_remark.setFixedHeight(
            self.text_remark.fontMetrics().lineSpacing() * TEXT_REMARK_LINES + TEXT_REMARK_PADDING
        )
        meta_layout.addWidget(self.text_remark)
        panel.addWidget(self.meta_group)
        self.meta_group.setEnabled(False)

        # ---------- marker lists ----------
        self.points_group = QtWidgets.QGroupBox("标记信息")
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
        points_layout.addWidget(QtWidgets.QLabel("叶基 (B1)"))
        points_layout.addWidget(self.list_base)
        points_layout.addWidget(self.btn_toggle_base)
        points_layout.addWidget(QtWidgets.QLabel("叶尖 (T1)"))
        points_layout.addWidget(self.list_tip)
        points_layout.addWidget(self.btn_toggle_tip)
        points_layout.addWidget(QtWidgets.QLabel("叶长控制点 (C1..Cn)"))
        points_layout.addWidget(self.list_ctrl)
        points_layout.addWidget(self.btn_toggle_ctrl)
        points_layout.addWidget(QtWidgets.QLabel("叶宽端点 (W1/W2)"))
        points_layout.addWidget(self.list_width)
        points_layout.addWidget(self.btn_toggle_width)
        points_layout.addWidget(QtWidgets.QLabel("叶宽控制点 (WC1..WCn)"))
        points_layout.addWidget(self.list_width_ctrl)
        points_layout.addWidget(self.btn_toggle_width_ctrl)
        panel.addWidget(self.points_group)

        # ---------- leaf area ----------
        self.leaf_area_group = QtWidgets.QGroupBox("叶面积")
        leaf_area_layout = QtWidgets.QVBoxLayout(self.leaf_area_group)
        leaf_area_layout.addWidget(self.btn_compute_leaf_area)
        leaf_area_layout.addWidget(self.btn_compute_leaf_projected_area)
        panel.addWidget(self.leaf_area_group)

        # ---------- leaf angles ----------
        self.leaf_angle_group = QtWidgets.QGroupBox("叶角度")
        leaf_angle_layout = QtWidgets.QVBoxLayout(self.leaf_angle_group)
        leaf_angle_layout.addWidget(self.btn_compute_leaf_inclination)
        leaf_angle_layout.addWidget(self.btn_compute_leaf_stem_angle)
        panel.addWidget(self.leaf_angle_group)

        # ---------- phenotype table ----------
        self.phenotype_group = QtWidgets.QGroupBox("表型信息")
        phenotype_layout = QtWidgets.QVBoxLayout(self.phenotype_group)
        self.table_phenotype = QtWidgets.QTableWidget(0, 4)
        self.table_phenotype.setHorizontalHeaderLabels(["实例", "语义", "表型名字", "表型值"])
        header = self.table_phenotype.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        self.table_phenotype.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table_phenotype.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.table_phenotype.setAlternatingRowColors(True)
        self.table_phenotype.setWordWrap(False)
        self.table_phenotype.setTextElideMode(QtCore.Qt.ElideRight)
        phenotype_layout.addWidget(self.table_phenotype)
        self.btn_export_phenotype = QtWidgets.QPushButton("导出表型CSV")
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

        view_tools_group = QtWidgets.QGroupBox("视图操作")
        view_tools_layout = QtWidgets.QVBoxLayout(view_tools_group)
        self.btn_view_front = QtWidgets.QPushButton("正视图")
        self.btn_view_side = QtWidgets.QPushButton("侧视图")
        self.btn_view_top = QtWidgets.QPushButton("顶视图")
        self.btn_pick_view_center = QtWidgets.QPushButton("选择视图中心")
        self.btn_toggle_aabb = QtWidgets.QPushButton("显示AABB盒")
        self.btn_toggle_obb = QtWidgets.QPushButton("显示OBB盒")
        self.btn_toggle_stem_cyl = QtWidgets.QPushButton("显示茎圆柱")
        self.btn_toggle_stem_path = QtWidgets.QPushButton("显示茎长路径")
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

        measure_group = QtWidgets.QGroupBox("测量")
        measure_layout = QtWidgets.QVBoxLayout(measure_group)
        measure_layout.addWidget(self.btn_toggle_measure)

        right_layout.addWidget(view_tools_group)
        right_layout.addWidget(measure_group)
        right_layout.addStretch(1)

        self.view_legend_group = QtWidgets.QGroupBox("当前视图标签")
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

        anno_default = self._settings.value("anno_semantic", "叶", type=str)
        if anno_default in ["叶", "茎", "花", "果"]:
            self.combo_anno_semantic.setCurrentText(anno_default)
            self.annotate_semantic = self._get_annotation_semantic_key()
        plant_default = self._settings.value("plant_type", "玉米", type=str)
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

        self.act_view_rgb.triggered.connect(lambda: self._set_view_mode("RGB"))
        self.act_view_sem.triggered.connect(lambda: self._set_view_mode("语义"))
        self.act_view_inst.triggered.connect(lambda: self._set_view_mode("实例"))
        self.act_view_label.triggered.connect(lambda: self._set_view_mode("表型标签"))
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
        self.btn_view_front.clicked.connect(self.on_view_front)
        self.btn_view_side.clicked.connect(self.on_view_side)
        self.btn_view_top.clicked.connect(self.on_view_top)
        self.btn_pick_view_center.clicked.connect(self.on_pick_view_center)
        self.btn_toggle_aabb.toggled.connect(self.on_toggle_aabb)
        self.btn_toggle_obb.toggled.connect(self.on_toggle_obb)
        self.btn_toggle_stem_cyl.toggled.connect(self.on_toggle_stem_cyl)
        self.btn_toggle_stem_path.toggled.connect(self.on_toggle_stem_path)
        for act in self.plant_type_actions.values():
            act.triggered.connect(self.on_plant_type_selected)

        self._install_vtk_pick_observer()

        self._update_buttons()
        self._update_status("提示：标注模式用 Shift+左键 选点。")
        self._refresh_point_lists()
        self._refresh_instance_meta_ui()

    # ----------------------------
    # status
    # ----------------------------
