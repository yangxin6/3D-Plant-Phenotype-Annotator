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

# 窗口尺寸相关的数值
PANEL_WIDTH_DEFAULT = 280  # 左侧面板默认宽度
PANEL_WIDTH_LEAF_ANNOTATE = 260  # 叶子标注模式下左侧面板宽度
PANEL_INNER_PADDING = 24  # 左侧面板内部留白（用于控件宽度计算）
PANEL_ROW_LABEL_SPACE = 110  # 左侧面板带标签行预留宽度（用于下拉框）
PANEL_COMBO_MIN_WIDTH = 80  # 下拉框最小宽度
PANEL_LIST_MIN_WIDTH = 200  # 列表最小宽度
PANEL_BUTTON_MIN_WIDTH = 80  # 按钮最小宽度
TEXT_REMARK_LINES = 2  # 实例备注文本框显示的行数
TEXT_REMARK_PADDING = 12  # 实例备注文本框额外内边距高度
LIST_BASE_MAX_HEIGHT = 55  # 叶基列表最大高度
LIST_TIP_MAX_HEIGHT = 55  # 叶尖列表最大高度
LIST_WIDTH_MAX_HEIGHT = 65  # 叶宽端点列表最大高度
LIST_CTRL_MAX_HEIGHT = 120  # 叶长控制点列表最大高度
LIST_WIDTH_CTRL_MAX_HEIGHT = 120  # 叶宽控制点列表最大高度
VIEW_LEGEND_HEIGHT = 160  # 右下角标签映射区域固定高度
LABEL_SWATCH_SIZE = 12  # 标签颜色块尺寸


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
    lut = {}
    for k in uniq:
        kk = int(k)
        if kk == -1:
            lut[kk] = np.array([128, 128, 128], dtype=np.uint8)
        elif kk == 0:
            lut[kk] = np.array([255, 0, 0], dtype=np.uint8)        # red
        elif kk == 1:
            lut[kk] = np.array([0, 255, 0], dtype=np.uint8)        # green
        elif kk == 2:
            lut[kk] = np.array([0, 0, 255], dtype=np.uint8)        # blue
        elif kk == 3:
            lut[kk] = np.array([139, 69, 19], dtype=np.uint8)      # brown
        elif kk == 4:
            lut[kk] = np.array([128, 0, 128], dtype=np.uint8)      # purple
        elif kk == 5:
            lut[kk] = np.array([255, 165, 0], dtype=np.uint8)      # orange
        else:
            lut[kk] = stable_color_from_id(kk)
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

    def __init__(self):
        super().__init__()
        self.setWindowTitle("植物表型标注工具")
        self._settings = QtCore.QSettings("LeafAnnotator", "3d_plant_phenotype")

        menu = self.menuBar()
        file_menu = menu.addMenu("文件")
        self.act_load = file_menu.addAction("加载点云")
        self.act_export_dir = file_menu.addAction("导出目录")
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

        self.btn_export.clicked.connect(self.on_save_annotations)

        self.text_remark.textChanged.connect(self.on_instance_meta_changed)
        self.combo_label_desc.currentIndexChanged.connect(self.on_instance_meta_changed)
        self.act_load.triggered.connect(self.on_load)
        self.act_export_dir.triggered.connect(self.on_choose_export_dir)
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

    def _start_busy_dialog(self, text: str):
        dlg = QtWidgets.QProgressDialog(text, None, 0, 0, self)
        dlg.setWindowTitle("加载中")
        dlg.setWindowModality(QtCore.Qt.ApplicationModal)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.show()
        QtWidgets.QApplication.processEvents()
        return dlg

    def _finish_busy_dialog(self, dlg: Optional[QtWidgets.QProgressDialog]):
        if dlg is None:
            return
        dlg.close()
        dlg.deleteLater()

    def _set_instance_meta_ui(self, remark: str, label_desc: str):
        self.text_remark.blockSignals(True)
        self.combo_label_desc.blockSignals(True)
        self.text_remark.setPlainText("" if remark is None else str(remark))
        items = [self.combo_label_desc.itemText(i) for i in range(self.combo_label_desc.count())]
        if label_desc in items:
            self.combo_label_desc.setCurrentText(label_desc)
        else:
            self.combo_label_desc.setCurrentText("未选择")
        self.text_remark.blockSignals(False)
        self.combo_label_desc.blockSignals(False)

    def _refresh_instance_meta_ui(self):
        if (not self.annotating) or (self.session.current_inst_id is None):
            self._set_instance_meta_ui("", "")
            return
        remark, label_desc = self.session.get_instance_meta(self.session.current_inst_id)
        self._set_instance_meta_ui(remark, label_desc)

    def _get_current_instance_sem_label(self):
        if self.session.cloud is None:
            return None
        if self.session.current_inst_id is None or self.session.leaf_global_idx is None:
            return None
        if len(self.session.leaf_global_idx) == 0:
            return None
        sem = self.session.get_full_sem()[self.session.leaf_global_idx]
        if len(sem) == 0:
            return None
        vals, counts = np.unique(sem, return_counts=True)
        if len(vals) == 0:
            return None
        return int(vals[int(np.argmax(counts))])

    def _update_instance_sem_label(self):
        if not hasattr(self, "lbl_inst_sem"):
            return
        if (not self.annotating) or (self.session.current_inst_id is None):
            self.lbl_inst_sem.setText("语义标签：-")
            return
        sem_label = self._get_current_instance_sem_label()
        if sem_label is None:
            self.lbl_inst_sem.setText("语义标签：-")
        else:
            self.lbl_inst_sem.setText(f"语义标签：{sem_label}")

    def _update_semantic_map_from_ui(self):
        def _val(combo: QtWidgets.QComboBox):
            text = combo.currentText()
            if not text:
                return None
            v = int(text)
            return None if v < 0 else v

        self.session.semantic_map = {
            "leaf": _val(self.combo_sem_filter),
            "stem": _val(self.combo_stem_filter),
            "flower": _val(self.combo_flower_filter),
            "fruit": _val(self.combo_fruit_filter),
        }

    def _refresh_anno_semantic_options(self):
        if not hasattr(self, "combo_anno_semantic"):
            return
        model = self.combo_anno_semantic.model()
        if model is None or not hasattr(model, "item"):
            return
        mapping = {"叶": "leaf", "茎": "stem", "花": "flower", "果": "fruit"}
        enabled_any = False
        for i in range(self.combo_anno_semantic.count()):
            text = self.combo_anno_semantic.itemText(i)
            key = mapping.get(text)
            enabled = True
            if key is not None:
                val = self.session.semantic_map.get(key)
                enabled = val is not None and int(val) >= 0
            item = model.item(i)
            if item is not None:
                item.setEnabled(enabled)
            enabled_any = enabled_any or enabled
        if not enabled_any:
            return
        cur_idx = self.combo_anno_semantic.currentIndex()
        if cur_idx >= 0:
            item = model.item(cur_idx)
            if item is not None and not item.isEnabled():
                for i in range(self.combo_anno_semantic.count()):
                    item = model.item(i)
                    if item is not None and item.isEnabled():
                        self.combo_anno_semantic.setCurrentIndex(i)
                        break

    def _get_annotation_semantic_key(self) -> str:
        text = self.combo_anno_semantic.currentText()
        mapping = {"叶": "leaf", "茎": "stem", "花": "flower", "果": "fruit"}
        return mapping.get(text, "leaf")

    def _refresh_instance_list_for_annotation(self):
        key = self._get_annotation_semantic_key()
        sem_label = self.session.semantic_map.get(key)
        if sem_label is None:
            self.combo_inst.clear()
            self._update_buttons()
            return
        self._refresh_instance_list(target_sem=int(sem_label), keep_current=False)
        self._update_buttons()

    def on_anno_semantic_changed(self):
        key = self._get_annotation_semantic_key()
        self.annotate_semantic = key
        self._settings.setValue("anno_semantic", self.combo_anno_semantic.currentText())
        if not self.annotating and self.session.cloud is not None:
            self._refresh_instance_list_for_annotation()
        self._update_buttons()

    def _get_instance_sem_map(self) -> dict:
        if self.session.cloud is None:
            return {}
        inst = self.session.get_full_inst()
        sem = self.session.get_full_sem()
        mask = inst >= 0
        inst = inst[mask]
        sem = sem[mask]
        if len(inst) == 0:
            return {}
        out = {}
        for inst_id in np.unique(inst):
            vals = sem[inst == inst_id]
            if len(vals) == 0:
                continue
            uvals, counts = np.unique(vals, return_counts=True)
            out[int(inst_id)] = int(uvals[int(np.argmax(counts))])
        return out

    def _refresh_instance_list(self, target_sem: int = None, keep_current: bool = True):
        if self.session.cloud is None:
            return
        sem_map = self._get_instance_sem_map()
        ids = sorted(sem_map.keys())
        if target_sem is not None:
            ids = [i for i in ids if sem_map.get(i) == int(target_sem)]
        cur = self.combo_inst.currentText()
        self.combo_inst.blockSignals(True)
        self.combo_inst.clear()
        for _id in ids:
            self.combo_inst.addItem(str(int(_id)))
        self.combo_inst.blockSignals(False)
        if keep_current and cur:
            if cur in [self.combo_inst.itemText(i) for i in range(self.combo_inst.count())]:
                self.combo_inst.setCurrentText(cur)
                return
        if self.combo_inst.count() > 0:
            self.combo_inst.setCurrentIndex(0)

    def _refresh_sem_filter_options(self):
        if self.session.cloud is None:
            return
        sem_vals = self.session.get_full_sem()
        uniq = np.unique(sem_vals.astype(np.int64))
        uniq = uniq[uniq >= 0]
        opts = [-1] + uniq.tolist()
        self.combo_sem_filter.blockSignals(True)
        self.combo_sem_filter.clear()
        for v in opts:
            self.combo_sem_filter.addItem(str(int(v)))
        self.combo_sem_filter.blockSignals(False)
        if self.combo_sem_filter.count() > 0:
            prefer_sem = self.session.semantic_map.get("leaf")
            last_sem = self._settings.value("leaf_sem_label", "", type=str)
            items = [self.combo_sem_filter.itemText(i) for i in range(self.combo_sem_filter.count())]
            if prefer_sem is not None and str(int(prefer_sem)) in items:
                self.combo_sem_filter.setCurrentText(str(int(prefer_sem)))
            elif prefer_sem is None and "-1" in items:
                self.combo_sem_filter.setCurrentText("-1")
            elif last_sem and last_sem in items:
                self.combo_sem_filter.setCurrentText(last_sem)
            else:
                self.combo_sem_filter.setCurrentIndex(0)

        self.combo_stem_filter.blockSignals(True)
        self.combo_stem_filter.clear()
        for v in opts:
            self.combo_stem_filter.addItem(str(int(v)))
        self.combo_stem_filter.blockSignals(False)
        if self.combo_stem_filter.count() > 0:
            prefer_stem = self.session.semantic_map.get("stem")
            last_stem = self._settings.value("stem_sem_label", "", type=str)
            items = [self.combo_stem_filter.itemText(i) for i in range(self.combo_stem_filter.count())]
            if prefer_stem is not None and str(int(prefer_stem)) in items:
                self.combo_stem_filter.setCurrentText(str(int(prefer_stem)))
            elif prefer_stem is None and "-1" in items:
                self.combo_stem_filter.setCurrentText("-1")
            elif last_stem and last_stem in items:
                self.combo_stem_filter.setCurrentText(last_stem)
            else:
                self.combo_stem_filter.setCurrentIndex(0)

        self.combo_flower_filter.blockSignals(True)
        self.combo_flower_filter.clear()
        for v in opts:
            self.combo_flower_filter.addItem(str(int(v)))
        self.combo_flower_filter.blockSignals(False)
        if self.combo_flower_filter.count() > 0:
            prefer_flower = self.session.semantic_map.get("flower")
            last_flower = self._settings.value("flower_sem_label", "", type=str)
            items = [self.combo_flower_filter.itemText(i) for i in range(self.combo_flower_filter.count())]
            if prefer_flower is not None and str(int(prefer_flower)) in items:
                self.combo_flower_filter.setCurrentText(str(int(prefer_flower)))
            elif prefer_flower is None and "-1" in items:
                self.combo_flower_filter.setCurrentText("-1")
            elif last_flower and last_flower in items:
                self.combo_flower_filter.setCurrentText(last_flower)
            else:
                self.combo_flower_filter.setCurrentIndex(0)

        self.combo_fruit_filter.blockSignals(True)
        self.combo_fruit_filter.clear()
        for v in opts:
            self.combo_fruit_filter.addItem(str(int(v)))
        self.combo_fruit_filter.blockSignals(False)
        if self.combo_fruit_filter.count() > 0:
            prefer_fruit = self.session.semantic_map.get("fruit")
            last_fruit = self._settings.value("fruit_sem_label", "", type=str)
            items = [self.combo_fruit_filter.itemText(i) for i in range(self.combo_fruit_filter.count())]
            if prefer_fruit is not None and str(int(prefer_fruit)) in items:
                self.combo_fruit_filter.setCurrentText(str(int(prefer_fruit)))
            elif prefer_fruit is None and "-1" in items:
                self.combo_fruit_filter.setCurrentText("-1")
            elif last_fruit and last_fruit in items:
                self.combo_fruit_filter.setCurrentText(last_fruit)
            else:
                self.combo_fruit_filter.setCurrentIndex(0)

        self._update_semantic_map_from_ui()
        self._refresh_anno_semantic_options()
    def on_sem_filter_changed(self):
        if self.session.cloud is None:
            return
        text = self.combo_sem_filter.currentText()
        if not text:
            return
        self._settings.setValue("leaf_sem_label", text)
        self._update_semantic_map_from_ui()
        self._refresh_anno_semantic_options()
        if self._get_annotation_semantic_key() == "leaf":
            sem_val = int(text)
            if sem_val < 0:
                self.combo_inst.clear()
            else:
                self._refresh_instance_list(target_sem=sem_val, keep_current=False)
        if self.annotating and self.combo_inst.count() > 0:
            self.on_inst_changed()

    def on_stem_sem_changed(self):
        if self.session.cloud is None:
            return
        text = self.combo_stem_filter.currentText()
        if not text:
            return
        self._settings.setValue("stem_sem_label", text)
        self._update_semantic_map_from_ui()
        self._refresh_anno_semantic_options()
        if self._get_annotation_semantic_key() == "stem":
            self._refresh_instance_list_for_annotation()

    def on_flower_sem_changed(self):
        if self.session.cloud is None:
            return
        text = self.combo_flower_filter.currentText()
        if not text:
            return
        self._settings.setValue("flower_sem_label", text)
        self._update_semantic_map_from_ui()
        self._refresh_anno_semantic_options()
        if self._get_annotation_semantic_key() == "flower":
            self._refresh_instance_list_for_annotation()

    def on_fruit_sem_changed(self):
        if self.session.cloud is None:
            return
        text = self.combo_fruit_filter.currentText()
        if not text:
            return
        self._settings.setValue("fruit_sem_label", text)
        self._update_semantic_map_from_ui()
        self._refresh_anno_semantic_options()
        if self._get_annotation_semantic_key() == "fruit":
            self._refresh_instance_list_for_annotation()
    def _set_plant_type(self, plant_type: str, update_status: bool = True, save_settings: bool = True):
        if plant_type not in self.plant_type_actions:
            return
        self.session.plant_type = plant_type
        for t, act in self.plant_type_actions.items():
            act.blockSignals(True)
            act.setChecked(t == plant_type)
            act.blockSignals(False)
        if save_settings:
            self._settings.setValue("plant_type", plant_type)
        if update_status:
            self._update_status(f"已选择植物类型：{plant_type}")

    def on_plant_type_selected(self):
        act = self.sender()
        if act is None:
            return
        plant_type = act.text()
        self._set_plant_type(plant_type)

    def _clear_layout(self, layout: QtWidgets.QLayout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
            elif item.layout() is not None:
                self._clear_layout(item.layout())
            del item

    def _set_semantic_legend(self, labels):
        return

    def on_instance_meta_changed(self):
        if (not self.annotating) or (self.session.current_inst_id is None):
            return
        remark = self.text_remark.toPlainText()
        label_desc = self.combo_label_desc.currentText()
        if label_desc == "未选择":
            label_desc = ""
        self.session.set_instance_meta(self.session.current_inst_id, remark, label_desc)

    def _apply_left_panel_sizes(self, panel_width: int):
        inner_width = max(PANEL_LIST_MIN_WIDTH, panel_width - PANEL_INNER_PADDING)
        row_combo_width = max(PANEL_COMBO_MIN_WIDTH, panel_width - PANEL_ROW_LABEL_SPACE)
        button_width = max(PANEL_BUTTON_MIN_WIDTH, panel_width - PANEL_INNER_PADDING)

        for combo in [
            self.combo_anno_semantic, self.combo_sem_filter, self.combo_stem_filter,
            self.combo_flower_filter, self.combo_fruit_filter, self.combo_inst
        ]:
            combo.setFixedWidth(row_combo_width)

        self.combo_label_desc.setFixedWidth(inner_width)
        self.text_remark.setFixedWidth(inner_width)

        for lst in [
            self.list_base, self.list_tip, self.list_ctrl, self.list_width, self.list_width_ctrl
        ]:
            lst.setFixedWidth(inner_width)

        for btn in [
            self.btn_toggle_base, self.btn_toggle_tip, self.btn_toggle_ctrl,
            self.btn_toggle_width, self.btn_toggle_width_ctrl,
            self.btn_recommend_length, self.btn_generate_length,
            self.btn_recommend_width, self.btn_generate_width,
            self.btn_delete, self.btn_rename_ctrl, self.btn_rename_width_ctrl
        ]:
            btn.setFixedWidth(button_width)

    def _update_buttons(self):
        in_anno = self.annotating
        in_leaf_anno = self.annotating and self.annotate_semantic == "leaf"
        sem_key = self.annotate_semantic
        if not self.annotating:
            sem_key = self._get_annotation_semantic_key()
        in_stem_sem = sem_key == "stem"
        in_flower_fruit_sem = sem_key in ["flower", "fruit"]
        for b in [
            self.btn_toggle_base, self.btn_toggle_tip, self.btn_toggle_ctrl, self.btn_toggle_width,
            self.btn_recommend_length, self.btn_generate_length,
            self.btn_recommend_width, self.btn_generate_width,
            self.btn_delete, self.btn_rename_ctrl,
            self.btn_toggle_width_ctrl, self.btn_rename_width_ctrl,
        ]:
            b.setEnabled(in_leaf_anno)
        self.meta_group.setEnabled(in_anno)
        if hasattr(self, "points_group"):
            self.points_group.setVisible(in_leaf_anno)
        if hasattr(self, "phenotype_group"):
            self.phenotype_group.setVisible(not in_anno)
        if hasattr(self, "bbox_info"):
            self._update_bbox_info()
        if hasattr(self, "panel_widget"):
            width = PANEL_WIDTH_LEAF_ANNOTATE if in_leaf_anno else PANEL_WIDTH_DEFAULT
            self.panel_widget.setFixedWidth(width)
            self._apply_left_panel_sizes(width)

        has_cloud = self.session.cloud is not None
        can_export = has_cloud
        self.btn_export.setEnabled(can_export)
        self.btn_back_browse.setEnabled(has_cloud)
        self.btn_start_anno.setEnabled(has_cloud and self.combo_inst.count() > 0)

        self.act_export_dir.setEnabled(True)
        self.act_back_browse.setEnabled(has_cloud)
        self.act_start_anno.setEnabled(has_cloud and self.combo_inst.count() > 0)
        self.combo_inst.setEnabled(has_cloud and self.combo_inst.count() > 0)
        self.combo_anno_semantic.setEnabled(has_cloud and not self.annotating)
        self.combo_sem_filter.setEnabled(has_cloud and not self.annotating)
        self.combo_stem_filter.setEnabled(has_cloud and not self.annotating)
        self.combo_flower_filter.setEnabled(has_cloud and not self.annotating)
        self.combo_fruit_filter.setEnabled(has_cloud and not self.annotating)
        for a in [self.act_view_rgb, self.act_view_sem, self.act_view_inst, self.act_view_label]:
            a.setEnabled(has_cloud)

        self.act_save_labels.setEnabled(can_export)
        for a in [
            self.act_recommend_length, self.act_generate_length,
            self.act_recommend_width, self.act_generate_width,
            self.act_smooth_paths, self.act_export_labeled,
        ]:
            a.setEnabled(in_leaf_anno)
        self.act_compute_stem.setEnabled(has_cloud and in_stem_sem)
        self.act_compute_stem_length.setEnabled(has_cloud and in_stem_sem)
        self.act_compute_flower_fruit.setEnabled(has_cloud and in_flower_fruit_sem)
        self.btn_toggle_measure.setEnabled(has_cloud)
        stem_tools_visible = in_anno and self.annotate_semantic == "stem"
        if hasattr(self, "btn_toggle_stem_cyl"):
            self.btn_toggle_stem_cyl.setVisible(stem_tools_visible)
            self.btn_toggle_stem_cyl.setEnabled(has_cloud and stem_tools_visible)
        if hasattr(self, "btn_toggle_stem_path"):
            self.btn_toggle_stem_path.setVisible(stem_tools_visible)
            self.btn_toggle_stem_path.setEnabled(has_cloud and stem_tools_visible)
        if not stem_tools_visible:
            changed = False
            if hasattr(self, "btn_toggle_stem_cyl") and self.btn_toggle_stem_cyl.isChecked():
                self.btn_toggle_stem_cyl.blockSignals(True)
                self.btn_toggle_stem_cyl.setChecked(False)
                self.btn_toggle_stem_cyl.blockSignals(False)
                self._clear_stem_cylinders()
                changed = True
            if hasattr(self, "btn_toggle_stem_path") and self.btn_toggle_stem_path.isChecked():
                self.btn_toggle_stem_path.blockSignals(True)
                self.btn_toggle_stem_path.setChecked(False)
                self.btn_toggle_stem_path.blockSignals(False)
                self._clear_stem_length_paths()
                changed = True
            if changed:
                self._refresh_scene()
                self.plotter.render()
        for b in [
            self.btn_view_front, self.btn_view_side, self.btn_view_top,
            self.btn_pick_view_center, self.btn_toggle_aabb, self.btn_toggle_obb
        ]:
            b.setEnabled(has_cloud)

    # ----------------------------
    # vtk picking (Shift+Left Click)
    # ----------------------------
    def _install_vtk_pick_observer(self):
        if self._pick_observer_id is not None:
            return

        iren = self.plotter.interactor

        def _on_left_press(obj, event):
            if self.pick_mode == self.MODE_NONE and not self._pick_view_center:
                return
            if self.pick_mode == self.MODE_MEASURE or self._pick_view_center:
                if self.session.cloud is None:
                    return
            else:
                if (not self.annotating) or (self.session.ds is None):
                    return
            if iren.GetShiftKey() == 0:
                return

            x, y = iren.GetEventPosition()
            renderer = self.plotter.renderer

            self._vtk_picker.PickFromListOn()
            self._vtk_picker.InitializePickList()
            if self.pick_mode == self.MODE_MEASURE or self._pick_view_center:
                if self.annotating and self._actor_cloud_inst is not None:
                    try:
                        self._vtk_picker.AddPickList(self._actor_cloud_inst)
                    except Exception:
                        pass
                elif self._actor_cloud_full is not None:
                    try:
                        self._vtk_picker.AddPickList(self._actor_cloud_full)
                    except Exception:
                        pass
            elif self._actor_cloud_inst is not None:
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
            if self._pick_view_center:
                self._pick_view_center = False
                self._set_view_center(pick_pos)
            else:
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

        mode = self.combo_view.currentText()
        if mode == "RGB" and self.session.has_rgb():
            poly["rgb"] = self.session.get_full_rgb()
        elif mode == "语义":
            poly["rgb"] = colors_from_labels(self.session.get_full_sem())
        elif mode == "实例":
            poly["rgb"] = colors_from_labels(self.session.get_full_inst())
        elif mode == "表型标签":
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
            self.A_VIEW_CENTER
        ]:
            self._remove_actor(nm)

        self._update_measure_display()
        self.plotter.render()

    def _show_stem_only_scene(self):
        stem_pts = self._get_stem_points()
        if stem_pts is None or len(stem_pts) == 0:
            QMessageBox.information(self, "提示", "没有可显示的茎点云，请检查茎语义选择。")
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
            self.A_VIEW_CENTER, self.A_AABB, self.A_OBB
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
            self.A_VIEW_CENTER, self.A_AABB, self.A_OBB
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
        if self.combo_view.currentText() == "表型标签" and self.session.point_labels is not None:
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
        self._update_status(f"测距结果：{dist:.6f}")

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
        elif mode == self.MODE_WIDTH_CTRL:
            self.temp_width_ctrl_indices = []
            self.btn_toggle_width_ctrl.setText("叶宽控制点选择：开启")
        elif mode == self.MODE_MEASURE:
            self.temp_measure_p1 = None
            self.temp_measure_p2 = None
            self.btn_toggle_measure.setText("测距：开启")

        self._update_markers_temp()
        self._update_labels_temp()
        self.plotter.render()

        if mode == self.MODE_WIDTH:
            self._update_status("叶宽点选择：Shift+左键依次选择 W1/W2；关闭后保存。")
        elif mode == self.MODE_WIDTH_CTRL:
            self._update_status("叶宽控制点：Shift+左键添加；关闭后保存。")
        elif mode == self.MODE_MEASURE:
            self._update_status("测距：Shift+左键选择两个点，显示连线与距离。")
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
            if self.pick_mode == self.MODE_WIDTH_CTRL and len(self.temp_width_ctrl_indices) > 0:
                self.session.extend_width_ctrl(self.temp_width_ctrl_indices)

        if self.pick_mode == self.MODE_BASE:
            self.temp_base_idx = None
        elif self.pick_mode == self.MODE_TIP:
            self.temp_tip_idx = None
        elif self.pick_mode == self.MODE_CTRL:
            self.temp_ctrl_indices = []
        elif self.pick_mode == self.MODE_WIDTH:
            self.temp_w1_idx = None
            self.temp_w2_idx = None
        elif self.pick_mode == self.MODE_WIDTH_CTRL:
            self.temp_width_ctrl_indices = []
        elif self.pick_mode == self.MODE_MEASURE:
            self.temp_measure_p1 = None
            self.temp_measure_p2 = None

        self.pick_mode = self.MODE_NONE

        for b in [
            self.btn_toggle_base, self.btn_toggle_tip, self.btn_toggle_ctrl,
            self.btn_toggle_width, self.btn_toggle_width_ctrl, self.btn_toggle_measure
        ]:
            b.blockSignals(True)
            b.setChecked(False)
            b.blockSignals(False)

        self.btn_toggle_base.setText("叶基选择：关闭")
        self.btn_toggle_tip.setText("叶尖选择：关闭")
        self.btn_toggle_ctrl.setText("控制点选择：关闭")
        self.btn_toggle_width.setText("叶宽点选择：关闭 (W1/W2)")
        self.btn_toggle_width_ctrl.setText("叶宽控制点选择：关闭")
        self.btn_toggle_measure.setText("测距：关闭")

        self._update_markers_saved()
        self._update_markers_temp()
        self._update_labels_saved()
        self._update_labels_temp()
        self._update_lines()
        self._update_measure_display()
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

    def on_toggle_width_ctrl(self, checked: bool):
        if not self.annotating:
            return
        if checked:
            self._close_other_toggles(except_mode=self.MODE_WIDTH_CTRL)
            self._enter_mode(self.MODE_WIDTH_CTRL)
        else:
            if self.pick_mode == self.MODE_WIDTH_CTRL:
                self._exit_current_mode(commit=True)
                self._invalidate_results_after_point_change()
                self._update_status("叶宽控制点选择关闭：已保存。")

    def on_toggle_measure(self, checked: bool):
        if self.session.cloud is None:
            return
        self.btn_toggle_measure.blockSignals(True)
        self.btn_toggle_measure.setChecked(checked)
        self.btn_toggle_measure.blockSignals(False)
        if checked:
            self._close_other_toggles(except_mode=self.MODE_MEASURE)
            self._enter_mode(self.MODE_MEASURE)
        else:
            if self.pick_mode == self.MODE_MEASURE:
                self._exit_current_mode(commit=False)
                self._update_status("测距已关闭。")

    def _close_other_toggles(self, except_mode: str):
        mapping = {
            self.MODE_BASE: self.btn_toggle_base,
            self.MODE_TIP: self.btn_toggle_tip,
            self.MODE_CTRL: self.btn_toggle_ctrl,
            self.MODE_WIDTH: self.btn_toggle_width,
            self.MODE_WIDTH_CTRL: self.btn_toggle_width_ctrl,
            self.MODE_MEASURE: self.btn_toggle_measure,
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
        if self.pick_mode != self.MODE_MEASURE and self.session.ds is None:
            return

        p = np.array(point_xyz, dtype=np.float64)

        if self.pick_mode == self.MODE_BASE:
            idx = self.session.snap_to_ds_index(p)
            self.temp_base_idx = idx
        elif self.pick_mode == self.MODE_TIP:
            idx = self.session.snap_to_ds_index(p)
            self.temp_tip_idx = idx
        elif self.pick_mode == self.MODE_CTRL:
            idx = self.session.snap_to_ds_index(p)
            self.temp_ctrl_indices.append(idx)
        elif self.pick_mode == self.MODE_WIDTH:
            idx = self.session.snap_to_ds_index(p)
            if self.temp_w1_idx is None:
                self.temp_w1_idx = idx
            elif self.temp_w2_idx is None:
                self.temp_w2_idx = idx
            else:
                self.temp_w2_idx = idx
        elif self.pick_mode == self.MODE_WIDTH_CTRL:
            idx = self.session.snap_to_ds_index(p)
            self.temp_width_ctrl_indices.append(idx)
        elif self.pick_mode == self.MODE_MEASURE:
            if self.temp_measure_p1 is None:
                self.temp_measure_p1 = p.copy()
            elif self.temp_measure_p2 is None:
                self.temp_measure_p2 = p.copy()
            else:
                self.temp_measure_p1 = p.copy()
                self.temp_measure_p2 = None
            self._update_measure_display()

        self._update_markers_temp()
        self._update_labels_temp()
        self.plotter.render()

    # ----------------------------
    # delete actions
    # ----------------------------
    def on_ctrl_context_menu(self, pos):
        it = self.list_ctrl.itemAt(pos)
        if it is None:
            return
        if not it.isSelected():
            self.list_ctrl.selectionModel().select(
                self.list_ctrl.indexFromItem(it),
                QtCore.QItemSelectionModel.ClearAndSelect
            )
        menu = QtWidgets.QMenu(self)
        act_delete = menu.addAction("删除")
        act_rename = menu.addAction("修改顺序")
        action = menu.exec_(self.list_ctrl.mapToGlobal(pos))
        if action == act_delete:
            self._delete_selected_ctrl_items()
        elif action == act_rename:
            if len(self.list_ctrl.selectedItems()) > 1:
                QMessageBox.information(self, "提示", "修改顺序仅支持单个控制点。")
                return
            self.on_rename_ctrl()

    def on_width_ctrl_context_menu(self, pos):
        it = self.list_width_ctrl.itemAt(pos)
        if it is None:
            return
        if not it.isSelected():
            self.list_width_ctrl.selectionModel().select(
                self.list_width_ctrl.indexFromItem(it),
                QtCore.QItemSelectionModel.ClearAndSelect
            )
        menu = QtWidgets.QMenu(self)
        act_delete = menu.addAction("删除")
        act_rename = menu.addAction("修改顺序")
        action = menu.exec_(self.list_width_ctrl.mapToGlobal(pos))
        if action == act_delete:
            self._delete_selected_width_ctrl_items()
        elif action == act_rename:
            if len(self.list_width_ctrl.selectedItems()) > 1:
                QMessageBox.information(self, "提示", "修改顺序仅支持单个控制点。")
                return
            self.on_rename_width_ctrl()

    def on_base_context_menu(self, pos):
        it = self.list_base.itemAt(pos)
        if it is None:
            return
        self.list_base.setCurrentItem(it)
        menu = QtWidgets.QMenu(self)
        act_delete = menu.addAction("删除")
        action = menu.exec_(self.list_base.mapToGlobal(pos))
        if action == act_delete:
            self.on_delete_base()

    def on_tip_context_menu(self, pos):
        it = self.list_tip.itemAt(pos)
        if it is None:
            return
        self.list_tip.setCurrentItem(it)
        menu = QtWidgets.QMenu(self)
        act_delete = menu.addAction("删除")
        action = menu.exec_(self.list_tip.mapToGlobal(pos))
        if action == act_delete:
            self.on_delete_tip()

    def on_width_context_menu(self, pos):
        it = self.list_width.itemAt(pos)
        if it is None:
            return
        self.list_width.setCurrentItem(it)
        menu = QtWidgets.QMenu(self)
        act_delete = menu.addAction("删除")
        action = menu.exec_(self.list_width.mapToGlobal(pos))
        if action == act_delete:
            self.on_delete_width()

    def _delete_selected_ctrl_items(self):
        ctrl_items = self.list_ctrl.selectedItems()
        if not ctrl_items:
            return
        indices = []
        for it in ctrl_items:
            info = it.data(0x0100)
            if info and info[0] == "ctrl":
                indices.append(int(info[1]))
        if not indices:
            return
        for idx in sorted(set(indices), reverse=True):
            if 0 <= idx < len(self.session.ctrl_indices):
                self.session.ctrl_indices.pop(idx)
                if idx < len(self.session.ctrl_ids):
                    self.session.ctrl_ids.pop(idx)
        self._invalidate_results_after_point_change()
        self._update_markers_saved()
        self._update_labels_saved()
        self.plotter.render()
        self._refresh_point_lists()
        self._maybe_recommend_width_and_refresh()
        self._update_status("已删除选中的叶长控制点。")

    def _delete_selected_width_ctrl_items(self):
        wctrl_items = self.list_width_ctrl.selectedItems()
        if not wctrl_items:
            return
        indices = []
        for it in wctrl_items:
            info = it.data(0x0100)
            if info and info[0] == "wctrl":
                indices.append(int(info[1]))
        if not indices:
            return
        for idx in sorted(set(indices), reverse=True):
            if 0 <= idx < len(self.session.width_ctrl_indices):
                self.session.width_ctrl_indices.pop(idx)
                if idx < len(self.session.width_ctrl_ids):
                    self.session.width_ctrl_ids.pop(idx)
        self._invalidate_results_after_point_change()
        self._update_markers_saved()
        self._update_labels_saved()
        self.plotter.render()
        self._refresh_point_lists()
        self._update_status("已删除选中的叶宽控制点。")

    def on_delete_selected(self):
        if self.list_base.currentItem() is not None:
            self.on_delete_base()
            return
        if self.list_tip.currentItem() is not None:
            self.on_delete_tip()
            return
        ctrl_items = self.list_ctrl.selectedItems()
        if ctrl_items:
            self._delete_selected_ctrl_items()
            return
        if self.list_width.currentItem() is not None:
            self.on_delete_width()
            return
        wctrl_items = self.list_width_ctrl.selectedItems()
        if wctrl_items:
            self._delete_selected_width_ctrl_items()
            return
        QMessageBox.information(self, "提示", "请先在左侧列表中选择要删除的标记。")

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
            if idx_in_list < len(self.session.ctrl_ids):
                self.session.ctrl_ids.pop(idx_in_list)

        self._invalidate_results_after_point_change()
        self._update_markers_saved()
        self._update_labels_saved()
        self.plotter.render()
        self._refresh_point_lists()
        self._maybe_recommend_width_and_refresh()
        self._update_status("已删除选中的控制点。")

    def on_rename_ctrl(self):
        it = self.list_ctrl.currentItem()
        if it is None:
            return
        info = it.data(0x0100)
        if not info or info[0] != "ctrl":
            return
        idx_in_list = int(info[1])
        if idx_in_list >= len(self.session.ctrl_ids):
            return

        cur_id = self.session.ctrl_ids[idx_in_list]
        new_id, ok = QtWidgets.QInputDialog.getInt(
            self, "重命名叶长控制点",
            "输入新的编号（正整数）",
            value=int(cur_id), min=1, max=100000
        )
        if not ok:
            return
        new_id = int(new_id)
        if new_id in self.session.ctrl_ids and new_id != cur_id:
            QMessageBox.information(self, "提示", "编号已存在，请选择其他编号。")
            return

        self.session.ctrl_ids[idx_in_list] = new_id

        self._invalidate_results_after_point_change()
        self._update_markers_saved()
        self._update_labels_saved()
        self.plotter.render()
        self._refresh_point_lists()
        self._maybe_recommend_width_and_refresh()
        self._update_status("已修改叶长控制点编号。")

    def on_delete_width_ctrl(self):
        it = self.list_width_ctrl.currentItem()
        if it is None:
            return
        info = it.data(0x0100)
        if not info or info[0] != "wctrl":
            return
        idx_in_list = int(info[1])
        if 0 <= idx_in_list < len(self.session.width_ctrl_indices):
            self.session.width_ctrl_indices.pop(idx_in_list)
            if idx_in_list < len(self.session.width_ctrl_ids):
                self.session.width_ctrl_ids.pop(idx_in_list)

        self._invalidate_results_after_point_change()
        self._update_markers_saved()
        self._update_labels_saved()
        self.plotter.render()
        self._refresh_point_lists()
        self._update_status("已删除选中的叶宽控制点。")

    def on_rename_width_ctrl(self):
        it = self.list_width_ctrl.currentItem()
        if it is None:
            return
        info = it.data(0x0100)
        if not info or info[0] != "wctrl":
            return
        idx_in_list = int(info[1])
        if idx_in_list >= len(self.session.width_ctrl_ids):
            return

        cur_id = self.session.width_ctrl_ids[idx_in_list]
        new_id, ok = QtWidgets.QInputDialog.getInt(
            self, "重命名叶宽控制点",
            "输入新的编号（正整数）",
            value=int(cur_id), min=1, max=100000
        )
        if not ok:
            return
        new_id = int(new_id)
        if new_id in self.session.width_ctrl_ids and new_id != cur_id:
            QMessageBox.information(self, "提示", "编号已存在，请选择其他编号。")
            return

        self.session.width_ctrl_ids[idx_in_list] = new_id

        self._invalidate_results_after_point_change()
        self._update_markers_saved()
        self._update_labels_saved()
        self.plotter.render()
        self._refresh_point_lists()
        self._update_status("已修改叶宽控制点编号。")

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
        self._refresh_scene()

    def _set_view_mode(self, mode: str):
        self._refresh_scene(mode=mode)

    def _refresh_scene(self, mode: str = None):
        if mode is not None and self.combo_view.currentText() != mode:
            self.combo_view.blockSignals(True)
            self.combo_view.setCurrentText(mode)
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
            return
        if not self.annotating:
            self._show_browse_scene()
        else:
            self._show_annotate_scene()
        mode_text = self.combo_view.currentText()
        if mode_text == "RGB":
            self.act_view_rgb.setChecked(True)
        elif mode_text == "语义":
            self.act_view_sem.setChecked(True)
        elif mode_text == "实例":
            self.act_view_inst.setChecked(True)
        elif mode_text == "表型标签":
            self.act_view_label.setChecked(True)
        self._refresh_bbox_actors()
        self._update_view_legend()

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

        mode = self.combo_view.currentText()
        labels = []
        if mode == "语义":
            if not self.annotating:
                labels = np.unique(self.session.get_full_sem().astype(np.int64)).tolist()
            else:
                labels = []
        elif mode == "实例":
            labels = np.unique(self.session.get_full_inst().astype(np.int64)).tolist()
        elif mode == "表型标签":
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
            empty = QtWidgets.QLabel("无")
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

    def _update_phenotype_table(self):
        if not hasattr(self, "table_phenotype"):
            return
        self.table_phenotype.setRowCount(0)
        if self.session.cloud is None:
            return

        sem_map = self._get_instance_sem_map()
        label_to_name = {}
        for key, name in [("leaf", "叶"), ("stem", "茎"), ("flower", "花"), ("fruit", "果")]:
            v = self.session.semantic_map.get(key)
            if v is not None:
                label_to_name[int(v)] = name

        rows = []
        for inst_id, ann in self.session.annotations.items():
            sem_label = sem_map.get(int(inst_id), None)
            sem_text = "-" if sem_label is None else label_to_name.get(int(sem_label), str(int(sem_label)))
            if "length" in ann and ann.get("length") is not None:
                rows.append((inst_id, sem_text, "叶长", f"{float(ann['length']):.3f}"))
            if "width_path_length" in ann and ann.get("width_path_length") is not None:
                rows.append((inst_id, sem_text, "叶宽", f"{float(ann['width_path_length']):.3f}"))
            if "stem_diameter" in ann and ann.get("stem_diameter") is not None:
                rows.append((inst_id, sem_text, "茎粗", f"{float(ann['stem_diameter']):.3f}"))
            if "stem_length" in ann and ann.get("stem_length") is not None:
                rows.append((inst_id, sem_text, "茎长", f"{float(ann['stem_length']):.3f}"))
            if "flower_obb" in ann and isinstance(ann.get("flower_obb"), dict):
                lengths = ann["flower_obb"].get("lengths")
                if lengths is not None and len(lengths) == 3:
                    rows.append((inst_id, sem_text, "花OBB", f"{lengths[0]:.3f},{lengths[1]:.3f},{lengths[2]:.3f}"))
            if "fruit_obb" in ann and isinstance(ann.get("fruit_obb"), dict):
                lengths = ann["fruit_obb"].get("lengths")
                if lengths is not None and len(lengths) == 3:
                    rows.append((inst_id, sem_text, "果OBB", f"{lengths[0]:.3f},{lengths[1]:.3f},{lengths[2]:.3f}"))

        self.table_phenotype.setRowCount(len(rows))
        for r, (inst_id, sem_text, name, value) in enumerate(rows):
            for c, val in enumerate([str(int(inst_id)), str(sem_text), str(name), str(value)]):
                it = QtWidgets.QTableWidgetItem(val)
                it.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
                self.table_phenotype.setItem(r, c, it)

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
            self.bbox_info.setText("AABB：-\nOBB：-")
            return
        pts = self._get_bbox_points()
        if pts is None or len(pts) == 0:
            self.bbox_info.setText("AABB：-\nOBB：-")
            return

        aabb_text = "AABB：-"
        if self.btn_toggle_aabb.isChecked():
            mins = pts.min(axis=0)
            maxs = pts.max(axis=0)
            lengths = (maxs - mins).tolist()
            aabb_text = f"AABB：L={lengths[0]:.3f} W={lengths[1]:.3f} H={lengths[2]:.3f}"

        obb_text = "OBB：-"
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

        try:
            self.session.load(path)
        except Exception as e:
            QMessageBox.critical(self, "读取失败", str(e))
            return

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
            self.session.compute_stem_instance(inst_id)
            self._update_buttons()
            self._refresh_scene()
            self._refresh_point_lists()
            self._refresh_instance_meta_ui()
            self._update_instance_sem_label()
            self._update_view_legend()
            self._update_phenotype_table()
            self._update_status(f"茎标注：inst_id={inst_id}，已计算茎粗/茎长。")
        elif key in ["flower", "fruit"]:
            self.session.compute_obb_instance(inst_id, key)
            self._update_buttons()
            self._refresh_scene()
            self._refresh_point_lists()
            self._refresh_instance_meta_ui()
            self._update_instance_sem_label()
            self._update_view_legend()
            self._update_phenotype_table()
            label = "花" if key == "flower" else "果"
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
                self.session.compute_stem_instance(inst_id)
                self._update_buttons()
                self._refresh_scene()
                self._refresh_point_lists()
                self._refresh_instance_meta_ui()
                self._update_instance_sem_label()
                self._update_view_legend()
                self._update_phenotype_table()
                self._update_status(f"茎标注：inst_id={inst_id}，已更新茎粗/茎长。")
            elif key in ["flower", "fruit"]:
                self.session.compute_obb_instance(inst_id, key)
                self._update_buttons()
                self._refresh_scene()
                self._refresh_point_lists()
                self._refresh_instance_meta_ui()
                self._update_instance_sem_label()
                self._update_view_legend()
                self._update_phenotype_table()
                label = "花" if key == "flower" else "果"
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
        self.session.point_labels = self.session.compute_point_labels(self.session.params.label_radius)
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
        try:
            count = self.session.compute_stem_diameter_structures()
        except Exception as e:
            QMessageBox.critical(self, "计算失败", str(e))
            return
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
        try:
            count = self.session.compute_stem_length_structures()
        except Exception as e:
            QMessageBox.critical(self, "计算失败", str(e))
            return
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
        try:
            counts = self.session.compute_flower_fruit_obb()
        except Exception as e:
            QMessageBox.critical(self, "计算失败", str(e))
            return
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
            self.session.point_labels = self.session.compute_point_labels(self.session.params.label_radius)
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
        QMessageBox.information(
            self,
            "快捷键/说明",
            "拾取：Shift + 左键\n"
            "切换：先开启对应拾取模式再点击点云\n"
            "叶宽：W1/W2 为端点，WC 为叶宽控制点\n"
            "保存标注：Ctrl + S",
        )

    def on_show_about(self):
        QMessageBox.information(
            self,
            "关于",
            "版本：v0.1.0\n"
            "开发人：杨鑫\n"
            "邮件：yangxinnc@163.com\n"
            "单位：沈阳农业大学 信息与电气工程学院",
        )
