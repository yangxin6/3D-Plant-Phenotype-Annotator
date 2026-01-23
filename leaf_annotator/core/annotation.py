import json
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from types import SimpleNamespace

from scipy.spatial import cKDTree, ConvexHull
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from core.io import PointCloudIO
from core.schema import CloudSchema, CloudParser, ParsedCloud
from core.sampling import VoxelSampler, Downsampled
from core.width import WidthEstimator  # ✅ 恢复：用于推荐叶宽点
from core.centerline import CenterlineExtractor


from core.annotation_parts.params import AnnotationParams
from core.annotation_parts.utils import (
    _polyline_length,
    _resample_polyline_step,
    _smooth_polyline_window,
    _build_knn_graph,
    _build_radius_graph,
    _shortest_path_indices,
)
from core.annotation_parts.base import BaseMixin
from core.annotation_parts.io import IoMixin
from core.annotation_parts.semantic import SemanticMixin
from core.annotation_parts.stem import StemMixin
from core.annotation_parts.obb import ObbMixin
from core.annotation_parts.instance import InstanceMixin
from core.annotation_parts.leaf import LeafMixin
from core.annotation_parts.cache import CacheMixin

class LeafAnnotationSession(BaseMixin, IoMixin, SemanticMixin, StemMixin, ObbMixin, InstanceMixin, LeafMixin, CacheMixin):
    def __init__(self, params: Optional[AnnotationParams] = None, schema: Optional[CloudSchema] = None):
        self.params = params or AnnotationParams()
        self.schema = schema or CloudSchema()

        self.file_path: Optional[str] = None
        self.cloud: Optional[ParsedCloud] = None
        self.plant_type: str = "玉米"
        self.semantic_map: Dict[str, Optional[int]] = {"leaf": None, "stem": None, "flower": None, "fruit": None}

        self.current_inst_id: Optional[int] = None
        self.leaf_pts: Optional[np.ndarray] = None
        self.leaf_global_idx: Optional[np.ndarray] = None

        self.ds: Optional[Downsampled] = None
        self.ds_tree: Optional[cKDTree] = None

        # saved picks (ds index)
        self.base_idx: Optional[int] = None
        self.tip_idx: Optional[int] = None
        self.ctrl_indices: List[int] = []
        self.ctrl_ids: List[int] = []
        self._next_ctrl_id: int = 10
        self.width_ctrl_indices: List[int] = []
        self.width_ctrl_ids: List[int] = []
        self._next_width_ctrl_id: int = 10

        # width endpoints (ds index)
        self.width_w1_idx: Optional[int] = None
        self.width_w2_idx: Optional[int] = None

        # 推荐端点只自动给一次；用户删除后不再自动冒出来
        self._width_recommended_once: bool = False

        # results
        self.centerline_result = None              # 有 smooth_points / length
        self.centerline_source: Optional[str] = None
        self.width_path_points: Optional[np.ndarray] = None
        self.width_path_length: Optional[float] = None
        self.width_path_indices: Optional[np.ndarray] = None
        self.point_labels: Optional[np.ndarray] = None
        self.full_point_labels: Optional[np.ndarray] = None

        self.annotations: Dict[int, Dict[str, Any]] = {}
        self.instance_meta: Dict[int, Dict[str, str]] = {}
        self.use_cached_results: bool = True

