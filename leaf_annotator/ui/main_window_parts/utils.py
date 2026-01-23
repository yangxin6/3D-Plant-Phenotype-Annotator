# ui/main_window_parts/utils.py
import numpy as np
import pyvista as pv


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
