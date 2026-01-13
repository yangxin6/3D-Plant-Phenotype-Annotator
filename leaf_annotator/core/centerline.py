import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from core.graph_path import KNNGraphPath


@dataclass
class CenterlineResult:
    raw_points: np.ndarray
    smooth_points: np.ndarray
    length: float
    path_indices: np.ndarray


class CenterlineExtractor:
    def __init__(self, k: int = 25, smooth_win: int = 9):
        self.graph = KNNGraphPath(k=k)
        self.smooth_win = max(3, int(smooth_win) | 1)  # odd

    @staticmethod
    def smooth_polyline(P: np.ndarray, win: int) -> np.ndarray:
        if len(P) < win:
            return P.copy()
        out = P.copy()
        half = win // 2
        for i in range(len(P)):
            a = max(0, i - half)
            b = min(len(P), i + half + 1)
            out[i] = P[a:b].mean(axis=0)
        return out

    @staticmethod
    def arclength(P: np.ndarray) -> float:
        if len(P) < 2:
            return 0.0
        return float(np.linalg.norm(P[1:] - P[:-1], axis=1).sum())

    @staticmethod
    def resample_by_step(P: np.ndarray, step: float) -> np.ndarray:
        if len(P) < 2:
            return P.copy()
        seg = np.linalg.norm(P[1:] - P[:-1], axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        L = s[-1]
        if L <= 1e-12:
            return P[:1].copy()
        m = max(2, int(np.floor(L / step)) + 1)
        s_new = np.linspace(0, L, m)

        out = []
        j = 0
        for sn in s_new:
            while j < len(s) - 2 and s[j + 1] < sn:
                j += 1
            denom = (s[j + 1] - s[j])
            t = 0.0 if abs(denom) < 1e-12 else (sn - s[j]) / denom
            out.append((1 - t) * P[j] + t * P[j + 1])
        return np.array(out)

    def extract(
        self,
        ds_points: np.ndarray,
        base_idx: int,
        tip_idx: int,
        ctrl_indices: Optional[List[int]] = None
    ) -> CenterlineResult:
        ctrl_indices = ctrl_indices or []
        chain = [base_idx] + list(ctrl_indices) + [tip_idx]

        W = self.graph.build_graph(ds_points)

        path_all: List[int] = []
        for a, b in zip(chain[:-1], chain[1:]):
            pidx = self.graph.shortest_path_indices(W, a, b)
            if path_all:
                pidx = pidx[1:]  # avoid duplicate joint
            path_all.extend(pidx.tolist())

        path_all = np.array(path_all, dtype=int)
        raw = ds_points[path_all]
        smooth = self.smooth_polyline(raw, win=self.smooth_win)
        L = self.arclength(smooth)

        return CenterlineResult(
            raw_points=raw,
            smooth_points=smooth,
            length=L,
            path_indices=path_all
        )
