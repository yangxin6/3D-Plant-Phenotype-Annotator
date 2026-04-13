import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from core.centerline import CenterlineExtractor


@dataclass
class WidthItem:
    s_index: int
    p: np.ndarray
    width: float
    pL: np.ndarray
    pR: np.ndarray
    slice_n: int
    tangent: np.ndarray
    perp_abs_dot: float
    score: float


@dataclass
class WidthResult:
    profile: List[WidthItem]
    max_item: Optional[WidthItem]


class WidthEstimator:
    def __init__(
        self,
        step: float = 0.01,
        slab_half: float = 0.005,
        radius: float = 0.10,
        min_slice_pts: int = 60,
        qlo: float = 0.05,
        qhi: float = 0.95,
        perp_lambda: float = 0.2,
    ):
        self.step = step
        self.slab_half = slab_half
        self.radius = radius
        self.min_slice_pts = min_slice_pts
        self.qlo = qlo
        self.qhi = qhi
        self.perp_lambda = perp_lambda

    @staticmethod
    def _plane_basis_from_tangent(t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = t / (np.linalg.norm(t) + 1e-12)
        z = np.array([0.0, 0.0, 1.0])
        y = np.array([0.0, 1.0, 0.0])

        u = np.cross(t, z)
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(t, y)
        u = u / (np.linalg.norm(u) + 1e-12)
        v = np.cross(t, u)
        v = v / (np.linalg.norm(v) + 1e-12)
        return u, v, t

    def _robust_width_slice(self, slice_pts_3d: np.ndarray, p: np.ndarray, u: np.ndarray, v: np.ndarray):
        X = slice_pts_3d - p
        x2 = np.stack([X @ u, X @ v], axis=1)

        mu = x2.mean(axis=0)
        Y = x2 - mu
        C = (Y.T @ Y) / max(1, len(Y) - 1)
        eigvals, eigvecs = np.linalg.eigh(C)
        a = eigvecs[:, np.argmax(eigvals)]

        proj = x2 @ a
        lo = np.quantile(proj, self.qlo)
        hi = np.quantile(proj, self.qhi)
        w = float(hi - lo)

        idx_lo = int(np.argmin(np.abs(proj - lo)))
        idx_hi = int(np.argmin(np.abs(proj - hi)))
        return w, slice_pts_3d[idx_lo], slice_pts_3d[idx_hi]

    @staticmethod
    def _max_width_slice(slice_pts_3d: np.ndarray, p: np.ndarray, u: np.ndarray, v: np.ndarray):
        X = slice_pts_3d - p
        x2 = np.stack([X @ u, X @ v], axis=1)
        if len(x2) < 2:
            return 0.0, slice_pts_3d[0], slice_pts_3d[0]

        D = squareform(pdist(x2, metric="euclidean"))
        i, j = np.unravel_index(np.argmax(D), D.shape)
        w = float(D[i, j])
        return w, slice_pts_3d[int(i)], slice_pts_3d[int(j)]

    def compute(self, leaf_points: np.ndarray, centerline_points: np.ndarray) -> WidthResult:
        C = CenterlineExtractor.resample_by_step(centerline_points, step=self.step)
        if len(C) < 2:
            return WidthResult(profile=[], max_item=None)

        tree = cKDTree(leaf_points)
        profile: List[WidthItem] = []
        max_item: Optional[WidthItem] = None

        for s_idx in range(len(C) - 1):
            a = C[s_idx]
            b = C[s_idx + 1]
            ab = b - a
            seg_len = float(np.linalg.norm(ab))
            if seg_len < 1e-9:
                continue

            mid = 0.5 * (a + b)
            search_r = float(np.sqrt((0.5 * seg_len) ** 2 + self.radius ** 2))
            idx = tree.query_ball_point(mid, r=search_r)
            if len(idx) < self.min_slice_pts:
                continue
            cand = leaf_points[idx]

            t = (cand - a) @ ab / (seg_len * seg_len)
            mask = (t >= 0.0) & (t <= 1.0)
            cand = cand[mask]
            if len(cand) < self.min_slice_pts:
                continue

            proj = a + np.outer(t[mask], ab)
            d = np.linalg.norm(cand - proj, axis=1)
            cand = cand[d <= self.radius]
            if len(cand) < self.min_slice_pts:
                continue

            u, v, _ = self._plane_basis_from_tangent(ab)
            X = cand - a
            x2 = np.stack([X @ u, X @ v], axis=1)
            if len(x2) < 2:
                continue

            D = squareform(pdist(x2, metric="euclidean"))
            i, j = np.unravel_index(np.argmax(D), D.shape)
            w = float(D[i, j])
            pL = cand[int(i)]
            pR = cand[int(j)]
            width_vec = np.asarray(pR - pL, dtype=np.float64)
            width_norm = float(np.linalg.norm(width_vec))
            if width_norm <= 1e-12:
                perp_abs_dot = 1.0
            else:
                width_dir = width_vec / width_norm
                tangent = ab / seg_len
                perp_abs_dot = float(abs(np.dot(width_dir, tangent)))
            score = float(w * (1.0 - float(self.perp_lambda) * perp_abs_dot))

            item = WidthItem(
                s_index=int(s_idx),
                p=mid.copy(),
                width=w,
                pL=pL.copy(),
                pR=pR.copy(),
                slice_n=int(len(cand)),
                tangent=(ab / seg_len).copy(),
                perp_abs_dot=float(perp_abs_dot),
                score=score,
            )
            profile.append(item)
            if max_item is None or item.score > max_item.score:
                max_item = item

        return WidthResult(profile=profile, max_item=max_item)
