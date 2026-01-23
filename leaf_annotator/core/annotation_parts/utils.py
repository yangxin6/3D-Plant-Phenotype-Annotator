# core/annotation_parts/utils.py
from typing import List, Optional

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

def _polyline_length(P: Optional[np.ndarray]) -> Optional[float]:
    if P is None or len(P) < 2:
        return None
    return float(np.linalg.norm(P[1:] - P[:-1], axis=1).sum())


def _resample_polyline_step(P: np.ndarray, step: float) -> np.ndarray:
    if P is None or len(P) < 2:
        return np.asarray(P).copy()
    seg = np.linalg.norm(P[1:] - P[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = s[-1]
    if L <= 1e-12 or step <= 0:
        return P.copy()
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


def _smooth_polyline_window(P: Optional[np.ndarray], win: int) -> Optional[np.ndarray]:
    if P is None:
        return None
    P = np.asarray(P, dtype=np.float64)
    if len(P) < 2:
        return P.copy()
    w = max(3, int(win) | 1)
    if len(P) < w:
        return P.copy()
    out = P.copy()
    half = w // 2
    for i in range(len(P)):
        a = max(0, i - half)
        b = min(len(P), i + half + 1)
        out[i] = P[a:b].mean(axis=0)
    out[0] = P[0]
    out[-1] = P[-1]
    return out


def _build_knn_graph(points: np.ndarray, k: int) -> csr_matrix:
    n = int(points.shape[0])
    if n == 0:
        return csr_matrix((0, 0), dtype=np.float64)

    kk = max(1, int(k))
    tree = cKDTree(points)

    dists, nbrs = tree.query(points, k=min(kk + 1, n))
    if dists.ndim == 1:
        dists = dists[:, None]
        nbrs = nbrs[:, None]

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for i in range(n):
        for jpos in range(1, nbrs.shape[1]):
            j = int(nbrs[i, jpos])
            if j < 0 or j >= n or j == i:
                continue
            w = float(dists[i, jpos])
            rows.append(i); cols.append(j); data.append(w)
            rows.append(j); cols.append(i); data.append(w)

    G = csr_matrix(
        (np.array(data, dtype=np.float64),
         (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(n, n)
    )
    return G


def _build_radius_graph(points: np.ndarray, radius: float) -> csr_matrix:
    n = int(points.shape[0])
    if n == 0:
        return csr_matrix((0, 0), dtype=np.float64)
    r = float(radius)
    if r <= 0:
        raise ValueError("graph_radius must be > 0")

    tree = cKDTree(points)
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for i in range(n):
        nbrs = tree.query_ball_point(points[i], r=r)
        for j in nbrs:
            if j == i:
                continue
            w = float(np.linalg.norm(points[i] - points[j]))
            rows.append(i); cols.append(j); data.append(w)

    G = csr_matrix(
        (np.array(data, dtype=np.float64),
         (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(n, n)
    )
    return G


def _shortest_path_indices(G: csr_matrix, start: int, end: int) -> Optional[List[int]]:
    n = G.shape[0]
    if start < 0 or start >= n or end < 0 or end >= n:
        return None
    if start == end:
        return [start]

    dist, pred = dijkstra(G, indices=start, return_predecessors=True)
    if np.isinf(dist[end]):
        return None

    path = [end]
    cur = end
    while cur != start:
        cur = int(pred[cur])
        if cur < 0:
            return None
        path.append(cur)
    path.reverse()
    return path
