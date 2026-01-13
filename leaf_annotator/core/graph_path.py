import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


class KNNGraphPath:
    def __init__(self, k: int = 25):
        if k < 3:
            raise ValueError("k 至少 3")
        self.k = k

    def build_graph(self, points: np.ndarray) -> csr_matrix:
        tree = cKDTree(points)
        dists, nbrs = tree.query(points, k=self.k + 1)  # include self
        rows, cols, data = [], [], []
        n = len(points)

        for i in range(n):
            for j, dist in zip(nbrs[i, 1:], dists[i, 1:]):  # skip self
                rows.append(i); cols.append(int(j)); data.append(float(dist))
                rows.append(int(j)); cols.append(i); data.append(float(dist))

        return csr_matrix((data, (rows, cols)), shape=(n, n))

    @staticmethod
    def shortest_path_indices(W: csr_matrix, start_idx: int, end_idx: int) -> np.ndarray:
        dist, pred = dijkstra(W, directed=False, indices=start_idx, return_predecessors=True)
        if np.isinf(dist[end_idx]):
            raise RuntimeError("kNN 图不连通：起点到终点不可达。请增大 k 或减小 voxel。")

        path = []
        cur = end_idx
        while cur != -9999 and cur != start_idx:
            path.append(int(cur))
            cur = pred[cur]
        path.append(int(start_idx))
        path.reverse()
        return np.array(path, dtype=int)
