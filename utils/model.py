
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from collections import deque
import numpy as np
from utils.distance_functions import distance_to_point, pairwise_distance

# assumes you already have:
# - distance_to_point(Xf, Xf[i], metric) -> (n,) distances

@dataclass
class DBSCAN:
    eps: float
    min_samples: int
    metric: str = "euclidean"

    labels_: Optional[np.ndarray] = None  # cluster id >=0, NOISE=-1

    def fit(self, X: np.ndarray) -> "DBSCAN":
        """
        DBSCAN clustering algorithm implementation.
        """
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n,d).")

        n, _ = X.shape
        if self.eps <= 0:
            raise ValueError("eps must be > 0")
        if self.min_samples < 1:
            raise ValueError("min_samples must be >= 1")

        Xf = X.astype(np.float64, copy=False)

        UNCLASSIFIED = -2
        NOISE = -1

        labels = np.full(n, UNCLASSIFIED, dtype=np.int32)

        def region_query(i: int) -> np.ndarray:
            # N_eps(i) including i itself
            d = distance_to_point(Xf, Xf[i], self.metric)
            return np.where(d <= self.eps)[0]

        def expand_cluster(start_idx: int, cluster_id: int) -> bool:
            """
            ExpandCluster(objectSet, StartObject, ClusterId, eps, MinPts) -> Boolean
            following the slide pseudocode. :contentReference[oaicite:1]{index=1}
            """
            seeds = region_query(start_idx)

            # if |seeds| < MinPts -> StartObject is NOISE and return false
            if seeds.size < self.min_samples:
                labels[start_idx] = NOISE
                return False

            # forall o in seeds: o.ClId = ClusterId
            labels[seeds] = cluster_id

            # remove StartObject from seeds
            # managing seeds as a queue of indices to process
            q = deque(int(i) for i in seeds if int(i) != start_idx)
            in_queue = np.zeros(n, dtype=bool)
            for i in q:
                in_queue[i] = True

            while q:
                o = q.popleft()
                in_queue[o] = False

                neighborhood = region_query(o)

                # if |Neighborhood| >= MinPts then o is core object
                if neighborhood.size >= self.min_samples:
                    for p in neighborhood:
                        p = int(p)

                        # if p.ClId in {UNCLASSIFIED, NOISE}
                        if labels[p] == UNCLASSIFIED or labels[p] == NOISE:
                            # if p was UNCLASSIFIED, add to seeds
                            if labels[p] == UNCLASSIFIED and (not in_queue[p]):
                                q.append(p)
                                in_queue[p] = True

                            # p.ClId := ClusterId
                            labels[p] = cluster_id

            return True

        cluster_id = 0
        for i in range(n):
            if labels[i] == UNCLASSIFIED:
                if expand_cluster(i, cluster_id):
                    cluster_id += 1

        # Converting to  UNCLASSIFIED 
        labels[labels == UNCLASSIFIED] = NOISE

        self.labels_ = labels
        return self


# ============================================================
# K-Means
# ============================================================

@dataclass
class KMeans:
    k: int
    metric: str = "euclidean"
    max_iter: int = 50
    tol: float = 1e-4
    seed: Optional[int] = None

    centroids_: Optional[np.ndarray] = None  # (k,d)
    labels_: Optional[np.ndarray] = None     # (n,)

    def fit(self, X: np.ndarray) -> "KMeans":
        """
        Fit k-means on X (n,d).
        """
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n,d).")

        n, d = X.shape
        if self.k < 1 or self.k > n:
            raise ValueError(f"k must be in [1, n]. Got k={self.k}, n={n}")

        Xf = X.astype(np.float64, copy=False)
        rng = np.random.default_rng(self.seed)

        # init centroids by sampling k points
        init_idx = rng.choice(n, size=self.k, replace=False)
        centroids = Xf[init_idx].copy()

        labels = np.zeros(n, dtype=np.int32)

        for _ in range(self.max_iter):
            # assignment
            dist = pairwise_distance(Xf, centroids, self.metric)  # (n,k)
            new_labels = np.argmin(dist, axis=1).astype(np.int32)

            # update
            new_centroids = centroids.copy()
            for j in range(self.k):
                mask = (new_labels == j)
                if np.any(mask):
                    new_centroids[j] = Xf[mask].mean(axis=0)
                else:
                    # re-seed empty cluster
                    new_centroids[j] = Xf[rng.integers(0, n)]

            # convergence
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            labels = new_labels
            if shift <= self.tol:
                break

        self.centroids_ = centroids
        self.labels_ = labels
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.centroids_ is None:
            raise RuntimeError("KMeans not fitted yet.")
        Xf = X.astype(np.float64, copy=False)
        dist = pairwise_distance(Xf, self.centroids_, self.metric)
        return np.argmin(dist, axis=1).astype(np.int32)
