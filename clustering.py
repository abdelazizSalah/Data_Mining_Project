#!/usr/bin/env python3
"""
Foundations of Data Mining – Task 1: Clustering Algorithms for Image Segmentation

Implements from scratch:
  - k-means clustering
  - DBSCAN clustering
Distance metrics:
  - Euclidean
  - Manhattan
  - Maximum (Chebyshev)

Image segmentation:
  - Each pixel is a 3D RGB vector [R,G,B]
  - After clustering, recolor each pixel by its cluster's average RGB color

CLI examples:
  python segment.py --image input.jpg --algorithm kmeans --k 8 --distance euclidean --out out_kmeans.png
  python segment.py --image input.jpg --algorithm dbscan --eps 12 --min-samples 8 --distance manhattan --out out_dbscan.png

Dependencies:
  pip install numpy pillow
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image


# ============================================================
# Distance functions (vectorized)
# ============================================================
from utils.distance_functions import pairwise_distance, distance_to_point


# ============================================================
# Image helpers
# ============================================================

from utils.image_helpers import load_image_as_vectors, vectors_to_image, recolor_by_cluster_mean, downsample_image_vectors, upscale_labels_nn


# ============================================================
# K-Means (from scratch)
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


# ============================================================
# DBSCAN (from scratch) – simple O(n^2) implementation
# ============================================================

@dataclass
class DBSCAN:
    eps: float
    min_samples: int
    metric: str = "euclidean"

    labels_: Optional[np.ndarray] = None  # (n,) -1=noise, 0..C-1 clusters

    def fit(self, X: np.ndarray) -> "DBSCAN":
        """
        Fit DBSCAN on X (n,d).
        Basic implementation: region queries are O(n), overall O(n^2).
        """
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n,d).")

        n, _ = X.shape
        if self.eps <= 0:
            raise ValueError("eps must be > 0")
        if self.min_samples < 1:
            raise ValueError("min_samples must be >= 1")

        Xf = X.astype(np.float64, copy=False)
        labels = np.full(n, -1, dtype=np.int32)
        visited = np.zeros(n, dtype=bool)

        cluster_id = 0

        def region_query(i: int) -> np.ndarray:
            d = distance_to_point(Xf, Xf[i], self.metric)
            return np.where(d <= self.eps)[0]

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True

            neighbors = region_query(i)
            if neighbors.size < self.min_samples:
                labels[i] = -1
                continue

            # start new cluster
            labels[i] = cluster_id
            seeds = list(neighbors.tolist())

            k = 0
            while k < len(seeds):
                j = seeds[k]

                if not visited[j]:
                    visited[j] = True
                    neighbors_j = region_query(j)
                    if neighbors_j.size >= self.min_samples:
                        # add neighbors (avoid duplicates)
                        for nj in neighbors_j.tolist():
                            if nj not in seeds:
                                seeds.append(nj)

                if labels[j] == -1:
                    labels[j] = cluster_id

                k += 1

            cluster_id += 1

        self.labels_ = labels
        return self


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 1: Image segmentation with k-means or DBSCAN (from scratch).")
    p.add_argument("--image", required=True, help="Input image path (jpg/png/...).")
    p.add_argument("--out", default="segmented.png", help="Output image path.")

    p.add_argument("--algorithm", choices=["kmeans", "dbscan"], required=True, help="Clustering algorithm.")
    p.add_argument("--distance", choices=["euclidean", "manhattan", "maximum"], default="euclidean",
                   help="Distance metric.")

    # k-means params
    p.add_argument("--k", type=int, default=8, help="k-means: number of clusters.")
    p.add_argument("--max-iter", type=int, default=50, help="k-means: max iterations.")
    p.add_argument("--tol", type=float, default=1e-4, help="k-means: centroid shift tolerance.")
    p.add_argument("--seed", type=int, default=None, help="k-means: random seed (optional).")

    # dbscan params
    p.add_argument("--eps", type=float, default=12.0, help="DBSCAN: neighborhood radius eps.")
    p.add_argument("--min-samples", type=int, default=8, help="DBSCAN: min_samples.")
    p.add_argument("--downsample", type=int, default=1,
                   help="DBSCAN speed helper. 2 means process every 2nd pixel in x/y (default: 1).")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    print("Starting image segmentation with clustering...")
    X, size_wh = load_image_as_vectors(args.image)
    w, h = size_wh
    n = X.shape[0]
    print(f"Loaded: {args.image} ({w}x{h}) => {n} pixels")

    if args.algorithm == "kmeans":
        t0 = time.time()
        model = KMeans(
            k=args.k,
            metric=args.distance,
            max_iter=args.max_iter,
            tol=args.tol,
            seed=args.seed,
        ).fit(X)
        t1 = time.time()

        labels = model.labels_
        X_seg = recolor_by_cluster_mean(X, labels)
        img_out = vectors_to_image(X_seg, size_wh)
        img_out.save(args.out)

        print(f"k-means finished: k={args.k}, distance={args.distance}")
        print(f"Clusters: {len(np.unique(labels))}")
        print(f"Time: {t1 - t0:.3f}s")
        print(f"Saved: {args.out}")
        return 0

    if args.algorithm == "dbscan":
        factor = max(1, int(args.downsample))
        X_ds, size_ds = downsample_image_vectors(X, size_wh, factor)
        print(f"DBSCAN on {X_ds.shape[0]} points (downsample={factor})")

        t0 = time.time()
        model = DBSCAN(
            eps=args.eps,
            min_samples=args.min_samples,
            metric=args.distance,
        ).fit(X_ds)
        t1 = time.time()

        labels_ds = model.labels_
        labels_full = upscale_labels_nn(labels_ds, size_wh, factor)

        X_seg = recolor_by_cluster_mean(X, labels_full)
        img_out = vectors_to_image(X_seg, size_wh)
        img_out.save(args.out)

        n_clusters = len(set(labels_ds.tolist())) - (1 if -1 in labels_ds else 0)
        n_noise = int(np.sum(labels_ds == -1))

        print(f"DBSCAN finished: eps={args.eps}, min_samples={args.min_samples}, distance={args.distance}")
        print(f"Clusters (downsampled): {n_clusters}, noise points (downsampled): {n_noise}")
        print(f"Time: {t1 - t0:.3f}s (simple O(n^2) implementation)")
        print(f"Saved: {args.out}")
        return 0

    raise RuntimeError("Unknown algorithm (should be unreachable).")


if __name__ == "__main__":
    raise SystemExit(main())
