#!/usr/bin/env python3
"""
Foundations of Data Mining – Task 1: Clustering Algorithms for Image Segmentation

Implements:
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
  python segment.py --image input.jpg --algorithm kmeans --k 5 --distance euclidean --out out_kmeans.png
  python segment.py --image input.jpg --algorithm dbscan --eps 4 --min-samples 5 --distance manhattan --out out_dbscan.png

Dependencies:
  pip install numpy pillow
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from PIL import Image


# ============================================================
# Image helpers
# ============================================================

from utils.image_helpers import load_image_as_vectors, vectors_to_image, recolor_by_cluster_mean, downsample_image_vectors, upscale_labels_nn


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 1: Image segmentation with k-means or DBSCAN.")
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
        from utils.model import KMeans
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
        from utils.model import DBSCAN
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
