from typing import Tuple
import numpy as np
from PIL import Image


def load_image_as_vectors(path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Load image, return:
      X: (n,3) float64 RGB vectors in [0,255]
      size_wh: (w,h)
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    arr = np.asarray(img, dtype=np.float64)  # (h,w,3)
    X = arr.reshape(-1, 3)
    return X, (w, h)


def vectors_to_image(X: np.ndarray, size_wh: Tuple[int, int]) -> Image.Image:
    """
    Convert (n,3) vectors into RGB PIL image of size (w,h).
    """
    w, h = size_wh
    arr = np.clip(X.reshape(h, w, 3), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def recolor_by_cluster_mean(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Recolor each point by mean color of its cluster label.
    Noise label (-1): keep original color.
    """
    Xf = X.astype(np.float64, copy=False)
    X_out = Xf.copy()

    for lab in np.unique(labels):
        if lab == -1:
            continue
        mask = (labels == lab)
        if np.any(mask):
            mean_color = Xf[mask].mean(axis=0)
            X_out[mask] = mean_color

    return X_out


def downsample_image_vectors(
    X: np.ndarray, size_wh: Tuple[int, int], factor: int
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Downsample image vectors by taking every factor-th pixel in x and y.
    Returns downsampled X and downsampled size.
    """
    if factor <= 1:
        return X, size_wh

    w, h = size_wh
    img = X.reshape(h, w, 3)
    img_ds = img[::factor, ::factor, :]
    h2, w2, _ = img_ds.shape
    return img_ds.reshape(-1, 3), (w2, h2)


def upscale_labels_nn(labels_ds: np.ndarray, size_wh: Tuple[int, int], factor: int) -> np.ndarray:
    """
    Nearest-neighbor upscale of labels from downsampled grid back to original grid.
    """
    if factor <= 1:
        return labels_ds

    w, h = size_wh
    h_ds = (h + factor - 1) // factor
    w_ds = (w + factor - 1) // factor

    lab_img_ds = labels_ds.reshape(h_ds, w_ds)
    lab_full = np.repeat(np.repeat(lab_img_ds, factor, axis=0), factor, axis=1)
    lab_full = lab_full[:h, :w]
    return lab_full.reshape(-1)