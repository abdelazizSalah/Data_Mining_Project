import numpy as np

def pairwise_distance(X: np.ndarray, Y: np.ndarray, metric: str) -> np.ndarray:
    """
    Compute pairwise distances between rows of X (n,d) and rows of Y (m,d).
    Returns a (n,m) matrix.
    """
    diff = X[:, None, :] - Y[None, :, :]  # (n,m,d)
    if metric == "euclidean":
        return np.sqrt(np.sum(diff * diff, axis=2))
    if metric == "manhattan":
        return np.sum(np.abs(diff), axis=2)
    if metric == "maximum":
        return np.max(np.abs(diff), axis=2)
    raise ValueError(f"Unknown metric: {metric}")


def distance_to_point(X: np.ndarray, p: np.ndarray, metric: str) -> np.ndarray:
    """
    Distances between each row of X (n,d) and single point p (d,).
    Returns (n,) array.
    """
    diff = X - p
    if metric == "euclidean":
        return np.sqrt(np.sum(diff * diff, axis=1))
    if metric == "manhattan":
        return np.sum(np.abs(diff), axis=1)
    if metric == "maximum":
        return np.max(np.abs(diff), axis=1)
    raise ValueError(f"Unknown metric: {metric}")
