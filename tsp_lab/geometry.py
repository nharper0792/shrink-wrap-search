import numpy as np


def random_points(n, seed=None, low=0.0, high=100.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(n, 2))


def clustered_points(n, seed=None, n_clusters=4, spread=8.0, low=0.0, high=100.0):
    """Non-convex-friendly instance: points drawn from a few Gaussian blobs."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(low + spread, high - spread, size=(n_clusters, 2))
    assignments = rng.integers(0, n_clusters, size=n)
    pts = centers[assignments] + rng.normal(0.0, spread, size=(n, 2))
    return np.clip(pts, low, high)


def tour_length(points, tour):
    pts = points[tour]
    diffs = pts - np.roll(pts, -1, axis=0)
    return float(np.sqrt((diffs ** 2).sum(axis=1)).sum())


def centroid(points):
    return points.mean(axis=0)


def is_valid_tour(tour, n):
    return sorted(tour) == list(range(n))
