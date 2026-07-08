import itertools
import math

import numpy as np

from .geometry import tour_length
from .hull import convex_hull_indices


# ---------------------------------------------------------------------------
# Approach 1: angular sort ("wedge") heuristic
#
# Draw a circle centered on the centroid of the points, radius = distance to
# the farthest point. Sort points by the angle they make with the centroid
# and visit them in that (clockwise) order. This is the classic "angular
# sweep" TSP construction heuristic.
# ---------------------------------------------------------------------------

def angular_sort_tour(points, clockwise=True):
    c = points.mean(axis=0)
    vecs = points - c
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])
    order = np.argsort(-angles if clockwise else angles)
    return order.tolist()


def bounding_circle(points):
    c = points.mean(axis=0)
    r = float(np.linalg.norm(points - c, axis=1).max())
    return c, r


# ---------------------------------------------------------------------------
# Approach 2: shrink-wrap ("vacuum bag") heuristic
#
# Physically: a membrane shrinks onto the point set from the outside. It
# first conforms to the convex hull (the points it reaches first), sticks
# there, and then keeps sinking into each "pocket" between consecutive
# stuck points, conforming to the convex hull of whatever points are left
# in that pocket, recursively, until every point is stuck to the membrane.
#
# This is equivalent to recursive convex-hull peeling, where the remaining
# (interior) points are assigned to a pocket by which pair of hull vertices
# their angle (measured from the hull's own centroid) falls between.
# ---------------------------------------------------------------------------

def _bucket_by_wedge(points, hull, interior, c):
    hull_angles = [math.atan2(points[h][1] - c[1], points[h][0] - c[0]) for h in hull]
    hull_angles = [a % (2 * math.pi) for a in hull_angles]
    pockets = [[] for _ in hull]
    m = len(hull)
    for p in interior:
        a = math.atan2(points[p][1] - c[1], points[p][0] - c[0]) % (2 * math.pi)
        placed = False
        for e in range(m):
            lo = hull_angles[e]
            hi = hull_angles[(e + 1) % m]
            span = (hi - lo) % (2 * math.pi)
            if span == 0:
                span = 2 * math.pi
            offset = (a - lo) % (2 * math.pi)
            if offset <= span:
                pockets[e].append(p)
                placed = True
                break
        if not placed:
            pockets[-1].append(p)
    return pockets


def _shrink_wrap_recursive(points, idx, trace, depth):
    if len(idx) <= 2:
        order = list(idx)
        trace.append({"depth": depth, "hull": order, "pockets": {}, "centroid": None})
        return order

    hull = convex_hull_indices(points, idx)

    if len(hull) < 3 or len(hull) == len(idx):
        order = hull if len(hull) == len(idx) else sorted(idx, key=lambda i: (points[i][0], points[i][1]))
        trace.append({"depth": depth, "hull": order, "pockets": {}, "centroid": None})
        return order

    hull_set = set(hull)
    interior = [i for i in idx if i not in hull_set]
    c = points[hull].mean(axis=0)
    pockets = _bucket_by_wedge(points, hull, interior, c)

    trace.append({
        "depth": depth,
        "hull": list(hull),
        "pockets": {e: list(pockets[e]) for e in range(len(hull)) if pockets[e]},
        "centroid": c,
    })

    tour = []
    for e, h in enumerate(hull):
        tour.append(h)
        if pockets[e]:
            tour.extend(_shrink_wrap_recursive(points, pockets[e], trace, depth + 1))
    return tour


def shrink_wrap_tour(points, return_trace=False):
    idx = list(range(len(points)))
    trace = []
    tour = _shrink_wrap_recursive(points, idx, trace, depth=0)
    if return_trace:
        return tour, trace
    return tour


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def nearest_neighbor_tour(points, start=0):
    n = len(points)
    visited = np.zeros(n, dtype=bool)
    tour = [start]
    visited[start] = True
    for _ in range(n - 1):
        last = tour[-1]
        d = np.linalg.norm(points - points[last], axis=1)
        d[visited] = np.inf
        nxt = int(np.argmin(d))
        tour.append(nxt)
        visited[nxt] = True
    return tour


def two_opt(points, tour, max_passes=200):
    n = len(tour)
    best = list(tour)
    dmat = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    improved = True
    while improved and max_passes > 0:
        improved = False
        max_passes -= 1
        for i in range(n - 1):
            a, b = best[i], best[i + 1]
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                c, d = best[j], best[(j + 1) % n]
                if dmat[a, b] + dmat[c, d] > dmat[a, c] + dmat[b, d] + 1e-9:
                    best[i + 1:j + 1] = best[i + 1:j + 1][::-1]
                    a, b = best[i], best[i + 1]
                    improved = True
    return best


def nearest_neighbor_2opt_tour(points, start=0):
    return two_opt(points, nearest_neighbor_tour(points, start=start))


def brute_force_tour(points, max_n=10, force=False):
    n = len(points)
    if n > max_n and not force:
        raise ValueError(
            f"n={n} exceeds the brute-force safety cap of {max_n} "
            f"(n! permutations grows explosively); pass force=True to override"
        )
    if n <= 1:
        return list(range(n))
    best_tour, best_len = None, math.inf
    for perm in itertools.permutations(range(1, n)):
        t = [0, *perm]
        L = tour_length(points, t)
        if L < best_len:
            best_len, best_tour = L, t
    return best_tour


def held_karp_tour(points, max_n=15, force=False):
    """Exact DP solution, O(2^n * n^2). Practical up to ~15 points."""
    n = len(points)
    if n > max_n and not force:
        raise ValueError(
            f"n={n} exceeds the Held-Karp safety cap of {max_n}; pass force=True to override"
        )
    if n <= 1:
        return list(range(n))
    if n == 2:
        return [0, 1]

    dmat = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    C = {(1 << i, i): (dmat[0, i], 0) for i in range(1, n)}

    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            bits = 0
            for b in subset:
                bits |= 1 << b
            for k in subset:
                prev = bits & ~(1 << k)
                best = min(
                    (C[(prev, m)][0] + dmat[m, k], m)
                    for m in subset if m != k
                )
                C[(bits, k)] = best

    bits = (1 << n) - 2  # all nodes except 0
    opt_cost, parent = min(
        (C[(bits, k)][0] + dmat[k, 0], k) for k in range(1, n)
    )
    path = []
    for _ in range(n - 1):
        path.append(parent)
        bits, parent = bits & ~(1 << parent), C[(bits, parent)][1]
    path.append(0)
    path.reverse()
    return path


def exact_tour(points, max_n=15, force=False):
    """Best available exact method for the given size."""
    return held_karp_tour(points, max_n=max_n, force=force)
