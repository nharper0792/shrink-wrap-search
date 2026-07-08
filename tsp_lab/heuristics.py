import itertools
import math

import numpy as np

from .geometry import tour_length


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
# Physically: a circle centered on the centroid shrinks inward uniformly, so
# it contacts points strictly in order of decreasing distance from the
# centroid (the farthest point sticks first, the closest sticks last). Each
# newly-stuck point gets spliced into the loop of already-stuck points at
# whichever edge it forms the shallowest triangle with -- shallowness is the
# point's perpendicular distance to that edge *segment* (clamped to the
# segment, not the infinite line through it, so a point can't claim an edge
# it isn't actually near just because it's collinear with the edge's line).
# The membrane deforms least where the new point barely pokes above the
# nearest existing surface, and that's where it catches.
# ---------------------------------------------------------------------------

def _triangle_height(points, p, a, b):
    ax, ay = points[a]
    bx, by = points[b]
    px, py = points[p]
    abx, aby = bx - ax, by - ay
    seg_len2 = abx * abx + aby * aby
    if seg_len2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / seg_len2
    t = max(0.0, min(1.0, t))
    projx, projy = ax + t * abx, ay + t * aby
    return math.hypot(px - projx, py - projy)


def shrink_wrap_tour(points, return_trace=False):
    n = len(points)
    if n <= 3:
        tour = list(range(n))
        trace = [{"step": 0, "inserted": None, "edge": None, "path": list(tour)}]
        if return_trace:
            return tour, trace
        return tour

    c = points.mean(axis=0)
    dists = np.linalg.norm(points - c, axis=1)
    order = np.argsort(-dists).tolist()  # farthest first: contact order of a shrinking circle

    path = order[:3]
    trace = [{"step": 0, "inserted": None, "edge": None, "path": list(path)}]

    for step, p in enumerate(order[3:], start=1):
        m = len(path)
        best_i, best_h = 0, math.inf
        for i in range(m):
            a, b = path[i], path[(i + 1) % m]
            h = _triangle_height(points, p, a, b)
            if h < best_h:
                best_h, best_i = h, i
        a, b = path[best_i], path[(best_i + 1) % m]
        path.insert(best_i + 1, p)
        trace.append({"step": step, "inserted": p, "edge": (a, b), "path": list(path)})

    if return_trace:
        return path, trace
    return path


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
