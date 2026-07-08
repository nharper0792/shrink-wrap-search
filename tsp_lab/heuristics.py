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
# Approach 3: orbit & recenter
#
# A walker starts on the original bounding circle and orbits the centroid in
# a fixed rotational direction. For the still-unvisited points, one sweep
# ("leg") works like this: on the first tiny step, any point whose distance
# to the walker just *increased* is disqualified for this leg (it's more
# than a half-turn "behind" the walker's heading). Among the survivors, the
# walker keeps going until one point's distance stops decreasing -- i.e. the
# walker has just passed its closest approach -- and that point is selected.
# The walker jumps to it and a new leg starts from there, on the *same*
# circle.
#
# If a leg disqualifies every remaining point (nothing is "ahead"), the
# circle is recalculated from just the remaining points (new centroid, new
# radius), and the walker marches from its current spot onto that new circle
# -- preserving its angular bearing around the new center -- before resuming
# orbiting in the same rotational direction.
#
# Assumptions not pinned down by the description, called out explicitly:
# the walker's start position (start_angle, default 0 rad on the original
# circle) and the exact "march to the new circle" mechanic (here: move
# radially from the current position until reaching the new circle,
# preserving the angle around the new center -- i.e. only the radius
# changes, not the bearing).
#
# If a run gets stuck (recalculating without ever finding a next point --
# capped at max_recalcs) some points may never get picked up by the walker.
# Those are inserted afterward next to whichever already-placed point has
# the closest angle to them, both measured from the *original* center and
# the *original* start angle.
# ---------------------------------------------------------------------------

def _circular_distance(a, b):
    d = abs(a - b) % (2 * math.pi)
    return min(d, 2 * math.pi - d)


def orbit_recenter_tour(points, direction=-1, start_angle=0.0, max_recalcs=None, return_trace=False):
    n = len(points)
    if n <= 2:
        tour = list(range(n))
        trace = [{"type": "trivial", "path": list(tour)}]
        return (tour, trace) if return_trace else tour

    center0 = points.mean(axis=0)
    R0 = float(np.linalg.norm(points - center0, axis=1).max())
    start_pos = center0 + R0 * np.array([math.cos(start_angle), math.sin(start_angle)])

    remaining = set(range(n))
    tour = []
    cur_center, cur_radius, cur_pos, cur_theta = center0, R0, start_pos, start_angle
    trace = [{"type": "start", "center": cur_center.copy(), "radius": cur_radius,
              "cur_pos": cur_pos.copy(), "path": []}]

    if max_recalcs is None:
        max_recalcs = max(20, 4 * n)
    recalcs = 0

    def forward_offset(phi, theta):
        return (theta - phi) % (2 * math.pi) if direction < 0 else (phi - theta) % (2 * math.pi)

    while remaining:
        idx_list = list(remaining)
        vecs = points[idx_list] - cur_center
        phis = np.arctan2(vecs[:, 1], vecs[:, 0])
        offsets = np.array([forward_offset(p, cur_theta) for p in phis])
        kept_mask = offsets <= math.pi

        if kept_mask.any():
            kept_idx = np.array(idx_list)[kept_mask]
            best = int(kept_idx[np.argmin(offsets[kept_mask])])
            disqualified = [i for i, m in zip(idx_list, kept_mask) if not m]

            tour.append(best)
            remaining.discard(best)
            cur_pos = points[best]
            cur_theta = math.atan2(cur_pos[1] - cur_center[1], cur_pos[0] - cur_center[0])
            trace.append({"type": "select", "center": cur_center.copy(), "radius": cur_radius,
                          "selected": best, "disqualified": disqualified,
                          "cur_pos": cur_pos.copy(), "path": list(tour)})
        else:
            recalcs += 1
            if recalcs > max_recalcs or len(remaining) <= 1:
                break
            rem_pts = points[list(remaining)]
            new_center = rem_pts.mean(axis=0)
            new_R = float(np.linalg.norm(rem_pts - new_center, axis=1).max())
            if new_R < 1e-12:
                break
            theta_snap = math.atan2(cur_pos[1] - new_center[1], cur_pos[0] - new_center[0])
            cur_pos = new_center + new_R * np.array([math.cos(theta_snap), math.sin(theta_snap)])
            cur_center, cur_radius, cur_theta = new_center, new_R, theta_snap
            trace.append({"type": "recalc", "center": cur_center.copy(), "radius": cur_radius,
                          "cur_pos": cur_pos.copy(), "path": list(tour)})

    leftover = list(remaining)
    if leftover:
        def ang0(i):
            v = points[i] - center0
            return math.atan2(v[1], v[0])

        for q in sorted(leftover, key=lambda i: (ang0(i) - start_angle) % (2 * math.pi)):
            aq = ang0(q)
            best_pos = min(range(len(tour)), key=lambda p: _circular_distance(ang0(tour[p]), aq))
            tour.insert(best_pos + 1, q)
            trace.append({"type": "insert", "inserted": q, "after": tour[best_pos], "path": list(tour)})

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
