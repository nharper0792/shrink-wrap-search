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
    """Shallowness metric: perpendicular distance from p to segment (a, b)."""
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


def _cheapest_insertion_cost(points, p, a, b):
    """Textbook insertion metric: extra tour length added by splicing p into (a, b)."""
    ax, ay = points[a]
    bx, by = points[b]
    px, py = points[p]
    return (math.hypot(px - ax, py - ay) + math.hypot(bx - px, by - py)
            - math.hypot(bx - ax, by - ay))


def _shrink_wrap_generic(points, cost_fn, return_trace=False):
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
        best_i, best_cost = 0, math.inf
        for i in range(m):
            a, b = path[i], path[(i + 1) % m]
            cost = cost_fn(points, p, a, b)
            if cost < best_cost:
                best_cost, best_i = cost, i
        a, b = path[best_i], path[(best_i + 1) % m]
        path.insert(best_i + 1, p)
        trace.append({"step": step, "inserted": p, "edge": (a, b), "path": list(path)})

    if return_trace:
        return path, trace
    return path


def shrink_wrap_tour(points, return_trace=False):
    return _shrink_wrap_generic(points, _triangle_height, return_trace)


def shrink_wrap_cheapest_tour(points, return_trace=False):
    """Same recursive-insertion structure, but the textbook cheapest-insertion
    cost (added tour length) instead of geometric shallowness decides where
    each point splices in."""
    return _shrink_wrap_generic(points, _cheapest_insertion_cost, return_trace)


# ---------------------------------------------------------------------------
# Approach 3: orbit & recenter
#
# A walker starts on the original bounding circle and orbits the centroid in
# a fixed rotational direction. It picks off one point per lap fragment: on
# the first tiny step, any point whose distance to the walker just
# *increased* is disqualified for this leg (it's more than a half-turn
# "behind" the walker's heading); among the survivors, the walker keeps
# going until one point's distance stops decreasing -- it has just passed
# that point's closest approach -- and that point is selected.
#
# After *every* selection, the circle is recalculated from just the
# still-unvisited points (new centroid, new radius), and the walker marches
# from wherever it is onto that new circle -- preserving its angular bearing
# around the new center, i.e. only the radius changes, not the heading --
# before resuming orbiting in the same rotational direction. So the circle
# continuously shrinks and re-centers onto whatever's left, one point at a
# time.
#
# This is well-defined for any point layout: since the circle is always
# exactly centered on the true centroid of the remaining points, and a point
# set's own centroid can never have an empty arc wider than 180 degrees
# around it (otherwise the centroid wouldn't balance), there's always at
# least one point within the forward half-turn to select. No stuck/fallback
# handling is needed.
#
# Two things aren't pinned down by the description and are left as
# parameters: the walker's start position (`start_angle`, default 0 rad on
# the original circle) and the rotational sense (`direction`, default -1 =
# clockwise, matching angular_sort_tour's default).
# ---------------------------------------------------------------------------

def _select_min_offset(idx_list, offsets, points, cur_pos, tour):
    return idx_list[int(np.argmin(offsets))]


def _orientation(a, b, c):
    val = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if abs(val) < 1e-12:
        return 0
    return 1 if val > 0 else 2


def _on_segment(a, b, c):
    return (min(a[0], b[0]) - 1e-9 <= c[0] <= max(a[0], b[0]) + 1e-9 and
            min(a[1], b[1]) - 1e-9 <= c[1] <= max(a[1], b[1]) + 1e-9)


def _segments_intersect(p1, p2, p3, p4):
    o1, o2 = _orientation(p1, p2, p3), _orientation(p1, p2, p4)
    o3, o4 = _orientation(p3, p4, p1), _orientation(p3, p4, p2)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1, p2, p3):
        return True
    if o2 == 0 and _on_segment(p1, p2, p4):
        return True
    if o3 == 0 and _on_segment(p3, p4, p1):
        return True
    if o4 == 0 and _on_segment(p3, p4, p2):
        return True
    return False


def _select_min_offset_noncross(idx_list, offsets, points, cur_pos, tour):
    """Among candidates, prefer the smallest forward offset whose connecting
    edge doesn't cross an already-placed tour edge; falls back to the
    smallest offset overall if every candidate would cross something."""
    order = np.argsort(offsets)
    # drop the most recent edge: it may share cur_pos as an endpoint, which
    # the intersection test would otherwise flag as a false-positive "cross"
    edges = list(zip(tour[:-1], tour[1:]))[:-1] if len(tour) >= 2 else []
    for oi in order:
        cand = idx_list[oi]
        p_cand = points[cand]
        if not any(_segments_intersect(cur_pos, p_cand, points[a], points[b]) for a, b in edges):
            return cand
    return idx_list[int(order[0])]


def orbit_recenter_tour(points, direction=-1, start_angle=0.0, recenter_every=1,
                        select_fn=None, return_trace=False):
    """recenter_every=1 recalculates the circle after every point (the
    original design); larger values orbit the same circle for several picks
    before recentering. select_fn(idx_list, offsets, points, cur_pos, tour)
    picks which candidate to select among the "kept" (forward-half) ones --
    defaults to smallest offset; pass _select_min_offset_noncross to add a
    self-intersection check."""
    select_fn = select_fn or _select_min_offset
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

    def forward_offset(phi, theta):
        return (theta - phi) % (2 * math.pi) if direction < 0 else (phi - theta) % (2 * math.pi)

    first_leg = True
    since_recenter = 0
    while remaining:
        if not first_leg and since_recenter >= recenter_every:
            rem_idx = list(remaining)
            rem_pts = points[rem_idx]
            new_center = rem_pts.mean(axis=0)
            new_R = float(np.linalg.norm(rem_pts - new_center, axis=1).max())
            theta_snap = (math.atan2(cur_pos[1] - new_center[1], cur_pos[0] - new_center[0])
                          if new_R > 1e-12 else cur_theta)
            cur_pos = new_center + new_R * np.array([math.cos(theta_snap), math.sin(theta_snap)])
            cur_center, cur_radius, cur_theta = new_center, new_R, theta_snap
            trace.append({"type": "recalc", "center": cur_center.copy(), "radius": cur_radius,
                          "cur_pos": cur_pos.copy(), "path": list(tour)})
            since_recenter = 0

        idx_list = list(remaining)
        if len(idx_list) == 1:
            best, disqualified = idx_list[0], []
        else:
            vecs = points[idx_list] - cur_center
            phis = np.arctan2(vecs[:, 1], vecs[:, 0])
            offsets = np.array([forward_offset(p, cur_theta) for p in phis])
            kept_mask = offsets <= math.pi
            if not kept_mask.any():  # defensive only -- see proof above, shouldn't trigger
                kept_mask = np.ones_like(kept_mask, dtype=bool)
            kept_idx = np.array(idx_list)[kept_mask]
            kept_offsets = offsets[kept_mask]
            best = select_fn(kept_idx.tolist(), kept_offsets, points, cur_pos, tour)
            disqualified = [i for i, m in zip(idx_list, kept_mask) if not m]

        tour.append(best)
        remaining.discard(best)
        cur_pos = points[best]
        cur_theta = math.atan2(cur_pos[1] - cur_center[1], cur_pos[0] - cur_center[0])
        trace.append({"type": "select", "center": cur_center.copy(), "radius": cur_radius,
                      "selected": best, "disqualified": disqualified,
                      "cur_pos": cur_pos.copy(), "path": list(tour)})
        first_leg = False
        since_recenter += 1

    if return_trace:
        return tour, trace
    return tour


def orbit_recenter_noncross_tour(points, **kwargs):
    """Per-step crossing avoidance. Empirically this makes ~no difference:
    the open path built leg-by-leg essentially never crosses itself under
    this selection rule: all of the self-crossings observed turn out to be
    the *closing* edge (last point back to the first), which forms only
    after the loop ends and is invisible to any per-step check. Kept as a
    correctly-implemented negative result; see orbit_recenter_2opt_tour for
    the fix informed by that finding."""
    return orbit_recenter_tour(points, select_fn=_select_min_offset_noncross, **kwargs)


def orbit_recenter_2opt_tour(points):
    """A cleanup pass sees the closing edge (2-opt checks all edge pairs,
    wraparound included), so unlike orbit_recenter_noncross_tour this
    actually eliminates the self-crossings."""
    return two_opt(points, orbit_recenter_tour(points))


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


def or_opt(points, tour, max_passes=100):
    """Local search: repeatedly relocate a single point to wherever it's
    cheapest to reinsert, if that's cheaper than leaving it where it is.
    Targets a different failure mode than 2-opt: a single point visited
    "out of order" (e.g. Angular Sort's radial zigzags) rather than a
    crossing pair of edges."""
    n = len(tour)
    best = list(tour)
    dmat = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    improved = True
    while improved and max_passes > 0 and n >= 4:
        improved = False
        max_passes -= 1
        for i in range(n):
            p = best[i]
            prev_p, next_p = best[i - 1], best[(i + 1) % n]
            removal_gain = dmat[prev_p, p] + dmat[p, next_p] - dmat[prev_p, next_p]

            best_j, best_delta = None, -1e-9
            for j in range(n):
                if j == i or (j + 1) % n == i:
                    continue
                a, b = best[j], best[(j + 1) % n]
                insertion_cost = dmat[a, p] + dmat[p, b] - dmat[a, b]
                delta = removal_gain - insertion_cost
                if delta > best_delta:
                    best_delta, best_j = delta, j

            if best_j is not None:
                new_tour = best[:i] + best[i + 1:]
                insert_at = best_j if best_j < i else best_j - 1
                new_tour.insert(insert_at + 1, p)
                best = new_tour
                improved = True
    return best


def angular_sort_oropt_tour(points, clockwise=True):
    return or_opt(points, angular_sort_tour(points, clockwise=clockwise))


def shrink_wrap_2opt_tour(points):
    return two_opt(points, shrink_wrap_tour(points))


def shrink_wrap_cheapest_2opt_tour(points):
    return two_opt(points, shrink_wrap_cheapest_tour(points))


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


# ---------------------------------------------------------------------------
# Recursive clustering wrapper
#
# Group points into clusters of mutually "sufficiently close" points
# (connected components under a distance threshold), treat each cluster as
# a single point at its centroid, and run the same tour-construction
# function on those centroids to get a macro-order of clusters. Then
# recurse into each cluster to order its actual points the same way, and
# concatenate in macro-order. Works with any of the tour functions above.
#
# The default threshold is picked from the minimum spanning tree: sort the
# MST edge weights and cut at the biggest relative jump (gap) between
# consecutive weights, taking the geometric mean of the two edges around
# that jump. This is the standard trick for turning single-linkage
# clustering into something that doesn't need a pre-chosen cluster count --
# it naturally finds a handful of clusters on genuinely separated data, and
# degrades to near-singleton "clusters" (i.e. a no-op once `clustered_tour`
# falls back to the base algorithm) on data with no real cluster structure,
# since there's no standout gap to cut at.
# ---------------------------------------------------------------------------

def _mst_edge_weights(points):
    n = len(points)
    if n <= 1:
        return np.array([])
    dmat = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    key = dmat[0].copy()
    key[0] = np.inf
    weights = []
    for _ in range(n - 1):
        masked = np.where(in_tree, np.inf, key)
        u = int(np.argmin(masked))
        weights.append(float(masked[u]))
        in_tree[u] = True
        key = np.minimum(key, dmat[u])
    return np.array(weights)


def _default_cluster_threshold(points):
    w = np.sort(_mst_edge_weights(points))
    if len(w) < 2:
        return float(w[0]) * 1.5 if len(w) else 0.0
    ratios = w[1:] / np.maximum(w[:-1], 1e-9)
    gap_i = int(np.argmax(ratios))
    return float(math.sqrt(w[gap_i] * w[gap_i + 1]))


def cluster_by_threshold(points, threshold=None):
    """Connected components under 'distance <= threshold'. Returns a list of
    index-arrays (relative to `points`)."""
    n = len(points)
    if n == 0:
        return []
    if threshold is None:
        threshold = _default_cluster_threshold(points)
    dmat = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    adj = dmat <= threshold

    visited = np.zeros(n, dtype=bool)
    clusters = []
    for i in range(n):
        if visited[i]:
            continue
        stack, comp = [i], []
        visited[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in np.nonzero(adj[u] & ~visited)[0]:
                visited[v] = True
                stack.append(v)
        clusters.append(np.array(comp))
    return clusters


def _best_rotation(sub_tour, sub_points, anchor, next_anchor):
    """Pick the rotation + direction of a cyclic sub-tour that best faces its
    macro-neighbors: minimizes dist(anchor, entry point) + dist(exit point,
    next_anchor). Total internal length of a cycle is direction-invariant,
    so this doesn't change the sub-tour's own quality -- only which point it
    presents at each end to the rest of the tour."""
    m = len(sub_tour)
    if m <= 1 or (anchor is None and next_anchor is None):
        return sub_tour
    best_seq, best_cost = sub_tour, math.inf
    for start in range(m):
        for direction in (1, -1):
            seq = [sub_tour[(start + direction * k) % m] for k in range(m)]
            cost = 0.0
            if anchor is not None:
                cost += float(np.linalg.norm(sub_points[seq[0]] - anchor))
            if next_anchor is not None:
                cost += float(np.linalg.norm(sub_points[seq[-1]] - next_anchor))
            if cost < best_cost:
                best_cost, best_seq = cost, seq
    return best_seq


def clustered_tour(points, base_tour_fn, threshold=None, min_cluster_size=4, max_depth=6,
                   optimize_endpoints=False, _depth=0):
    n = len(points)
    if n <= min_cluster_size or _depth >= max_depth:
        return base_tour_fn(points)

    clusters = cluster_by_threshold(points, threshold)
    if len(clusters) <= 1:
        return base_tour_fn(points)

    centroids = np.array([points[c].mean(axis=0) for c in clusters])
    macro_order = base_tour_fn(centroids)
    m = len(macro_order)

    tour = []
    for pos, ci in enumerate(macro_order):
        idx = clusters[ci]
        if len(idx) == 1:
            tour.append(int(idx[0]))
            continue

        sub_tour = clustered_tour(points[idx], base_tour_fn, threshold, min_cluster_size,
                                  max_depth, optimize_endpoints, _depth + 1)
        if optimize_endpoints:
            anchor = points[tour[-1]] if tour else None
            next_ci = macro_order[(pos + 1) % m]
            next_anchor = centroids[next_ci] if pos + 1 < m else None
            sub_tour = _best_rotation(sub_tour, points[idx], anchor, next_anchor)

        tour.extend(int(idx[i]) for i in sub_tour)
    return tour


def orbit_recenter_clustered_tour(points, **kwargs):
    return clustered_tour(points, orbit_recenter_tour, **kwargs)


def orbit_recenter_clustered_v2_tour(points, **kwargs):
    """Clustered orbit-and-recenter with endpoint optimization."""
    return clustered_tour(points, orbit_recenter_tour, optimize_endpoints=True, **kwargs)


def shrink_wrap_clustered_tour(points, **kwargs):
    """Recursive clustering wrapped around the best shrink-wrap variant, with
    endpoint optimization -- tests whether clustering helps an already-good
    heuristic, not just a struggling one."""
    return clustered_tour(points, shrink_wrap_cheapest_2opt_tour, optimize_endpoints=True, **kwargs)
