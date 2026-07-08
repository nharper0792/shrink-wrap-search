def convex_hull_indices(points, idx):
    """Monotone-chain convex hull over points[idx]. Returns a CCW-ordered
    subsequence of idx. Collinear boundary points are dropped (they become
    interior points at the next recursion level instead)."""
    pts_idx = sorted(idx, key=lambda i: (points[i][0], points[i][1]))
    if len(pts_idx) <= 2:
        return list(pts_idx)

    def cross(o, a, b):
        ox, oy = points[o]
        ax, ay = points[a]
        bx, by = points[b]
        return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)

    lower = []
    for i in pts_idx:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], i) <= 0:
            lower.pop()
        lower.append(i)

    upper = []
    for i in reversed(pts_idx):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], i) <= 0:
            upper.pop()
        upper.append(i)

    return lower[:-1] + upper[:-1]
