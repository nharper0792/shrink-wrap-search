# TSP shrink-wrap search

Two geometric TSP construction heuristics, compared against a real exact
solver and a real fast heuristic:

1. **Angular Sort ("wedge")** — draw a circle centered on the centroid of
   the points, radius = distance to the farthest point. Sweep the circle in
   one direction and visit points in the order the sweep passes them
   (equivalently: sort points by angle from the centroid).
   `tsp_lab/heuristics.py::angular_sort_tour`

2. **Shrink-Wrap ("vacuum bag")** — a circle centered on the centroid shrinks
   inward uniformly, so it contacts points strictly in order of decreasing
   distance from the centroid (farthest sticks first, closest sticks last).
   Each newly-stuck point is spliced into the loop of already-stuck points
   at whichever edge it forms the *shallowest triangle* with — the point's
   perpendicular distance to that edge segment (not the infinite line
   through it, so a point can't claim a distant-but-collinear edge). The
   membrane deforms least where the new point barely pokes above the
   nearest existing surface, and that's where it catches. This is a
   geometric variant of the classic "farthest-point insertion" heuristic.
   `tsp_lab/heuristics.py::shrink_wrap_tour`

Both are compared against:

- **Exact (Held–Karp)** — DP, optimal, O(2^n · n²), practical to n ≈ 13.
- **Nearest-Neighbor + 2-opt** — a standard fast heuristic, used as the
  quality reference once n is too large to solve exactly.
- **Brute force** (`heuristics.brute_force_tour`) — literal permutation
  search, available for n ≲ 10, for when you want to see the actual
  factorial search rather than the DP.

## A note on "constant time"

Both heuristics are fast, but they're not O(1): every correct tour has to
look at every point at least once, so the floor is O(n). Angular Sort is
O(n log n) (one sort). Shrink-Wrap is O(n²) worst case (for each of the n
points it scans every edge currently in the loop, and the loop grows by
one edge per point). What they *avoid* is the combinatorial blow-up —
O(n!) for brute force, O(2^n · n²) for the exact DP — which is why they
stay fast into the hundreds of points where the exact methods become
impossible. "Constant time" isn't quite right; "doesn't get combinatorially
worse" is.

## What the numbers actually show

From `output/benchmark.png` / `benchmark.csv` (median tour length ÷
best-known length, uniform random points, 5 trials per size):

| n  | Nearest-Neighbor+2opt | Angular Sort | Shrink-Wrap |
|----|------------------------|--------------|-------------|
| 6  | 1.00                   | 1.00         | 1.00        |
| 10 | 1.00                   | 1.00         | 1.02        |
| 20 | 1.00                   | 1.16         | 1.03        |
| 50 | 1.00                   | 1.45         | 1.06        |
| 75 | 1.00                   | 1.65         | 1.10        |

Angular Sort degrades steadily as n grows — by n=75 it's running 65% longer
than nearest-neighbor+2-opt, because it never compares candidate edges
against each other, so nothing corrects a bad choice once made. Shrink-Wrap
holds up much better: it stays within ~10% of nearest-neighbor+2-opt even
at n=75, and clearly beats Angular Sort past n≈15. That tracks with TSP
literature — inserting points in decreasing distance from the centroid at
their nearest edge is a geometric cousin of "farthest-point insertion,"
which is a genuinely competitive classic heuristic (the intuition being:
resolving the big, coarse structure of the tour first, then filling in
details, avoids the kind of long "return trip" edges that plague
nearest-neighbor-style greedy construction). Both heuristics are still
1-3 orders of magnitude faster than 2-opt at n=75 (see the runtime panel).

## Running it

```bash
pip install -r requirements.txt

# static comparison + both animations + benchmark, all in ./output
python demo.py --n 12 --seed 1

# just the benchmark, larger sweep
python demo.py --skip-animations --benchmark-ns 10 20 40 80 160 --benchmark-trials 8

# literal brute force instead of Held-Karp, for small n
python -c "
from tsp_lab.geometry import random_points, tour_length
from tsp_lab.heuristics import brute_force_tour
pts = random_points(9, seed=3)
tour = brute_force_tour(pts)
print(tour, tour_length(pts, tour))
"
```

Outputs land in `output/`:

- `comparison.png` — all methods on the same point set, tour drawn, length
  and %-above-best and wall-clock time labeled.
- `angular_sweep.gif` — the radial sweep animation for Angular Sort.
- `shrink_wrap.gif` — the shrinking-circle animation for Shrink-Wrap: the
  dashed circle's radius is the distance of the point currently being
  placed, and the dotted triangle shows the edge it just snapped into.
- `benchmark.png` / `benchmark.csv` — solution quality and runtime vs. n
  across repeated random instances.

## Layout

```
tsp_lab/
  geometry.py     point generation, tour length, centroid
  heuristics.py   both heuristics + brute force / Held-Karp / NN+2-opt baselines
  benchmark.py    run all methods across n and seeds, save CSV
  visualize.py    static comparison plot, both animations, benchmark plots
demo.py           CLI entry point that runs everything above
```
