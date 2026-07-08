# TSP shrink-wrap search

Two geometric TSP construction heuristics, compared against a real exact
solver and a real fast heuristic:

1. **Angular Sort ("wedge")** — draw a circle centered on the centroid of
   the points, radius = distance to the farthest point. Sweep the circle in
   one direction and visit points in the order the sweep passes them
   (equivalently: sort points by angle from the centroid).
   `tsp_lab/heuristics.py::angular_sort_tour`

2. **Shrink-Wrap ("vacuum bag")** — imagine a membrane shrinking onto the
   points from outside. It conforms to the convex hull first (those points
   "stick" first), then keeps sinking into each pocket between consecutive
   stuck points, conforming to the convex hull of whatever's left in that
   pocket, recursively, until every point is stuck. Implemented as
   recursive convex-hull peeling, with interior points assigned to a pocket
   by which pair of hull vertices their angle falls between.
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
O(n log n) (one sort). Shrink-Wrap is O(n log n) on average (each
recursion level does a convex hull in O(k log k) over a shrinking point
set). What they *avoid* is the combinatorial blow-up — O(n!) for brute
force, O(2^n · n²) for the exact DP — which is why they stay fast into the
hundreds or thousands of points where the exact methods become
impossible. "Constant time" isn't quite right; "doesn't get combinatorially
worse" is.

## What the numbers actually show

From `output/benchmark.png` / `benchmark.csv` (median tour length ÷
best-known length, uniform random points, 5 trials per size):

| n  | Nearest-Neighbor+2opt | Angular Sort | Shrink-Wrap |
|----|------------------------|--------------|-------------|
| 6  | 1.00                   | 1.00         | 1.00        |
| 10 | 1.00                   | 1.00         | 1.06        |
| 20 | 1.00                   | 1.16         | 1.27        |
| 50 | 1.00                   | 1.45         | 1.45        |
| 75 | 1.00                   | 1.65         | 1.70        |

Both heuristics find optimal or near-optimal tours on small instances, but
degrade steadily as n grows — by n=75 they're running 65-70% longer than
nearest-neighbor+2-opt. That's expected: neither one ever compares
candidate edges against each other, so nothing corrects a locally bad
choice. Angular Sort and Shrink-Wrap land in roughly the same quality band
as each other; Shrink-Wrap is consistently a little worse because each
recursive "pocket" is stitched back into the tour independently, which
adds detours nearest-neighbor-style methods don't have. Both are still
2-3 orders of magnitude faster than 2-opt at n=75 (see the runtime panel),
so the tradeoff is real: cheap and fast vs. tight and comparatively slower.

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
- `shrink_wrap.gif` — the recursive peeling animation for Shrink-Wrap
  (point color = recursion depth = how many layers deep that point was
  when the membrane reached it).
- `benchmark.png` / `benchmark.csv` — solution quality and runtime vs. n
  across repeated random instances.

## Layout

```
tsp_lab/
  geometry.py     point generation, tour length, centroid
  hull.py         monotone-chain convex hull
  heuristics.py   both heuristics + brute force / Held-Karp / NN+2-opt baselines
  benchmark.py    run all methods across n and seeds, save CSV
  visualize.py    static comparison plot, both animations, benchmark plots
demo.py           CLI entry point that runs everything above
```
