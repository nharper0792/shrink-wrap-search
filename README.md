# TSP shrink-wrap search

Three geometric TSP construction heuristics, compared against a real exact
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

3. **Orbit & Recenter** — a walker orbits the centroid of the still-unvisited
   points in a fixed rotational direction. On the first tiny step of a leg,
   any point whose distance to the walker just *increased* is disqualified
   (it's more than a half-turn "behind" the current heading); among the
   survivors, the walker keeps going until one point's distance stops
   decreasing — it just passed that point's closest approach — and that
   point is selected. After *every* selection the circle is recalculated
   from just the points still left (new centroid, new radius), and the
   walker marches from wherever it is onto that new circle, preserving its
   angular bearing, before resuming orbiting in the same direction. So the
   circle continuously shrinks and re-centers, one point at a time.
   `tsp_lab/heuristics.py::orbit_recenter_tour`

   This one went through two wrong turns worth recording. First attempt:
   recalculate only when a leg's disqualify step wipes out every remaining
   candidate ("gets stuck"). That trigger turns out to be mathematically
   unreachable — a point set's own centroid can never have an empty arc
   wider than 180° around it (otherwise the centroid wouldn't balance), and
   since the walker always consumes points in strict angular order, it can
   never open a gap bigger than what the data already had. Verified this
   both with a proof and empirically (several adversarial layouts, zero
   triggers). The actual intent was simpler: recalculate after *every*
   point, not just when stuck — that's what's implemented now, and it
   conveniently makes the "never stuck" guarantee unconditional (the circle
   is always exactly centered on the true centroid of the candidate set
   being tested, so the same guarantee applies fresh on every leg).

All three are compared against:

- **Exact (Held–Karp)** — DP, optimal, O(2^n · n²), practical to n ≈ 13.
- **Nearest-Neighbor + 2-opt** — a standard fast heuristic, used as the
  quality reference once n is too large to solve exactly.
- **Brute force** (`heuristics.brute_force_tour`) — literal permutation
  search, available for n ≲ 10, for when you want to see the actual
  factorial search rather than the DP.

## A note on "constant time"

None of these are O(1): every correct tour has to look at every point at
least once, so the floor is O(n). Angular Sort is O(n log n) (one sort).
Shrink-Wrap and Orbit & Recenter are both O(n²) worst case — Shrink-Wrap
scans every edge currently in the loop for each point it inserts; Orbit &
Recenter recomputes a centroid and rescans every remaining point on every
single leg. What they *avoid* is the combinatorial blow-up — O(n!) for
brute force, O(2^n · n²) for the exact DP — which is why they stay fast
into the hundreds of points where the exact methods become impossible.
"Constant time" isn't quite right; "doesn't get combinatorially worse" is.

## What the numbers actually show

From `output/benchmark.png` / `benchmark.csv` (median tour length ÷
best-known length, uniform random points, 5 trials per size):

| n  | Nearest-Neighbor+2opt | Angular Sort | Shrink-Wrap | Orbit & Recenter |
|----|------------------------|--------------|-------------|-------------------|
| 6  | 1.00                   | 1.00         | 1.00        | 1.00              |
| 10 | 1.00                   | 1.00         | 1.02        | 1.03              |
| 20 | 1.00                   | 1.16         | 1.03        | 1.15              |
| 50 | 1.00                   | 1.45         | 1.06        | 1.35              |
| 75 | 1.00                   | 1.65         | 1.10        | 1.88              |

Angular Sort degrades steadily as n grows — by n=75 it's running 65% longer
than nearest-neighbor+2-opt, because it never compares candidate edges
against each other, so nothing corrects a bad choice once made. Shrink-Wrap
holds up much better: it stays within ~10% of nearest-neighbor+2-opt even
at n=75, and clearly beats the other two past n≈15. That tracks with TSP
literature — inserting points in decreasing distance from the centroid at
their nearest edge is a geometric cousin of "farthest-point insertion,"
which is a genuinely competitive classic heuristic (the intuition being:
resolving the big, coarse structure of the tour first, then filling in
details, avoids the kind of long "return trip" edges that plague
nearest-neighbor-style greedy construction).

Orbit & Recenter is the surprise: it's competitive at small n but is
actually the *worst* of the three by n=75 — worse even than plain Angular
Sort. Watching the animation makes it clear why: recentering after every
single point means the reference frame (and the walker's heading within
it) shifts constantly, and the walker sometimes has to lunge across the
point cloud to reach the new "forward" candidate — the comparison image
and animation both show visible self-crossings that neither Angular Sort
nor Shrink-Wrap produce. Continuously re-centering sounds like it should
help (it's the same instinct behind Shrink-Wrap), but here it actively
fights the "keep moving in one direction" constraint rather than
complementing it. All three heuristics are still 1-3 orders of magnitude
faster than 2-opt at n=75 (see the runtime panel).

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
- `orbit_recenter.gif` — the dashed circle jumps and resizes after every
  point as it recenters on what's left; the open marker is the walker.
- `benchmark.png` / `benchmark.csv` — solution quality and runtime vs. n
  across repeated random instances.

## Layout

```
tsp_lab/
  geometry.py     point generation, tour length, centroid
  heuristics.py   all three heuristics + brute force / Held-Karp / NN+2-opt baselines
  benchmark.py    run all methods across n and seeds, save CSV
  visualize.py    static comparison plot, animations, benchmark plots
demo.py           CLI entry point that runs everything above
```
