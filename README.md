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

Plus a meta-heuristic that wraps any of the three: **recursive clustering**
(`tsp_lab/heuristics.py::clustered_tour`, convenience wrapper
`orbit_recenter_clustered_tour`). Group points into clusters of mutually
"sufficiently close" points (connected components under a distance
threshold), collapse each cluster to its centroid, run the *same* tour
function on those centroids to get a macro-order of clusters, then recurse
into each cluster to order its actual points the same way, and concatenate
in macro-order. The threshold is picked automatically from the minimum
spanning tree: sort the MST edge weights and cut at the single biggest
relative jump between consecutive weights. That's the standard trick for
single-linkage clustering without a pre-chosen cluster count — it finds a
handful of clusters on genuinely separated data, and degrades to
near-singleton "clusters" (i.e. a no-op, since `clustered_tour` falls back
to running the base algorithm directly once clustering finds nothing) on
data with no real structure, since there's no standout gap to cut at.
`output/cluster_structure.png` shows what the threshold actually finds on
a 5-blob synthetic instance.

All of the above are compared against:

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

| n  | NN+2opt | Angular Sort | Shrink-Wrap | Orbit & Recenter | Orbit + clustering |
|----|---------|--------------|-------------|-------------------|---------------------|
| 6  | 1.00    | 1.00         | 1.00        | 1.00              | 1.04                |
| 10 | 1.00    | 1.00         | 1.02        | 1.03              | 1.07                |
| 20 | 1.00    | 1.16         | 1.03        | 1.15              | 1.20                |
| 50 | 1.00    | 1.45         | 1.06        | 1.35              | 1.45                |
| 75 | 1.00    | 1.65         | 1.10        | 1.88              | 1.79                |

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
complementing it. All heuristics are still 1-3 orders of magnitude faster
than 2-opt at n=75 (see the runtime panel).

Wrapping Orbit & Recenter in recursive clustering doesn't help on *uniform*
data — slightly worse at every size in the table above, which makes sense:
there's no real cluster structure to exploit, so the MST-gap threshold
mostly just carves the cloud into small arbitrary groups and adds macro/
micro-stitching overhead for nothing. But on data with genuine cluster
structure (`output/benchmark_clustered.png`, five well-separated Gaussian
blobs) it's a real, if partial, fix: at n=75, plain Orbit & Recenter scores
1.74 vs. clustering's 1.30 — most of the way back to Shrink-Wrap's 1.13 on
the same data. The clustering wrapper doesn't change what happens *inside*
a cluster (the self-crossing problem can still happen there), but it stops
the walker from being dragged across the whole canvas by a distant
recentering — each cluster gets solved as its own small, contained problem,
and only the macro-order between cluster centroids has to deal with the
long-range structure.

## Round 2: testing improvement ideas per family

Each heuristic got 2-4 targeted improvement ideas, implemented in
`tsp_lab/heuristics.py` and tested against its own family's baseline via
`families.py` (which also produces a final comparison of each family's
winner against the others). Full results in `output/family_*.png` and
`output/final_comparison_*.png`; the short version:

- **Shrink-Wrap**: `shrink_wrap_cheapest_tour` (textbook cheapest-insertion
  metric instead of geometric shallowness) is a modest win. The much bigger
  win is a plain 2-opt cleanup pass — `shrink_wrap_2opt_tour` lands at
  1.00-1.04× best-known at every size tested, essentially solving the
  problem outright.
- **Angular Sort**: `angular_sort_oropt_tour` (an Or-opt pass — relocate
  one out-of-place point at a time, targeting exactly the radial zigzag
  visible in the animation) is dramatic: from 1.63× best-known at n=75 down
  to ~1.04×.
- **Orbit & Recenter**: recentering less often (every 3 or 10 points
  instead of every point) gave mixed, inconsistent results — not a
  reliable fix. The crossing-avoidance idea, tested as literally specified
  (reject a candidate if its edge crosses an already-placed edge), turned
  out to be a **provable no-op**: it produced byte-identical tours to the
  baseline across every test instance. The reason is structural — every
  self-crossing found (209 of them, across 234 sampled instances) involved
  the *closing* edge (last point back to the first), which only forms
  after the walk ends and is invisible to any per-step check. A plain
  2-opt cleanup pass (`orbit_recenter_2opt_tour`) *does* see that edge and
  fixes it completely (verified: 0 crossings across the same sample,
  down from 51), taking Orbit & Recenter from the worst heuristic
  (1.71× best-known at n=75) to competitive (1.03-1.05×).
- **Clustering**: adding endpoint optimization (rotate each cluster's
  sub-tour to face its macro-neighbors before splicing it in) gave a small
  further improvement. The bigger lever was *which* heuristic gets
  clustered — wrapping the already-strong Shrink-Wrap (cheapest-insertion
  + 2-opt) instead of the weak Orbit & Recenter
  (`shrink_wrap_clustered_tour`) reaches 1.00-1.05× best-known on both
  uniform and genuinely clustered data.

The pattern across all four families is the same: **every geometric
construction idea benefits enormously from being paired with a cheap local
-search cleanup pass** (2-opt or Or-opt). The constructive heuristics differ
a lot in how good a *starting point* they hand to local search, but once
local search runs, most of that gap closes. `final_comparison_uniform.png`
/ `final_comparison_clustered.png` put every family's winner on the same
axes: with a cleanup pass attached, all three families land in roughly the
same 1.00-1.05× best-known neighborhood, and Angular Sort — the "worst"
original idea — improves the most in relative terms (65% too long →
essentially optimal) because it started from the simplest, most
easily-corrected construction.

## Outputs

Outputs land in `output/`:

- `comparison.png` — all methods on the same point set, tour drawn, length
  and %-above-best and wall-clock time labeled.
- `angular_sweep.gif` — the radial sweep animation for Angular Sort.
- `shrink_wrap.gif` — the shrinking-circle animation for Shrink-Wrap: the
  dashed circle's radius is the distance of the point currently being
  placed, and the dotted triangle shows the edge it just snapped into.
- `orbit_recenter.gif` — the dashed circle jumps and resizes after every
  point as it recenters on what's left; the open marker is the walker.
- `cluster_structure.png` — what the MST-gap threshold finds on a synthetic
  5-blob instance (each color/✕ is one cluster and its centroid).
- `benchmark.png` / `benchmark.csv` — solution quality and runtime vs. n,
  uniform random instances.
- `benchmark_clustered.png` / `benchmark_clustered.csv` — the same sweep on
  genuinely clustered instances, to see where the clustering wrapper
  actually earns its keep.
- `family_shrinkwrap.png`, `family_angular.png`, `family_orbit.png`,
  `family_clustering.png` / `family_clustering_clustered.png` (+ matching
  `.csv`) — each family's improvement ideas benchmarked against its own
  baseline.
- `final_comparison_uniform.png` / `final_comparison_clustered.png` (+
  `.csv`) — every family's winning variant plotted against the others.

## Running it

```bash
pip install -r requirements.txt

# static comparison + animations + both benchmark sweeps, all in ./output
python demo.py --n 12 --seed 1

# just the benchmarks, larger sweep
python demo.py --skip-animations --benchmark-ns 10 20 40 80 160 --benchmark-trials 8

# test every improvement idea against its family, then compare the winners
python families.py --outdir output

# literal brute force instead of Held-Karp, for small n
python -c "
from tsp_lab.geometry import random_points, tour_length
from tsp_lab.heuristics import brute_force_tour
pts = random_points(9, seed=3)
tour = brute_force_tour(pts)
print(tour, tour_length(pts, tour))
"

# wrap any of the three heuristics in recursive clustering
python -c "
from tsp_lab.geometry import clustered_points, tour_length
from tsp_lab.heuristics import clustered_tour, shrink_wrap_tour
pts = clustered_points(80, seed=1, n_clusters=6, spread=3.0)
tour = clustered_tour(pts, shrink_wrap_tour)
print(tour_length(pts, tour))
"
```

## Layout

```
tsp_lab/
  geometry.py     point generation, tour length, centroid
  heuristics.py   all three heuristics + their improvement variants +
                  the clustering wrapper + brute force / Held-Karp /
                  NN+2-opt / 2-opt / Or-opt baselines and local search
  benchmark.py    run a set of methods across n and seeds, save CSV
  visualize.py    static comparison plot, animations, benchmark plots
demo.py           CLI entry point for the original three-heuristic demo
families.py       CLI entry point for the improvement-idea testing round
```
