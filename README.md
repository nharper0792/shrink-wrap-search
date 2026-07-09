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
Shrink-Wrap's O(n²) *is* a removable implementation cost, though, not a
fundamental one — see "Round 3" below, which gets it down to O(n).

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

## Round 3: can Shrink-Wrap run in O(1)?

No — placing n points is at least Ω(n) work no matter what, since writing
down an n-point tour takes n steps by itself. But `shrink_wrap_tour` is
O(n²), not because it has to be, but because of two removable costs: each
insertion (a) scans *every* edge currently in the path to find the
shallowest one, and (b) splices into a Python list with `.insert()`, which
is itself O(path length) since everything after it has to shift.

`shrink_wrap_gridded_tour` (`tsp_lab/heuristics.py`) fixes both: the path
is a doubly-linked list (O(1) splice), and each insertion only checks the
edges incident to a fixed number of nearby already-placed points (found via
a uniform spatial grid, sized once up front), instead of the whole path.
That candidate count is a hard cap — genuinely O(1) per step, not just
small — so the algorithm is O(n) overall. Verified empirically
(`scaling_test.py`, `output/scaling.png`): candidates examined per
insertion stay pinned at ~8 regardless of n (measured from n=50 to
n=6400), and wall-clock time diverges from the O(n²) baseline exactly like
a textbook n vs. n² comparison — 51× faster by n=6400 (1.4s vs. 71.6s),
and the gridded version keeps going past n=50,000 (24s) where the baseline
would take roughly two hours (quadratic extrapolation from the n=6400
point).

| n | baseline (O(n²)) | gridded (O(n)) | speedup | quality vs. baseline |
|---|---|---|---|---|
| 400 | 0.29s | 0.05s | 5.9× | 1.20× |
| 1,600 | 4.37s | 0.24s | 18.1× | 1.19× |
| 6,400 | 71.6s | 1.40s | 51.0× | 1.15× |
| 51,200 | ~2h (extrapolated) | 24.0s | — | — |

The cost: capping the search means it's no longer guaranteed to find the
globally shallowest edge, only the best one nearby, so tour length runs
about 5-20% longer than the full O(n²) search — but that cost stays in a
stable band as n grows rather than compounding, which is what makes the
trade worth it at scale. (For reference, the O(n²) baseline itself is
already only ~10% off best-known at moderate n — see the table above —
so the gridded version's absolute quality is still reasonable, just not
as tight.)

## Round 4: does the orbit radius matter?

Orbit & Recenter's walker starts each leg on a circle of some radius
around the current center. Does changing that radius — bounding (max
distance, the default), standard deviation, mean distance, midpoint of the
distance range, or arbitrary constants — change the resulting tour?

For the no-recenter version (one fixed circle, orbited the whole way
through), no: proven and verified (30/30 tests, 5 radii × 6 test
instances, every one producing the exact same tour length as
`angular_sort_tour`). The reason is a property of circles, not of this
algorithm specifically — the point on a circle closest to any target is
always along the ray from the circle's center through that target,
regardless of the circle's radius, so only the *center* determines
selection order.

For Orbit & Recenter (recentering after every pick, or every k picks), the
same turns out to be true, but the reason is different and was only found
by tracing the code: after every selection, the walker's position gets
overwritten with the true coordinates of the point it just picked — so the
synthetic, radius-dependent position computed during the *previous*
recentering step is never actually read by anything that affects a
decision. It only ever appears in the animation trace.

Testing this surfaced a real bug, though: one guard in the recentering
logic checked `new_R > 1e-12` to decide whether to compute a fresh bearing
or reuse the old one — meant to catch "the remaining points have collapsed
to a single location," but actually keyed off whichever radius metric was
in use. Standard deviation is 0 for *any* two points relative to their own
midpoint (they're always equidistant from it), so whenever exactly two
points remained, the std-dev radius setting would silently reuse a stale
bearing instead of computing the correct one — a genuine behavior
difference between radius choices, but a numerical artifact, not a
geometric one. Fixed by checking the actual degenerate condition (walker
position ≈ new center) instead of the radius value; verified clean
afterward (0/840 runs differed across 14 seeds × 5 recenter-frequencies ×
2 data types × 6 radius functions, once the guard checks the right thing).
Confirmed the fix doesn't change any previously-reported result for the
default (bounding-radius) setting on ordinary data — it only fires in the
degenerate two-point case.

`orbit_recenter_tour` now takes a `radius_fn(points, center)` parameter for
anyone who wants to keep experimenting with it, though the finding is that
it's a free parameter with no effect on output — the real levers are the
center (recentering) and the recenter frequency, both already covered in
Round 2.

## Round 5: marching along a tangent line instead of snapping radially

The default recentering behavior "teleports" the walker onto the new
circle at the angle that preserves its bearing from the old center. An
alternative: instead of preserving the angle, have the walker continue in
a straight line along the direction it was *already heading* (the tangent
direction of the old circle at its current position) until that line
grazes the new circle, then resume orbiting from there. A point outside a
circle has exactly two tangent lines to it; `orbit_recenter_tangent_tour`
(`tangent_snap=True` on `orbit_recenter_tour`) picks whichever one bends
the walker's path least. Falls back to the radial snap in the (rare) case
where the walker ends up inside the new circle, where no real tangent
line exists.

It's a real behavior change this time — unlike the radius experiments,
tangent-line marching is **consistently worse**: median 1.22-1.31×
longer than the radial-snap default across n=10-75 (20 seeds each,
uniform random points), beating the default in fewer than 5% of trials
even at n=75.

The reason is worth spelling out, because it's the same trap the
crossing-avoidance idea (Round 2) fell into from a different angle: this
algorithm has no notion of "distance traveled by the walker" in its cost
function. `tour_length` only sums distances between the *actual selected
points*, in order — the virtual march between circles is pure
bookkeeping that exists only to produce a bearing (`cur_theta`) for the
next selection. A "smoother," less sharply-bending path sounds like it
should help, by analogy with a real walker minimizing travel — but
nothing here is minimizing travel; the only thing that matters is which
angle the walker arrives at, because that determines which point gets
picked next. Preserving the angle directly (the radial snap) turns out to
correlate with useful next-picks better than deriving a new angle from a
tangent-line construction that optimizes for path smoothness instead.

## Round 6: does Shrink-Wrap actually beat standard practice at high volume?

Earlier in this project, `shrink_wrap_gridded_tour` was compared against
`nearest_neighbor_2opt_tour` and won on speed at scale (51x faster at
n=6,400). That comparison was honest but not fair: naive nearest-neighbor
is O(n²) and full 2-opt is O(n²) per pass, so `nearest_neighbor_2opt_tour`
was never a real stand-in for "standard practice at high volume" — nobody
actually runs that at n=80,000. This round builds the techniques real
large-scale TSP practice actually uses and re-runs the comparison fairly:

- **Hilbert curve sort** (`hilbert_curve_tour`) — sort points by position
  along a space-filling curve. O(n log n), no iteration at all.
- **Grid-accelerated nearest-neighbor** (`nearest_neighbor_fast_tour`) —
  the same greedy rule as before, but each step queries the spatial grid
  (built once) instead of scanning every remaining point. Verified to
  produce byte-identical output to the slow version, just faster.
- **Neighbor-list-restricted 2-opt** (`neighbor_list_2opt`) — 2-opt
  checked only against each point's k-nearest-neighbor candidate list
  (k=8) instead of all O(n) other edges. This candidate-list restriction
  is the actual mechanism real solvers (Lin-Kernighan-style) use to make
  local search scale — "check every pair" was never standard practice.

Benchmarked all three, each with and without the neighbor-list 2-opt
cleanup, against `shrink_wrap_gridded_tour` (+ the same cleanup) from
n=1,000 to n=80,000 (`high_volume_test.py`, `output/high_volume.png`).
Median ratio to the best tour found at each size:

| n | Hilbert | Hilbert+2opt | NN (fast) | NN (fast)+2opt | Shrink-Wrap (gridded) | Shrink-Wrap+2opt |
|---|---|---|---|---|---|---|
| 1,000 | 1.24× | 1.09× | 1.14× | **1.00×** | 1.47× | 1.04× |
| 10,000 | 1.24× | 1.06× | 1.12× | **1.00×** | 1.50× | 1.06× |
| 80,000 | 1.25× | 1.06× | 1.13× | **1.00×** | 1.52× | 1.06× |

The honest verdict: **no.** On raw construction, grid-accelerated
nearest-neighbor beats both Hilbert curve and gridded Shrink-Wrap at
every single size tested — Shrink-Wrap's grid-bounded search is
consistently the *worst* of the three raw constructions, not competitive
with plain nearest-neighbor the way the O(n²) version looked earlier.
After the same cleanup pass, nearest-neighbor + neighbor-list 2-opt is
the best of all six methods at every size, without exception. Shrink-Wrap
+ cleanup lands roughly tied with Hilbert + cleanup — sometimes a hair
ahead, sometimes a hair behind — but it's also consistently the
*slowest* full pipeline of the three at large n (78.7s vs. Hilbert's
31.3s and nearest-neighbor's 67.8s at n=80,000), because it pays for both
the slowest raw construction and a comparable cleanup cost.

So the earlier "51x faster than standard practice" claim doesn't survive
contact with what standard practice actually is at volume. Against a
lazy, unaccelerated baseline, Shrink-Wrap's speedup looked dramatic; against
the real thing, it's slower and lower-quality than a textbook
nearest-neighbor-plus-candidate-list pipeline, and even the trivial
Hilbert-curve sort — one line of bit-interleaving — beats it on raw
construction speed by two orders of magnitude while landing in the same
final quality range after cleanup. Shrink-Wrap remains the most
interesting *construction rule* of everything tried in this project
(Round 1-2), but "the standard, boring techniques, implemented properly"
is still the thing to reach for at real high-volume scale.

## Round 7: a fourth idea — wedge & radial fragments

A new construction idea, tested as its own family from scratch:

1. Take the centroid of whatever points are still unplaced.
2. Select the point closest to that centroid (the "anchor").
3. Capture every other unplaced point within an angular wedge around the
   centroid-to-anchor direction, and order the captured set (anchor
   included) by the *signed* angular offset from that direction — negative
   offset = "before" the anchor, positive = "after", so the anchor always
   sits in the middle of a small fan-shaped fragment rather than at one
   end. (An earlier draft ordered by distance from the centroid instead;
   corrected after clarification — see `plot_wedge_fragments`, which shows
   the fragments visibly fanning out from each anchor once fixed, instead
   of the radial spikes the first version produced.)
4. Repeat against whatever's left until every point belongs to some
   fragment, then merge fragments into one tour by repeatedly joining
   whichever two fragment endpoints (from different fragments, checking
   all four head/tail combinations) are closest, until one remains.

`wedge_radial_tour`. A second variant, `wedge_tip_tour`, moves the
wedge's apex from the centroid onto the anchor point itself — a cone
reaching outward from the anchor rather than a pie-slice through the
centroid — ordering its captures by distance from the anchor (nearest
first) instead of angle, since a point-apex cone has no "before" side to
fan into. A cone from a point sweeps far less area than a pie-slice from
the centroid at the same angle, so it needs a wider angle to capture a
comparable number of points per cycle; tested angles from 2° to 150°
anyway, and — for both variants — *narrower* consistently wins (best
around 8-15°), with quality degrading smoothly as the wedge widens rather
than there being a "wide is better" regime to find.

The distinctive, and frankly rough, result: **raw construction quality
degrades sharply with n**, unlike every other heuristic in this project
(Angular Sort, Shrink-Wrap, Orbit & Recenter, Hilbert, nearest-neighbor —
all stay roughly flat, 1.1-1.5× best-known regardless of n). Wedge &
Radial Fragments instead climbs from ~2.5× at n=200 to ~24× at n=20,000
(`wedge_radial`) — worse than everything else tested in this entire
project by a wide margin, and the only heuristic whose relative quality
visibly worsens with scale rather than holding steady.

| n | Wedge&Radial (raw) | +2-opt | Wedge-tip (raw) | +2-opt | NN(fast)+2opt | Hilbert+2opt | Shrink-Wrap(gridded)+2opt |
|---|---|---|---|---|---|---|---|
| 200 | 2.47× | 1.01× | 1.48× | 1.06× | 1.00× | 1.10× | 1.08× |
| 1,000 | 5.40× | 1.03× | 1.99× | 1.06× | 1.00× | 1.11× | 1.06× |
| 5,000-10,000 | 16.1× | 1.06× | 3.79× | 1.04× | 1.00× | 1.06× | 1.06× |
| 20,000 | 23.5× | 1.10× | — | — | 1.00× | 1.07× | 1.07× |

But cleaned up, it's a genuinely different story: at small-to-medium n
(200-3,000), **`wedge_radial_2opt` is the best of the non-nearest-neighbor
methods**, edging out both Hilbert+2-opt and Shrink-Wrap(gridded)+2-opt —
the 2-opt cleanup pass fully absorbs the rough construction. That edge
erodes and reverses by n=20,000 (1.10× vs. Hilbert's 1.07× and
Shrink-Wrap's 1.07×), for two compounding reasons: the raw construction
damage grows faster than 2-opt's ability to fix it, and — `wedge_tip`
especially — narrow capture cones mean many more fragments (275 at
n=30,000 vs. `wedge_radial`'s 63 at the same size), and the fragment
-merging step is O(F³), so more fragments costs real time: `wedge_tip`
takes 35s at n=30,000 against `wedge_radial`'s 0.47s for the same raw
construction step, and `wedge_radial_2opt` is consistently the *slowest*
full pipeline of anything benchmarked in Round 6 or 7 at large n.

So: a construction rule that's bad on its own but gets fully rescued by
local search at moderate scale, and genuinely the worst-scaling idea in
the whole project once you actually push n up — a useful reminder that
"looks fine after cleanup at n=1,000" and "still fine at n=20,000" are
different claims, and this project's benchmarks now go far enough to
tell them apart.

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
- `scaling.png` — Shrink-Wrap's O(n²) baseline vs. the grid-bounded O(n)
  variant, runtime (log-log) and quality cost vs. n up to 51,200 points.
- `high_volume.png` / `high_volume.csv` — Shrink-Wrap (gridded) vs. actual
  standard practice (Hilbert curve, grid-accelerated nearest-neighbor,
  neighbor-list-restricted 2-opt) from n=1,000 to n=80,000.
- `wedge_fragments.png` / `wedge_tip_fragments.png` — the fragments Wedge &
  Radial Fragments builds before merging, one color per fragment (open
  marker = anchor point) — the centroid-anchored version fans out from
  each anchor, the point-apex version spikes outward from it.
- `wedge_comparison.png` / `wedge_comparison_cleaned.png` (+ `.csv`) —
  Wedge & Radial Fragments (raw and +2-opt) against Hilbert curve,
  nearest-neighbor, and Shrink-Wrap (all gridded/fast + cleaned), n=200 to
  n=20,000. The "cleaned" chart omits the raw construction lines so the
  competitive +2-opt comparison is actually legible.

## Running it

```bash
pip install -r requirements.txt

# static comparison + animations + both benchmark sweeps, all in ./output
python demo.py --n 12 --seed 1

# just the benchmarks, larger sweep
python demo.py --skip-animations --benchmark-ns 10 20 40 80 160 --benchmark-trials 8

# test every improvement idea against its family, then compare the winners
python families.py --outdir output

# O(n^2) vs O(n) scaling test for Shrink-Wrap (takes a few minutes -- runs
# the O(n^2) baseline up to n=6400)
python scaling_test.py --outdir output

# Shrink-Wrap (gridded) vs actual standard practice at high volume
# (Hilbert curve, grid-accelerated NN, neighbor-list 2-opt), n up to 80,000
# -- takes several minutes
python high_volume_test.py --outdir output

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
  heuristics.py   all three original heuristics + wedge & radial fragments
                  (centroid-apex and point-apex variants) + their improvement
                  variants + the clustering wrapper + brute force / Held-Karp /
                  NN+2-opt / 2-opt / Or-opt baselines and local search +
                  standard-practice-at-scale baselines (Hilbert curve,
                  grid-accelerated NN, neighbor-list 2-opt)
  benchmark.py    run a set of methods across n and seeds, save CSV
  visualize.py    static comparison plot, animations, benchmark plots,
                  wedge fragment plots
demo.py               CLI entry point for the original three-heuristic demo
families.py           CLI entry point for the improvement-idea testing round
scaling_test.py       CLI entry point for the O(n^2) vs O(n) scaling test
high_volume_test.py   CLI entry point for the standard-practice-at-scale comparison
```
