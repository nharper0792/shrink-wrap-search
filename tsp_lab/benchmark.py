import csv
import time

from .geometry import clustered_points, random_points, tour_length
from .heuristics import (
    angular_sort_tour,
    exact_tour,
    nearest_neighbor_2opt_tour,
    orbit_recenter_clustered_tour,
    orbit_recenter_tour,
    shrink_wrap_tour,
)

METHODS = {
    "angular": angular_sort_tour,
    "shrinkwrap": shrink_wrap_tour,
    "orbit": orbit_recenter_tour,
    "orbit_clustered": orbit_recenter_clustered_tour,
    "nn2opt": nearest_neighbor_2opt_tour,
}

EXACT_MAX_N = 13  # Held-Karp is O(2^n n^2); keep the benchmark fast


def run_benchmark(ns=(6, 8, 10, 12, 15, 20, 30, 50, 75), trials=5, seed=0, point_generator=None,
                  methods=None):
    """Run every method on `trials` random instances per n. When n is small
    enough, the Held-Karp exact solution is used as the quality reference
    ('best'); otherwise the best tour found by any method on that instance
    stands in as the reference, and is labeled accordingly.

    `point_generator(n, seed=...)` defaults to `random_points` (uniform);
    pass `clustered_points` to see how the methods -- especially
    orbit_clustered -- do on data with genuine cluster structure.

    `methods` defaults to the module-level METHODS dict; pass a custom
    {name: fn} dict to benchmark a different set (e.g. one heuristic
    family's variants)."""
    point_generator = point_generator or random_points
    methods = METHODS if methods is None else methods
    records = []
    rng_seed = seed
    for n in ns:
        for trial in range(trials):
            rng_seed += 1
            pts = point_generator(n, seed=rng_seed)
            lengths = {}

            for name, fn in methods.items():
                t0 = time.perf_counter()
                tour = fn(pts)
                dt = time.perf_counter() - t0
                L = tour_length(pts, tour)
                lengths[name] = L
                records.append({"n": n, "trial": trial, "seed": rng_seed, "method": name,
                                 "length": L, "time": dt})

            if n <= EXACT_MAX_N:
                t0 = time.perf_counter()
                tour = exact_tour(pts)
                dt = time.perf_counter() - t0
                L = tour_length(pts, tour)
                lengths["exact"] = L
                records.append({"n": n, "trial": trial, "seed": rng_seed, "method": "exact",
                                 "length": L, "time": dt})

            best = min(lengths.values())
            for r in records:
                if r["n"] == n and r["trial"] == trial:
                    r["ratio_to_best"] = r["length"] / best

    return records


def save_csv(records, path):
    if not records:
        return path
    fields = ["n", "trial", "seed", "method", "length", "time", "ratio_to_best"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return path
