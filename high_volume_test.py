#!/usr/bin/env python3
"""Empirical comparison of our Shrink-Wrap approaches against actual
standard practice for high-volume TSP (n in the thousands to hundreds of
thousands) -- not the O(n^2) nn2opt used as a reference elsewhere in this
project, which itself doesn't scale past a few thousand points.
"""
import argparse
import csv
import os
import time

from tsp_lab.geometry import random_points, tour_length
from tsp_lab.heuristics import (
    hilbert_curve_2opt_tour,
    hilbert_curve_tour,
    nearest_neighbor_fast_2opt_tour,
    nearest_neighbor_fast_tour,
    shrink_wrap_gridded_2opt_tour,
    shrink_wrap_gridded_tour,
)
from tsp_lab.visualize import plot_benchmark

METHODS = {
    "hilbert": hilbert_curve_tour,
    "hilbert_2opt": hilbert_curve_2opt_tour,
    "nn_fast": nearest_neighbor_fast_tour,
    "nn_fast_2opt": nearest_neighbor_fast_2opt_tour,
    "shrinkwrap_gridded": shrink_wrap_gridded_tour,
    "shrinkwrap_gridded_2opt": shrink_wrap_gridded_2opt_tour,
}


def run(ns_trials, seed, outdir):
    records = []
    rng_seed = seed
    for n, trials in ns_trials:
        for trial in range(trials):
            rng_seed += 1
            pts = random_points(n, seed=rng_seed)
            lengths = {}
            for name, fn in METHODS.items():
                t0 = time.perf_counter()
                tour = fn(pts)
                dt = time.perf_counter() - t0
                L = tour_length(pts, tour)
                lengths[name] = L
                records.append({"n": n, "trial": trial, "seed": rng_seed, "method": name,
                                 "length": L, "time": dt})
                print(f"  n={n:7d} trial={trial} {name:24s} time={dt:8.3f}s len={L:12.1f}", flush=True)
            best = min(lengths.values())
            for r in records:
                if r["n"] == n and r["trial"] == trial:
                    r["ratio_to_best"] = r["length"] / best
    csv_path = os.path.join(outdir, "high_volume.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n", "trial", "seed", "method", "length", "time", "ratio_to_best"])
        w.writeheader()
        w.writerows(records)
    print("saved", csv_path)

    plot_path = plot_benchmark(
        records, os.path.join(outdir, "high_volume.png"),
        methods=("hilbert", "hilbert_2opt", "nn_fast", "nn_fast_2opt",
                "shrinkwrap_gridded", "shrinkwrap_gridded_2opt"),
    )
    print("saved", plot_path)
    return records


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ns_trials = [
        (1000, 5),
        (3000, 4),
        (10000, 3),
        (20000, 2),
        (40000, 1),
        (80000, 1),
    ]
    run(ns_trials, args.seed, args.outdir)
