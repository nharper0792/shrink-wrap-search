#!/usr/bin/env python3
"""Measure Shrink-Wrap's O(n^2) vs the grid-bounded O(n) variant, and how
much quality is traded away for the speedup.
"""
import argparse
import os
import time

from tsp_lab.geometry import random_points, tour_length
from tsp_lab.heuristics import shrink_wrap_gridded_tour, shrink_wrap_tour
from tsp_lab.visualize import plot_scaling


def main():
    ap = argparse.ArgumentParser(description="Shrink-Wrap O(n^2) vs O(n) scaling test")
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--measured-ns", type=int, nargs="+",
                     default=[50, 100, 200, 400, 800, 1600, 3200, 6400])
    ap.add_argument("--gridded-only-ns", type=int, nargs="+",
                     default=[12800, 25600, 51200])
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    measured = []
    for n in args.measured_ns:
        pts = random_points(n, seed=args.seed)

        t0 = time.perf_counter()
        tour_base = shrink_wrap_tour(pts)
        time_baseline = time.perf_counter() - t0

        t0 = time.perf_counter()
        tour_grid, stats = shrink_wrap_gridded_tour(pts, return_stats=True)
        time_gridded = time.perf_counter() - t0

        len_base = tour_length(pts, tour_base)
        len_grid = tour_length(pts, tour_grid)
        cand = stats["candidates_examined"]
        mean_cand = sum(cand) / len(cand) if cand else 0.0

        print(f"n={n:6d}  baseline={time_baseline:9.4f}s  gridded={time_gridded:9.4f}s  "
              f"speedup={time_baseline / time_gridded:7.1f}x  "
              f"quality_ratio={len_grid / len_base:.3f}  mean_candidates={mean_cand:.1f}")

        measured.append({
            "n": n, "time_baseline": time_baseline, "time_gridded": time_gridded,
            "ratio_to_baseline": len_grid / len_base, "mean_candidates": mean_cand,
        })

    gridded_extra = []
    for n in args.gridded_only_ns:
        pts = random_points(n, seed=args.seed)
        t0 = time.perf_counter()
        shrink_wrap_gridded_tour(pts)
        time_gridded = time.perf_counter() - t0
        print(f"n={n:6d}  gridded={time_gridded:9.4f}s  (baseline not run -- would take too long)")
        gridded_extra.append({"n": n, "time_gridded": time_gridded})

    anchor = (measured[-1]["n"], measured[-1]["time_baseline"])
    plot_path = plot_scaling(measured, gridded_extra, os.path.join(args.outdir, "scaling.png"),
                             quadratic_fit_from=anchor)
    print(f"saved {plot_path}")


if __name__ == "__main__":
    main()
