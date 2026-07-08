#!/usr/bin/env python3
import argparse
import os
import time

from tsp_lab.benchmark import run_benchmark, save_csv
from tsp_lab.geometry import random_points, clustered_points, tour_length
from tsp_lab.heuristics import (
    angular_sort_tour,
    exact_tour,
    nearest_neighbor_2opt_tour,
    shrink_wrap_tour,
)
from tsp_lab.visualize import (
    animate_angular_sweep,
    animate_shrink_wrap,
    plot_benchmark,
    plot_static_comparison,
)


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def run_static_comparison(points, outdir, exact_max_n=13):
    results = {}

    tour, dt = timed(angular_sort_tour, points)
    results["angular"] = {"tour": tour, "length": tour_length(points, tour), "time": dt}

    tour, dt = timed(shrink_wrap_tour, points)
    results["shrinkwrap"] = {"tour": tour, "length": tour_length(points, tour), "time": dt}

    tour, dt = timed(nearest_neighbor_2opt_tour, points)
    results["nn2opt"] = {"tour": tour, "length": tour_length(points, tour), "time": dt}

    if len(points) <= exact_max_n:
        tour, dt = timed(exact_tour, points)
        results["exact"] = {"tour": tour, "length": tour_length(points, tour), "time": dt}
        ordered = {k: results[k] for k in ["exact", "nn2opt", "angular", "shrinkwrap"]}
    else:
        ordered = {k: results[k] for k in ["nn2opt", "angular", "shrinkwrap"]}

    path = os.path.join(outdir, "comparison.png")
    plot_static_comparison(
        points, ordered, save_path=path,
        title=f"TSP heuristics on n={len(points)} points",
        suptitle_note="% shown is length above the best tour found among these methods"
        + ("" if len(points) <= exact_max_n else " (no exact solver run — n too large)"),
    )
    return path, ordered


def main():
    ap = argparse.ArgumentParser(description="Demo: circle-wedge and shrink-wrap TSP heuristics")
    ap.add_argument("--n", type=int, default=12, help="points for the static comparison")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--distribution", choices=["uniform", "clustered"], default="uniform")
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--anim-n", type=int, default=16,
                     help="points for the animations (a bit larger, so shrink-wrap shows multiple peel layers)")
    ap.add_argument("--anim-seed", type=int, default=8)
    ap.add_argument("--skip-animations", action="store_true")
    ap.add_argument("--skip-benchmark", action="store_true")
    ap.add_argument("--benchmark-trials", type=int, default=5)
    ap.add_argument("--benchmark-ns", type=int, nargs="+",
                     default=[6, 8, 10, 12, 15, 20, 30, 50, 75])
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.distribution == "uniform":
        points = random_points(args.n, seed=args.seed)
    else:
        points = clustered_points(args.n, seed=args.seed)

    print(f"[1/4] Static comparison (n={args.n}, {args.distribution})...")
    path, ordered = run_static_comparison(points, args.outdir)
    print(f"  saved {path}")
    for k, r in ordered.items():
        print(f"    {k:12s} length={r['length']:.2f}  time={r['time']*1000:.3f} ms")

    if not args.skip_animations:
        anim_points = (random_points(args.anim_n, seed=args.anim_seed) if args.distribution == "uniform"
                       else clustered_points(args.anim_n, seed=args.anim_seed))

        print("[2/4] Angular-sort sweep animation...")
        p1 = animate_angular_sweep(anim_points, os.path.join(args.outdir, "angular_sweep.gif"))
        print(f"  saved {p1}")

        print("[3/4] Shrink-wrap peeling animation...")
        tour, trace = shrink_wrap_tour(anim_points, return_trace=True)
        p2 = animate_shrink_wrap(anim_points, tour, trace, os.path.join(args.outdir, "shrink_wrap.gif"))
        print(f"  saved {p2}")
    else:
        print("[2/4] Skipping animations")
        print("[3/4] Skipping animations")

    if not args.skip_benchmark:
        print(f"[4/4] Benchmark across n={args.benchmark_ns} ({args.benchmark_trials} trials each)...")
        records = run_benchmark(ns=args.benchmark_ns, trials=args.benchmark_trials, seed=args.seed)
        csv_path = save_csv(records, os.path.join(args.outdir, "benchmark.csv"))
        plot_path = plot_benchmark(records, os.path.join(args.outdir, "benchmark.png"))
        print(f"  saved {csv_path}")
        print(f"  saved {plot_path}")
    else:
        print("[4/4] Skipping benchmark")


if __name__ == "__main__":
    main()
