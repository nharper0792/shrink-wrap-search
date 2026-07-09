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
    orbit_recenter_clustered_tour,
    orbit_recenter_tour,
    shrink_wrap_tour,
)
from tsp_lab.visualize import (
    animate_angular_sweep,
    animate_orbit_recenter,
    animate_shrink_wrap,
    plot_benchmark,
    plot_cluster_structure,
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

    tour, dt = timed(orbit_recenter_tour, points)
    results["orbit"] = {"tour": tour, "length": tour_length(points, tour), "time": dt}

    tour, dt = timed(orbit_recenter_clustered_tour, points)
    results["orbit_clustered"] = {"tour": tour, "length": tour_length(points, tour), "time": dt}

    tour, dt = timed(nearest_neighbor_2opt_tour, points)
    results["nn2opt"] = {"tour": tour, "length": tour_length(points, tour), "time": dt}

    order = ["nn2opt", "angular", "shrinkwrap", "orbit", "orbit_clustered"]
    if len(points) <= exact_max_n:
        tour, dt = timed(exact_tour, points)
        results["exact"] = {"tour": tour, "length": tour_length(points, tour), "time": dt}
        order = ["exact"] + order
    ordered = {k: results[k] for k in order}

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

    print(f"[1/6] Static comparison (n={args.n}, {args.distribution})...")
    path, ordered = run_static_comparison(points, args.outdir)
    print(f"  saved {path}")
    for k, r in ordered.items():
        print(f"    {k:16s} length={r['length']:.2f}  time={r['time']*1000:.3f} ms")

    if not args.skip_animations:
        anim_points = (random_points(args.anim_n, seed=args.anim_seed) if args.distribution == "uniform"
                       else clustered_points(args.anim_n, seed=args.anim_seed))

        print("[2/6] Angular-sort sweep animation...")
        p1 = animate_angular_sweep(anim_points, os.path.join(args.outdir, "angular_sweep.gif"))
        print(f"  saved {p1}")

        print("[3/6] Shrink-wrap peeling animation...")
        tour, trace = shrink_wrap_tour(anim_points, return_trace=True)
        p2 = animate_shrink_wrap(anim_points, tour, trace, os.path.join(args.outdir, "shrink_wrap.gif"))
        print(f"  saved {p2}")

        print("[4/6] Orbit & recenter animation...")
        tour, trace = orbit_recenter_tour(anim_points, return_trace=True)
        p3 = animate_orbit_recenter(anim_points, tour, trace, os.path.join(args.outdir, "orbit_recenter.gif"))
        print(f"  saved {p3}")
    else:
        print("[2/6] Skipping animations")
        print("[3/6] Skipping animations")
        print("[4/6] Skipping animations")

    print("[5/6] Cluster structure illustration...")
    cluster_demo_points = clustered_points(max(args.anim_n, 40), seed=args.anim_seed,
                                           n_clusters=5, spread=3.0)
    p4 = plot_cluster_structure(cluster_demo_points, os.path.join(args.outdir, "cluster_structure.png"))
    print(f"  saved {p4}")

    if not args.skip_benchmark:
        print(f"[6/6] Benchmark across n={args.benchmark_ns} ({args.benchmark_trials} trials each, uniform)...")
        records = run_benchmark(ns=args.benchmark_ns, trials=args.benchmark_trials, seed=args.seed)
        csv_path = save_csv(records, os.path.join(args.outdir, "benchmark.csv"))
        plot_path = plot_benchmark(records, os.path.join(args.outdir, "benchmark.png"))
        print(f"  saved {csv_path}")
        print(f"  saved {plot_path}")

        print(f"      Benchmark across n={args.benchmark_ns} ({args.benchmark_trials} trials each, clustered)...")
        cluster_gen = lambda n, seed: clustered_points(n, seed=seed, n_clusters=5, spread=3.0)
        records_c = run_benchmark(ns=args.benchmark_ns, trials=args.benchmark_trials, seed=args.seed,
                                  point_generator=cluster_gen)
        csv_path_c = save_csv(records_c, os.path.join(args.outdir, "benchmark_clustered.csv"))
        plot_path_c = plot_benchmark(records_c, os.path.join(args.outdir, "benchmark_clustered.png"))
        print(f"  saved {csv_path_c}")
        print(f"  saved {plot_path_c}")
    else:
        print("[6/6] Skipping benchmark")


if __name__ == "__main__":
    main()
