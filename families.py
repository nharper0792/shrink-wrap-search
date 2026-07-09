#!/usr/bin/env python3
"""Test every improvement idea against its own family's baseline, then pick
the winner from each family and compare those winners against each other.
"""
import argparse
import csv
import os
import statistics as st
from collections import defaultdict

from tsp_lab.benchmark import run_benchmark, save_csv
from tsp_lab.geometry import clustered_points
from tsp_lab.heuristics import (
    angular_sort_oropt_tour,
    angular_sort_tour,
    exact_tour,
    nearest_neighbor_2opt_tour,
    orbit_recenter_2opt_tour,
    orbit_recenter_clustered_tour,
    orbit_recenter_clustered_v2_tour,
    orbit_recenter_noncross_tour,
    orbit_recenter_tour,
    shrink_wrap_2opt_tour,
    shrink_wrap_cheapest_2opt_tour,
    shrink_wrap_cheapest_tour,
    shrink_wrap_clustered_tour,
    shrink_wrap_tour,
)
from tsp_lab.visualize import plot_benchmark


def _recenter_every(k):
    def fn(points):
        return orbit_recenter_tour(points, recenter_every=k)
    return fn


FAMILIES = {
    "shrinkwrap": {
        "methods": {
            "shrinkwrap": shrink_wrap_tour,
            "shrinkwrap_cheapest": shrink_wrap_cheapest_tour,
            "shrinkwrap_2opt": shrink_wrap_2opt_tour,
            "shrinkwrap_cheapest_2opt": shrink_wrap_cheapest_2opt_tour,
            "nn2opt": nearest_neighbor_2opt_tour,
        },
        "plot_methods": ("exact", "nn2opt", "shrinkwrap", "shrinkwrap_cheapest",
                         "shrinkwrap_2opt", "shrinkwrap_cheapest_2opt"),
        "baseline": "shrinkwrap",
        "candidates": ["shrinkwrap_cheapest", "shrinkwrap_2opt", "shrinkwrap_cheapest_2opt"],
        "datasets": ["uniform"],
    },
    "angular": {
        "methods": {
            "angular": angular_sort_tour,
            "angular_oropt": angular_sort_oropt_tour,
            "nn2opt": nearest_neighbor_2opt_tour,
        },
        "plot_methods": ("exact", "nn2opt", "angular", "angular_oropt"),
        "baseline": "angular",
        "candidates": ["angular_oropt"],
        "datasets": ["uniform"],
    },
    "orbit": {
        "methods": {
            "orbit": orbit_recenter_tour,
            "orbit_every3": _recenter_every(3),
            "orbit_every10": _recenter_every(10),
            "orbit_noncross": orbit_recenter_noncross_tour,
            "orbit_2opt": orbit_recenter_2opt_tour,
            "nn2opt": nearest_neighbor_2opt_tour,
        },
        "plot_methods": ("exact", "nn2opt", "orbit", "orbit_every3", "orbit_every10",
                         "orbit_noncross", "orbit_2opt"),
        "baseline": "orbit",
        "candidates": ["orbit_every3", "orbit_every10", "orbit_noncross", "orbit_2opt"],
        "datasets": ["uniform"],
    },
    "clustering": {
        "methods": {
            "orbit": orbit_recenter_tour,
            "orbit_clustered": orbit_recenter_clustered_tour,
            "orbit_clustered_v2": orbit_recenter_clustered_v2_tour,
            "shrinkwrap": shrink_wrap_tour,
            "shrinkwrap_clustered": shrink_wrap_clustered_tour,
            "nn2opt": nearest_neighbor_2opt_tour,
        },
        "plot_methods": ("exact", "nn2opt", "orbit", "orbit_clustered", "orbit_clustered_v2",
                         "shrinkwrap", "shrinkwrap_clustered"),
        "baseline": "orbit_clustered",
        "candidates": ["orbit_clustered_v2", "shrinkwrap_clustered"],
        "datasets": ["uniform", "clustered"],
    },
}

DATASET_GENERATORS = {
    "uniform": None,  # run_benchmark's default (random_points)
    "clustered": lambda n, seed: clustered_points(n, seed=seed, n_clusters=5, spread=3.0),
}


def median_ratio_at_n(records, method, n):
    vals = [r["ratio_to_best"] for r in records if r["method"] == method and r["n"] == n]
    return st.median(vals) if vals else None


def run_families(outdir, ns, trials, seed):
    winners = {}
    all_family_records = {}

    for family_name, spec in FAMILIES.items():
        for dataset in spec["datasets"]:
            gen = DATASET_GENERATORS[dataset]
            suffix = "" if dataset == "uniform" else f"_{dataset}"
            print(f"[family: {family_name}{suffix}] running benchmark...")
            records = run_benchmark(ns=ns, trials=trials, seed=seed,
                                    methods=spec["methods"], point_generator=gen)
            csv_path = save_csv(records, os.path.join(outdir, f"family_{family_name}{suffix}.csv"))
            plot_path = plot_benchmark(records, os.path.join(outdir, f"family_{family_name}{suffix}.png"),
                                       methods=spec["plot_methods"])
            print(f"  saved {csv_path}")
            print(f"  saved {plot_path}")
            all_family_records[(family_name, dataset)] = records

        # decide the winner using the *last* n on the *first* (primary) dataset
        primary_records = all_family_records[(family_name, spec["datasets"][0])]
        n_max = max(ns)
        baseline_ratio = median_ratio_at_n(primary_records, spec["baseline"], n_max)
        best_name, best_ratio = spec["baseline"], baseline_ratio
        for cand in spec["candidates"]:
            r = median_ratio_at_n(primary_records, cand, n_max)
            if r is not None and r < best_ratio:
                best_name, best_ratio = cand, r
        winners[family_name] = best_name
        print(f"[family: {family_name}] baseline={spec['baseline']} ({baseline_ratio:.3f}) "
              f"-> winner={best_name} ({best_ratio:.3f}) @ n={n_max}")

    return winners


FINAL_METHOD_FNS = {
    "angular": angular_sort_tour,
    "angular_oropt": angular_sort_oropt_tour,
    "shrinkwrap": shrink_wrap_tour,
    "shrinkwrap_cheapest": shrink_wrap_cheapest_tour,
    "shrinkwrap_2opt": shrink_wrap_2opt_tour,
    "shrinkwrap_cheapest_2opt": shrink_wrap_cheapest_2opt_tour,
    "orbit": orbit_recenter_tour,
    "orbit_every3": _recenter_every(3),
    "orbit_every10": _recenter_every(10),
    "orbit_noncross": orbit_recenter_noncross_tour,
    "orbit_2opt": orbit_recenter_2opt_tour,
    "orbit_clustered": orbit_recenter_clustered_tour,
    "orbit_clustered_v2": orbit_recenter_clustered_v2_tour,
    "shrinkwrap_clustered": shrink_wrap_clustered_tour,
    "nn2opt": nearest_neighbor_2opt_tour,
}


def run_final_comparison(outdir, ns, trials, seed, winners):
    final_methods = {"nn2opt": nearest_neighbor_2opt_tour}
    # baselines for context, plus each family's winner (skip duplicates)
    for name in ["angular", "shrinkwrap", "orbit", "orbit_clustered"]:
        final_methods[name] = FINAL_METHOD_FNS[name]
    for fam, winner in winners.items():
        final_methods[winner] = FINAL_METHOD_FNS[winner]

    raw_order = ["exact", "nn2opt",
                "angular", winners.get("angular"),
                "shrinkwrap", winners.get("shrinkwrap"),
                "orbit", winners.get("orbit"),
                "orbit_clustered", winners.get("clustering")]
    plot_order = list(dict.fromkeys(m for m in raw_order if m is not None))
    plot_order = [m for m in plot_order if m == "exact" or m in final_methods]

    for dataset in ["uniform", "clustered"]:
        gen = DATASET_GENERATORS[dataset]
        print(f"[final comparison: {dataset}] running benchmark...")
        records = run_benchmark(ns=ns, trials=trials, seed=seed,
                                methods=final_methods, point_generator=gen)
        csv_path = save_csv(records, os.path.join(outdir, f"final_comparison_{dataset}.csv"))
        plot_path = plot_benchmark(records, os.path.join(outdir, f"final_comparison_{dataset}.png"),
                                   methods=tuple(plot_order))
        print(f"  saved {csv_path}")
        print(f"  saved {plot_path}")


def main():
    ap = argparse.ArgumentParser(description="Test improvement ideas per family, then compare winners")
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--ns", type=int, nargs="+", default=[6, 8, 10, 12, 15, 20, 30, 50, 75])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    winners = run_families(args.outdir, args.ns, args.trials, args.seed)
    print("Winners:", winners)
    run_final_comparison(args.outdir, args.ns, args.trials, args.seed, winners)


if __name__ == "__main__":
    main()
