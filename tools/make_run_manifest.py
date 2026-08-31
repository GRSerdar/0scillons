#!/usr/bin/env python3
"""Write a CSV manifest of every simulation run found under $VSC_DATA/oscillon_runs.

The manifest records the parameters and solver outcome of each run so the set of
completed simulations is known independently of the (large, un-versioned) data
directory. Run this after any batch of jobs finishes and commit the result.

    python tools/make_run_manifest.py [runs_dir] [-o manifest.csv]
"""

import argparse
import csv
import os
import sys

import numpy as np

FIELDS = [
    "run_tag", "lambda_gb", "g2", "selfinteraction", "a_mg", "b_mg", "chi0",
    "coupling", "perturbation", "width", "T", "num_points_t", "r_max",
    "min_dr", "max_dr", "num_grid_points", "wall_time_s", "solver_success",
    "t_crash", "solver_message", "has_solution", "has_diagnostics",
    "has_eft_diagnostics", "size_bytes",
]


def scalar(value):
    """Unwrap the 0-d arrays that np.savez produces for plain Python scalars."""
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else array.tolist()


def collect(run_dir):
    row = {name: "" for name in FIELDS}
    row["run_tag"] = os.path.basename(run_dir)

    meta_path = os.path.join(run_dir, "metadata.npz")
    if os.path.exists(meta_path):
        with np.load(meta_path, allow_pickle=True) as meta:
            for key in meta.files:
                if key in row:
                    row[key] = scalar(meta[key])

    total = 0
    for name in os.listdir(run_dir):
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            total += os.path.getsize(path)
    row["size_bytes"] = total
    row["has_solution"] = os.path.exists(os.path.join(run_dir, "solution.npy"))
    row["has_diagnostics"] = os.path.exists(os.path.join(run_dir, "diagnostics.npz"))
    row["has_eft_diagnostics"] = os.path.exists(
        os.path.join(run_dir, "eft_diagnostics.npz")
    )
    return row


def main():
    default_runs = os.path.join(
        os.environ.get("VSC_DATA", os.path.expanduser("~")), "oscillon_runs"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_dir", nargs="?", default=default_runs)
    parser.add_argument("-o", "--output", default="run_manifest.csv")
    args = parser.parse_args()

    if not os.path.isdir(args.runs_dir):
        sys.exit(f"No such directory: {args.runs_dir}")

    run_dirs = sorted(
        os.path.join(args.runs_dir, name)
        for name in os.listdir(args.runs_dir)
        if os.path.isdir(os.path.join(args.runs_dir, name))
    )

    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for run_dir in run_dirs:
            writer.writerow(collect(run_dir))

    print(f"Wrote {len(run_dirs)} runs to {args.output}")


if __name__ == "__main__":
    main()
