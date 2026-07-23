"""Aggregate the per-run benchmark_*.csv files written by mpr_jax_pymc.py into a
single comparison table (one row per sampler x statistic run), in the style of
Baldy et al. 'DCM in Probabilistic Programming Languages'.

Usage:
    python benchmark_compare.py                 # uses today's results folder
    python benchmark_compare.py --dir <folder>  # a specific results folder
    python benchmark_compare.py --dir <folder> --out comparison.csv
"""
import argparse
import glob
import os

import pandas as pd

import utils

# Columns to show in the printed table (in order); all columns are still saved.
DISPLAY_COLS = [
    "sampler", "which_stat", "runtime_sec",
    "max_r_hat", "min_ess_bulk", "min_ess_bulk_per_sec",
    "rmse_param_mean", "rmse_fit_mean", "max_abs_z", "mean_shrinkage",
    "coverage_94hdi", "n_divergences",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=None,
                    help="Results folder with benchmark_*.csv (default: today's results folder).")
    ap.add_argument("--out", default=None,
                    help="Output CSV path (default: <dir>/benchmark_comparison.csv).")
    args = ap.parse_args()

    results_dir = args.dir or utils.results_folder()
    # Run-level files only: exclude the per-parameter benchmark_params_*.csv.
    files = sorted(f for f in glob.glob(os.path.join(results_dir, "benchmark_*.csv"))
                   if "benchmark_params_" not in os.path.basename(f)
                   and os.path.basename(f) != "benchmark_comparison.csv")
    if not files:
        print(f"No benchmark_*.csv found in {results_dir}")
        return

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    sort_cols = [c for c in ("which_stat", "sampler") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    out = args.out or os.path.join(results_dir, "benchmark_comparison.csv")
    df.to_csv(out, index=False)

    show = [c for c in DISPLAY_COLS if c in df.columns]
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", lambda x: f"{x:.4g}"):
        print(f"\n{len(df)} runs from {results_dir}\n")
        print(df[show].to_string(index=False))
    print(f"\nFull comparison table ({df.shape[1]} columns) saved to: {out}")


if __name__ == "__main__":
    main()
