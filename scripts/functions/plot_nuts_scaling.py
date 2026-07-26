"""Vectorized-NUTS chain-scaling figure: ESS/sec and runtime vs n_chains,
GPU (vectorized) vs CPU (parallel). Aggregates the per-run benchmark_*.csv
rows written by mpr_jax_numpyro.py for sampler==nuts.

Shows the GPU win: vectorized chains are ~free on-device, so ESS/sec rises with
n_chains at ~flat runtime, while CPU processes contend and flatten/degrade.

Usage:  python plot_nuts_scaling.py [--outdir DIR]
"""
import os, glob, argparse
from datetime import date
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--outdir", default=os.path.join(here, "..", "..", "results",
                                                     date.today().isoformat()))
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.outdir, "benchmark_*sampler_nuts_*.csv"))
    rows = [pd.read_csv(f) for f in files]
    if not rows:
        raise SystemExit(f"no nuts benchmark_*.csv in {args.outdir}")
    df = pd.concat(rows, ignore_index=True)
    df = df[df["sampler"] == "nuts"].sort_values("n_chains")

    colors = {"gpu": "#295785", "cpu": "#B5651D"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for plat, g in df.groupby("platform"):
        g = g.sort_values("n_chains")
        c = colors.get(plat, "#444")
        axes[0].plot(g["n_chains"], g["min_ess_bulk_per_sec"], "o-", color=c,
                     lw=2, ms=7, label=f"{plat}")
        axes[1].plot(g["n_chains"], g["runtime_sec"], "o-", color=c,
                     lw=2, ms=7, label=f"{plat}")

    axes[0].set_ylabel("ESS(bulk) / sec")
    axes[0].set_title("Throughput vs chains (higher = better)")
    axes[1].set_ylabel("runtime (s)")
    axes[1].set_title("Runtime vs chains")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("n_chains")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(frameon=False, title="backend")
    fig.suptitle("Vectorized-NUTS chain scaling: GPU (vectorized) vs CPU (parallel)")
    fig.tight_layout()

    out = os.path.join(args.outdir, "nuts_scaling.png")
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
