"""Batched-SMC particle-scaling figure: particle throughput vs n_particles,
GPU vs CPU, one panel per flavor (smc_lik, smc_abc). Aggregates the per-run
benchmark_*.csv rows written by mpr_jax_numpyro.py for the SMC samplers.

Shows the GPU win: n_particles is the on-device vmap batch, so GPU throughput
(particles/sec) scales up to large batches while CPU plateaus then collapses.

Usage:  python plot_smc_scaling.py [--outdir DIR]
"""
import os, glob, argparse
from datetime import date
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FLAVORS = ["smc_lik", "smc_abc"]


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--outdir", default=os.path.join(here, "..", "..", "results",
                                                     date.today().isoformat()))
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.outdir, "benchmark_*sampler_smc_*.csv"))
    rows = [pd.read_csv(f) for f in files]
    if not rows:
        raise SystemExit(f"no smc benchmark_*.csv in {args.outdir}")
    df = pd.concat(rows, ignore_index=True)
    df["particles_per_sec"] = df["n_particles"] / df["runtime_sec"]

    colors = {"gpu": "#295785", "cpu": "#B5651D"}
    fig, axes = plt.subplots(1, len(FLAVORS), figsize=(6 * len(FLAVORS), 5), squeeze=False)
    for j, flavor in enumerate(FLAVORS):
        ax = axes[0][j]
        sub = df[df["sampler"] == flavor]
        for plat, g in sub.groupby("platform"):
            g = g.sort_values("n_particles")
            ax.plot(g["n_particles"], g["particles_per_sec"], "o-",
                    color=colors.get(plat, "#444"), lw=2, ms=7, label=plat)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("n_particles")
        ax.set_ylabel("particles / sec")
        ax.set_title(flavor)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(frameon=False, title="backend")
    fig.suptitle("Batched-SMC particle throughput: GPU vs CPU")
    fig.tight_layout()

    out = os.path.join(args.outdir, "smc_scaling.png")
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
