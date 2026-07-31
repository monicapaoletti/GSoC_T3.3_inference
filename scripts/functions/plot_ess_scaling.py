#!/usr/bin/env python3
"""ESS/second against on-device batch width, per sampler -- replaces the two weak
throughput figures (smc_scaling.png and throughput_ess.png).

Why this exists. The earlier smc_scaling figure drew broken lines (points present but
unconnected where intermediate cells were missing), carried a single CPU point in one
panel and none in the other while being titled "GPU vs CPU", and plotted
"particles/sec", which mixes work done with time taken. The auto-generated
throughput_ess figure had the opposite failure: every cell across all G, both
features and all samplers dumped at each batch value, so it read as vertical clouds
with no sampler identity.

Design decisions worth stating:
  * LINES ARE GPU ONLY. The GPU sweeps the full batch grid 64..4096, so a line there
    is a measured trend. CPU does not: pymc cells sit at 2-4 chains and JAX-on-CPU
    cells at 64-256 particles -- two DIFFERENT implementations. Joining them would
    manufacture a trend that was never measured, which is exactly the flaw in the
    figure this replaces. They are drawn as markers instead.
  * Median across the four ground-truth G per (sampler, batch): the batching claim is
    about hardware, not about a particular coupling regime.
  * Faceted by feature (FC / FCD) because the two have different feature counts and
    therefore different absolute cost.

Usage:
  python plot_ess_scaling.py --master paper/master_results.csv --out DIR [DIR ...]
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical order (all six checks pass in light mode; the contrast WARN on
# the lighter hues is discharged by the legend + end-of-line direct labels).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SAMPLER_ORDER = ["smc_abc", "smc_lik", "demc", "rwmh", "slice"]
DPI = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--out", nargs="+", required=True)
    ap.add_argument("--name", default="ess_scaling.png")
    args = ap.parse_args()

    df = pd.read_csv(args.master)
    df = df.dropna(subset=["batch", "ess_per_sec", "which_stat", "platform"])
    df = df[df["ess_per_sec"] > 0]

    colors = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(SAMPLER_ORDER)}
    stats = [s for s in ("FC", "FCD") if (df["which_stat"] == s).any()]
    fig, axes = plt.subplots(1, len(stats), figsize=(5.6 * len(stats), 4.8), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, stat in zip(axes, stats):
        d = df[df["which_stat"] == stat]
        gpu = d[d["platform"] == "gpu"]
        for samp in SAMPLER_ORDER:
            g = gpu[gpu["sampler"] == samp]
            if g.empty:
                continue
            # median over the ground-truth G values at each batch width
            m = g.groupby("batch")["ess_per_sec"].median().sort_index()
            if len(m) < 2:
                ax.plot(m.index, m.values, "o", color=colors[samp], ms=8,
                        markeredgecolor="white", markeredgewidth=1.2, zorder=4)
                continue
            ax.plot(m.index, m.values, "-o", color=colors[samp], lw=2, ms=8,
                    markeredgecolor="white", markeredgewidth=1.2, zorder=4,
                    label=samp if ax is axes[0] else None)
            # direct label at the line end (discharges the contrast WARN)
            ax.annotate(samp, xy=(m.index[-1], m.values[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9,
                        color="#333333")

        # CPU cells as markers only -- never joined into a line (see module docstring)
        cpu = d[d["platform"] == "cpu"]
        jax_cpu = cpu[cpu["framework"] != "pymc"]
        pymc_cpu = cpu[cpu["framework"] == "pymc"]
        if not jax_cpu.empty:
            ax.plot(jax_cpu["batch"], jax_cpu["ess_per_sec"], "o", mfc="none",
                    mec="#444444", ms=9, mew=1.6, zorder=5,
                    label="JAX on CPU (same code)" if ax is axes[0] else None)
        if not pymc_cpu.empty:
            ax.plot(pymc_cpu["batch"], pymc_cpu["ess_per_sec"], "x", color="#888888",
                    ms=7, mew=1.5, zorder=5,
                    label="PyMC on CPU (other framework)" if ax is axes[0] else None)

        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("batch width (chains / particles)")
        ax.set_title(stat)
        ax.grid(True, which="major", ls="--", alpha=0.3, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_ylabel("ESS / second")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Sampling throughput vs on-device batch width", y=1.0)
    fig.tight_layout()

    for d_ in args.out:
        os.makedirs(d_, exist_ok=True)
        p = os.path.join(d_, args.name)
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        print(f"wrote {p} (dpi={DPI})")
    plt.close(fig)


if __name__ == "__main__":
    main()
