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
#
# COLOR ENCODES THE ALGORITHM, MARKER ENCODES WHERE IT RAN. That way PyMC's Metropolis
# sits in the same hue as the GPU rwmh line and can be read against it directly --
# colour follows the entity, not the platform. demcz gets its own slot because PyMC has
# it and the JAX side deliberately does not (see mcmc_jax docstring).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SAMPLER_ORDER = ["smc_abc", "smc_lik", "demc", "rwmh", "slice", "demcz"]
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
        ends = []
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
                    )
            ends.append((float(m.index[-1]), float(m.values[-1]), samp))

        # Direct labels at the line ends (these discharge the contrast WARN), pushed
        # apart where curves nearly coincide -- smc_lik and smc_abc overlap almost
        # exactly, so un-nudged labels print on top of one another.
        ends.sort(key=lambda t: t[1])
        min_gap = 0.075                      # in log10 units of the y axis
        placed = []
        for x_e, y_e, samp in ends:
            y_lab = np.log10(y_e)
            if placed and y_lab - placed[-1] < min_gap:
                y_lab = placed[-1] + min_gap
            placed.append(y_lab)
            ax.annotate(samp, xy=(x_e, 10 ** y_lab), xytext=(7, 0),
                        textcoords="offset points", va="center", fontsize=9,
                        color=colors[samp])

        # CPU cells as markers only -- never joined into a line (see module docstring).
        # Same hue as the algorithm's GPU line, so like can be read against like.
        cpu = d[d["platform"] == "cpu"]
        for samp in SAMPLER_ORDER:
            c = colors[samp]
            jc = cpu[(cpu["sampler"] == samp) & (cpu["framework"] != "pymc")]
            pc = cpu[(cpu["sampler"] == samp) & (cpu["framework"] == "pymc")]
            if not jc.empty:
                ax.plot(jc["batch"], jc["ess_per_sec"], "o", mfc="none", mec=c,
                        ms=10, mew=2.0, zorder=6)
            if not pc.empty:
                ax.plot(pc["batch"], pc["ess_per_sec"], "x", color=c,
                        ms=7, mew=1.8, alpha=0.9, zorder=5)

        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("batch width (chains / particles)")
        ax.set_title(stat)
        ax.grid(True, which="major", ls="--", alpha=0.3, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_ylabel("ESS / second")
    # Two legends: hue = algorithm, marker = where it ran. Keeping them separate stops
    # the reader having to decode a 12-entry cross-product.
    from matplotlib.lines import Line2D
    alg = [Line2D([], [], color=colors[s], lw=2, marker="o", ms=7,
                  markeredgecolor="white", label=s)
           for s in SAMPLER_ORDER if (df["sampler"] == s).any()]
    plat = [
        Line2D([], [], color="#555555", lw=2, marker="o", ms=7,
               markeredgecolor="white", label="JAX, GPU"),
        Line2D([], [], color="#555555", lw=0, marker="o", mfc="none", ms=9, mew=2,
               label="JAX, CPU (same code)"),
        Line2D([], [], color="#555555", lw=0, marker="x", ms=7, mew=1.8,
               label="PyMC, CPU (other framework)"),
    ]
    l1 = axes[0].legend(handles=alg, frameon=False, fontsize=9, loc="upper left",
                        title="algorithm", alignment="left")
    l1.get_title().set_fontsize(9)
    axes[0].add_artist(l1)
    axes[-1].legend(handles=plat, frameon=False, fontsize=9, loc="lower right",
                    title="backend", alignment="left")
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
