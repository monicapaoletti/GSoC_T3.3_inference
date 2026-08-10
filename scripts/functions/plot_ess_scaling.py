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
  * ONE ground-truth G by default (--G, 0.2), so every widest-batch point equals its
    cell in the benchmark table exactly. --pool_G instead keeps all four couplings and
    draws each point as the median with a min-max error bar, covering the per-coupling
    tables at once at the cost of that exact correspondence.
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
#
# IMPORTED, NOT RESTATED. This file used to carry its own six-entry copy of the palette
# and order while make_paper_assets had seven, so Fig. 4 silently dropped smc_abc_demc
# (32 cells, up to 5.7 ESS/s at batch 4096) and smc_abc_eps10 -- both of which Figs. 5-7
# show. Two copies of a "shared" convention is how they diverge; there is now one.
try:
    from make_paper_assets import (PALETTE, SAMPLER_ORDER, FADED, _HUE_ALIAS,
                                   _series_label)
except Exception:                                    # standalone fallback
    PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
               "#7d4bc4"]
    SAMPLER_ORDER = ["smc_abc", "smc_lik", "demc", "rwmh", "slice", "demcz",
                     "smc_abc_demc"]
    FADED, _HUE_ALIAS = set(), {}
    _series_label = lambda s, fw="": s
DPI = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--out", nargs="+", required=True)
    ap.add_argument("--name", default="ess_scaling.png")
    ap.add_argument("--G", type=float, default=0.2,
                    help="ground-truth coupling to display; must match --table_G in "
                         "make_paper_assets so figure and tables show the same cells")
    ap.add_argument("--pool_G", action="store_true",
                    help="pool ALL ground-truth couplings instead of showing one: each "
                         "point becomes the median over them with a min-max error bar. "
                         "Covers every per-G table at once, but then no point equals a "
                         "single table cell (--G does).")
    args = ap.parse_args()

    df = pd.read_csv(args.master)
    df = df.dropna(subset=["batch", "ess_per_sec", "which_stat", "platform"])
    df = df[df["ess_per_sec"] > 0]

    # --- select exactly what the benchmark TABLES select -------------------------------
    # Previously the two arms of this figure were summarised differently: GPU points were
    # a MEDIAN over the four ground-truth couplings, while CPU points were every run drawn
    # raw. A dot and a cross were therefore not comparable (one was the median of four,
    # the other one of four), and neither equalled the table cell, which reports a single
    # coupling. Three summaries of the same runs.
    #
    # Worse, no de-duplication was applied, so a cell re-run at several BUDGETS had its
    # short exploratory passes medianed with the budget-matched one: slice/FCD at batch
    # 4096 mixed n_draws of 30, 100 and 1000 and came out 12.8x more efficient than the
    # table says -- flattering exactly the sampler the text singles out as impractical.
    #
    # One rule for both arms now: the benchmark configuration, one coupling, latest run
    # per cell. Every point is a single run and the widest-batch point equals its table
    # cell exactly.
    if "SC_size" in df:
        df = df[df["SC_size"].astype(float).fillna(10) == 10]
    if "t_end" in df:
        df = df[df["t_end"].astype(float).fillna(30000) == 30000]
    #
    # --pool_G keeps all four couplings instead, and reports each cell as the MEDIAN over
    # them with a bar spanning the full min-max range. Same runs either way -- the only
    # question is whether G* is shown as a spread or held fixed. Pooling covers all four
    # supplementary tables at once but no point then equals a single table cell; the
    # single-coupling default does equal Table 4a exactly. Both are generated.
    if "G_true" in df and not args.pool_G:
        df = df[np.isclose(df["G_true"].astype(float), args.G)]
    # `framework` in the dedup key: JAX-on-CPU and PyMC-on-CPU are different
    # implementations and one must not evict the other (same fix as make_paper_assets).
    cell = [c for c in ("sampler", "framework", "platform", "which_stat", "batch")
            if c in df]
    if args.pool_G and "G_true" in df:
        cell = cell + ["G_true"]
    by = [c for c in ("n_draws", "n_warmup", "_mtime") if c in df]
    if by:
        df = (df.sort_values(by, na_position="first")
                .groupby(cell, dropna=False).tail(1))

    colors = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(SAMPLER_ORDER)}
    # Aliased variants share their parent's hue (smc_abc_eps10 -> smc_abc): same
    # algorithm, different setting, told apart by the faded/dashed flag and the legend.
    colors.update({k: colors[v] for k, v in _HUE_ALIAS.items() if v in colors})
    # SAMPLER_ORDER first for a fixed hue order, then any variant actually in the data --
    # iterating SAMPLER_ORDER alone is what dropped smc_abc_eps10 from this figure.
    present = set(df["sampler"].dropna().unique())
    order = [s for s in SAMPLER_ORDER if s in present] + \
            [s for s in sorted(present - set(SAMPLER_ORDER)) if s in colors]
    stats = [s for s in ("FC", "FCD") if (df["which_stat"] == s).any()]
    fig, axes = plt.subplots(1, len(stats), figsize=(5.6 * len(stats), 4.8), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, stat in zip(axes, stats):
        d = df[df["which_stat"] == stat]
        gpu = d[d["platform"] == "gpu"]
        ends = []
        for samp in order:
            g = gpu[gpu["sampler"] == samp]
            if g.empty:
                continue
            # One run per batch after the selection above, so this is a no-op guard
            # rather than an aggregation; it would only fire on a duplicate file. (It
            # used to be a median over the four ground-truth couplings, which is what
            # made this figure disagree with the tables.)
            m = g.groupby("batch")["ess_per_sec"].median().sort_index()
            # Spread over the couplings, drawn as a bar in --pool_G. Empty otherwise,
            # where each batch holds exactly one run and lo == hi == m.
            lo = g.groupby("batch")["ess_per_sec"].min().sort_index()
            hi = g.groupby("batch")["ess_per_sec"].max().sort_index()

            # A scaling figure must vary ONE thing. Most samplers hold their budget fixed
            # across the sweep -- demc and rwmh at 1000/1000, the SMC family at 50 stages
            # x 5 moves -- but slice was swept at 50/100 and only re-run budget-matched at
            # the widest batch. Joined naively its line FALLS between the last two points,
            # reading as "widening the batch made it less efficient", the opposite of the
            # truth: at matched budget 1024->4096 takes it from 0.140 to 0.616 ESS/s, and
            # the drop is the 12x longer runtime of the 1000/1000 re-run.
            #
            # So: the budget of the WIDEST batch is the reference (it is also the cell the
            # tables report and the one matched to demc/rwmh); points at any other budget
            # are faded, dashed, and never joined to it, so no segment spans a budget
            # change. n_warmup identifies the budget -- it is constant within every
            # sampler except slice.
            gb = g.dropna(subset=["n_warmup"])
            off = set()
            if len(gb):
                ref = float(gb.sort_values("batch")["n_warmup"].iloc[-1])
                off = {float(b) for b, w in zip(gb["batch"], gb["n_warmup"])
                       if float(w) != ref}
            m_ref = m[[b for b in m.index if float(b) not in off]]
            m_off = m[[b for b in m.index if float(b) in off]]

            if args.pool_G:
                # Bars first, under the markers. Asymmetric: [med-min, max-med].
                yerr = np.vstack([(m - lo).values, (hi - m).values])
                ax.errorbar(m.index, m.values, yerr=yerr, fmt="none",
                            ecolor=colors[samp], elinewidth=1.2, capsize=3,
                            alpha=0.55, zorder=2)
            if len(m) < 2:
                ax.plot(m.index, m.values, "o", color=colors[samp], ms=8,
                        markeredgecolor="white", markeredgewidth=1.2, zorder=4)
                continue
            if len(m_off):
                ax.plot(m_off.index, m_off.values,
                        "--o" if len(m_off) > 1 else "o",
                        color=colors[samp], lw=1.6, ms=7, alpha=0.40,
                        markeredgecolor="white", markeredgewidth=1.0, zorder=3)
            if len(m_ref):
                ax.plot(m_ref.index, m_ref.values,
                        "-o" if len(m_ref) > 1 else "o",
                        color=colors[samp], lw=2, ms=8,
                        markeredgecolor="white", markeredgewidth=1.2, zorder=4)
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
        for samp in order:
            c = colors[samp]
            # Faded for the "shown for contrast" variants, as in Figs. 5-7: smc_abc_eps10
            # shares smc_abc's hue, so without this the uncalibrated run would read as an
            # ordinary ABC measurement.
            af = 0.40 if samp in FADED else 1.0
            jc = cpu[(cpu["sampler"] == samp) & (cpu["framework"] != "pymc")]
            pc = cpu[(cpu["sampler"] == samp) & (cpu["framework"] == "pymc")]
            for src, kw in ((jc, dict(marker="o", mfc="none", mec=c, ms=10, mew=2.0,
                                      alpha=af, zorder=6)),
                            (pc, dict(marker="x", color=c, ms=7, mew=1.8,
                                      alpha=0.9 * af, zorder=5))):
                if src.empty:
                    continue
                if args.pool_G:
                    # Same median-and-range treatment as the GPU lines, or the two arms
                    # would again be summarised differently -- a marker meaning "median
                    # of four" next to one meaning "one of four" is the exact defect the
                    # selection block above was written to remove.
                    a = src.groupby("batch")["ess_per_sec"].agg(["median", "min", "max"])
                    ax.errorbar(a.index, a["median"],
                                yerr=np.vstack([(a["median"] - a["min"]).values,
                                                (a["max"] - a["median"]).values]),
                                fmt="none", ecolor=c, elinewidth=1.2, capsize=3,
                                alpha=0.55, zorder=4)
                    ax.plot(a.index, a["median"], linestyle="none", **kw)
                else:
                    ax.plot(src["batch"], src["ess_per_sec"], linestyle="none", **kw)

        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("batch width (chains / particles)")
        nG = df["G_true"].dropna().nunique() if "G_true" in df else 1
        ax.set_title(stat if not args.pool_G
                     else f"{stat}   (median over {nG} $G^\\star$, bars = range)",
                     fontsize=11 if not args.pool_G else 10)
        ax.grid(True, which="major", ls="--", alpha=0.3, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_ylabel("ESS / second")
    # Two legends: hue = algorithm, marker = where it ran. Keeping them separate stops
    # the reader having to decode a 12-entry cross-product.
    from matplotlib.lines import Line2D
    alg = [Line2D([], [], color=colors[s], lw=2, marker="o", ms=7,
                  markeredgecolor="white", alpha=0.45 if s in FADED else 1.0,
                  linestyle="--" if s in FADED else "-", label=_series_label(s))
           for s in order]
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
    fig.tight_layout()

    for d_ in args.out:
        os.makedirs(d_, exist_ok=True)
        p = os.path.join(d_, args.name)
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        print(f"wrote {p} (dpi={DPI})")
    plt.close(fig)


if __name__ == "__main__":
    main()
