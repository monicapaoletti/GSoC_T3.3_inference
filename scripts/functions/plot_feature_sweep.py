"""The four biologically interpretable features against coupling, as bars.

Meysam asked for two things that turn out to be the same figure: some tables shown as
bar plots, and a sweep over G reporting mean/sd of FC and FCD. This is the content of
Table tab:dynamics:G drawn rather than tabulated -- the table stays for the exact values,
the figure carries the shape of the dependence, which is what a reader actually takes
from it.

Reads the CSV that plot_model_dynamics.py writes beside its figure, so the two can never
disagree.

Usage: python plot_feature_sweep.py --stats results/<date>/model_dynamics_sweepG_tend300000_stats.csv \
                                    --out results/<date> [--paper paper/figures]
"""
import os, csv, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paper_style

# Two features, two moments. Mean and sd share a row so the eye compares like with like;
# FC and FCD share a column so the feature comparison Meysam asked about is the vertical
# one. Colour separates the feature, not the moment.
# Columns are FC / FCD / the underlying dynamics; rows are the first and second moment.
# The two dynamics panels are here so the figure can REPLACE tab:dynamics:G rather than
# duplicate it -- the text cites the firing rate, which a four-feature figure would drop.
PANELS = [("fc_mean",  "FC mean",           "#2a78d6"),
          ("fcd_mean", "FCD mean",          "#eb6834"),
          ("r_mean",   r"firing rate $\bar r$", "#1baf7a"),
          ("fc_sd",    "FC sd",             "#2a78d6"),
          ("fcd_sd",   "FCD sd",            "#eb6834"),
          ("bold_sd",  "BOLD sd",           "#1baf7a")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--paper", default=None)
    ap.add_argument("--name", default="feature_sweep.png")
    a = ap.parse_args()

    rows = sorted(csv.DictReader(open(a.stats)), key=lambda r: float(r["G"]))
    G = [float(r["G"]) for r in rows]
    x = np.arange(len(G))

    paper_style.use()
    fig, axes = plt.subplots(2, 3, figsize=paper_style.grid_figsize(2, 3, 1.0, 0.80),
                             sharex=True)
    for ax, (key, label, colour) in zip(axes.ravel(), PANELS):
        y = np.array([float(r[key]) for r in rows])
        bars = ax.bar(x, y, width=0.62, color=colour, edgecolor="white", linewidth=0.8,
                      zorder=3)
        # The working point is the claim the figure exists to make, so mark it rather
        # than leaving the reader to find the tallest bar.
        pk = int(np.argmax(y))
        bars[pk].set_edgecolor(paper_style.INK)
        bars[pk].set_linewidth(1.4)
        # Value labels only on the peak and the smallest: a number on every bar is noise,
        # and the table carries the full set anyway.
        for i in (pk, int(np.argmin(y))):
            ax.annotate(f"{y[i]:.3g}", (x[i], y[i]), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=8.5,
                        color=paper_style.INK2)
        ax.set_title(label, pad=6)
        ax.set_ylim(0, y.max() * 1.28)
        ax.grid(True, axis="y", zorder=0)
        ax.grid(False, axis="x")
    for ax in axes[1]:
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g:g}" for g in G])
        ax.set_xlabel(r"coupling $G^\star$")

    fig.tight_layout()
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, a.name)
    fig.savefig(out); plt.close(fig)
    print("wrote", out)
    if a.paper:
        import shutil; shutil.copy(out, os.path.join(a.paper, a.name))
        print("copied into", a.paper)


if __name__ == "__main__":
    main()
