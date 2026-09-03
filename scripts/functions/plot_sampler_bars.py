"""The benchmark comparison as bar charts, one panel per metric.

Meysam's item 3, following the last cell of ins-amu/DCM_PPLs
(PyMC/DCM_ERP_PyMC_multiple_ways.ipynb): a 2x2 grid of bars with the SAMPLERS on the x
axis and one metric per panel, each bar annotated with its value. That notebook shows
bulk-ESS, computational time, RMSE fit and RMSE params; the equivalents here are
ESS/s, wall-clock, recovery error and posterior width.

The table stays -- it is the reference a reader looks values up in. This is the ranking
at a glance, which a 13-column grid does not give.

Usage: python plot_sampler_bars.py --master paper/master_results.csv --G 0.5 \
                                   --which_stat FCD --out results/<date>
"""
import os, csv, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paper_style

GPU, CPU = "#2a78d6", "#eb6834"
# (column, panel title, log scale?, better direction)
METRICS = [("ess_per_sec", "ESS / second",          True,  "higher is better"),
           ("runtime_min", "wall-clock (min)",      True,  "lower is better"),
           ("abs_err",     r"recovery error $|\Delta G|$", True, "lower is better"),
           ("G_sd",        "posterior sd",          True,  None)]


def num(x):
    try:
        v = float(x)
        return v if v == v else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--G", default="0.5")
    ap.add_argument("--which_stat", default="FCD")
    ap.add_argument("--out", required=True)
    ap.add_argument("--paper", default=None)
    ap.add_argument("--name", default="sampler_bars.png")
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.master))
            if r["G_true"] == a.G and r["which_stat"] == a.which_stat]
    # One cell per sampler and platform: the widest batch it was run at, which is the
    # configuration the master table reports.
    best = {}
    for r in rows:
        k = (r["sampler"], r["platform"])
        b = num(r["batch"]) or 0
        if k not in best or b > (num(best[k]["batch"]) or 0):
            best[k] = r
    items = sorted(best.items(), key=lambda kv: (kv[0][1], kv[0][0]))
    labels = [s for (s, _p), _ in items]
    plats = [p for (_, p), _ in items]

    paper_style.use()
    fig, axes = plt.subplots(2, 2, figsize=paper_style.grid_figsize(2, 2, 1.0, 0.86))
    for ax, (col, title, logy, hint) in zip(axes.ravel(), METRICS):
        vals = []
        for (_, _), r in items:
            v = (num(r["runtime_sec"]) / 60.0 if col == "runtime_min"
                 else num(r.get(col)))
            vals.append(v)
        x = np.arange(len(items))
        shown = [v if (v is not None and v > 0) else np.nan for v in vals]
        cols = [GPU if p == "gpu" else CPU for p in plats]
        ax.bar(x, shown, width=0.72, color=cols, edgecolor="white", linewidth=0.7, zorder=3)
        if logy:
            ax.set_yscale("log")
        # value on every bar, as in the reference notebook
        for xi, v in zip(x, shown):
            if not np.isfinite(v):
                ax.annotate("n/a", (xi, ax.get_ylim()[0]), textcoords="offset points",
                            xytext=(0, 2), ha="center", fontsize=7, color=paper_style.INK2)
                continue
            ax.annotate(f"{v:.3g}", (xi, v), textcoords="offset points", xytext=(0, 2),
                        ha="center", fontsize=7, rotation=90, color=paper_style.INK2)
        ax.set_title(title + (f"\n({hint})" if hint else ""), pad=6, fontsize=9.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.grid(True, axis="y", zorder=0); ax.grid(False, axis="x")
        ax.margins(y=0.30)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 6)   # room for the rotated value labels

    handles = [plt.Rectangle((0, 0), 1, 1, color=GPU), plt.Rectangle((0, 0), 1, 1, color=CPU)]
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(handles, ["GPU", "CPU"], frameon=False, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), fontsize=9.5)

    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, a.name)
    fig.savefig(out); plt.close(fig)
    print("wrote", out)
    if a.paper:
        import shutil; shutil.copy(out, a.paper)


if __name__ == "__main__":
    main()
