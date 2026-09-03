"""Supplementary figure: why CPU forward-model throughput stops improving with batch.

The peak is NOT at a fixed batch width. Measured at three network sizes, the batch at
which throughput peaks scales inversely with the node count, so the product
batch x nodes is invariant at ~640 -- the signature of a per-step working set, not of
core oversubscription (which the 16- vs 64-core control also rules out).

Inputs (all written by the probes in results/<date>/):
  cpu_working_set.csv   sc_size x batch sweep      (nodes_probe.py)
  cpu_cliff_full.csv    10 nodes, all 64 cores     (cpu_cliff_probe.py)
  cpu_cliff_cores16.csv 10 nodes, taskset 16 cores (cpu_cliff_probe.py)

Usage: python plot_cpu_working_set.py --results results/<date> [--paper]
"""
import os, csv, argparse
from collections import defaultdict
import numpy as np
import matplotlib
import matplotlib as mpl
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paper_style

# Categorical slots 1-3, validated for CVD separation against the light surface. The
# aqua sits below 3:1 contrast, so every series is ALSO direct-labelled -- the relief
# the contrast check requires; identity is never carried by colour alone here.
HUES = {5: "#2a78d6", 10: "#eb6834", 20: "#1baf7a"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#d8d7d2", "#fcfcfb"

paper_style.use()

def read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))

def _pow2_ticks(ax, lo, hi, want=4):
    """Explicit powers-of-two ticks. At the authored (on-page) width matplotlib's base-2
    LogLocator thins the axis to a single label, which leaves the batch values unreadable."""
    ks = [k for k in range(0, 20) if lo <= 2 ** k <= hi]
    while len(ks) > want:
        ks = ks[::2]
    ax.set_xticks([2 ** k for k in ks])
    ax.set_xticklabels([rf"$2^{{{k}}}$" for k in ks])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())


def style(ax, xlabel, ylabel, title):
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8, color=INK, fontsize=9.5)
    ax.grid(True, which="major", color=GRID, lw=0.8, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

def label_end(ax, x, y, text, colour):
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(7, 0),
                color=colour, fontsize=8.5, va="center", weight="bold")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--paper", default=None, help="also copy the png into this figures/ dir")
    a = ap.parse_args()

    by_sc = defaultdict(list)
    for r in read(os.path.join(a.results, "cpu_working_set.csv")):
        by_sc[int(r["sc_size"])].append((int(r["batch"]), float(r["evals_per_s"])))
    for v in by_sc.values():
        v.sort()

    # Authored at the text width so \includegraphics does not rescale it; three
    # panels in 6.5in are narrow, so the height is raised to keep them square-ish.
    fig, axes = plt.subplots(1, 3, figsize=paper_style.grid_figsize(1, 3, 1.0, 1.35))

    # (a) raw: the peak MOVES with the node count
    for sc in sorted(by_sc):
        b, y = np.array(by_sc[sc]).T
        axes[0].plot(b, y, "-o", color=HUES[sc], lw=1.8, ms=5,
                     markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3)
        pk = int(b[np.argmax(y)])
        axes[0].plot([pk], [y.max()], "o", ms=11, mfc="none", mec=HUES[sc], mew=1.6, zorder=4)
        label_end(axes[0], b[-1], y[-1], f"{sc} nodes", HUES[sc])
    style(axes[0], "batch width $B$", "throughput (evaluations/s)",
          "(a) Peak moves with network size\n(circled: $B=128,\\ 64,\\ 32$)")

    # (b) rescaled: they collapse onto one curve peaking at batch x nodes ~ 640
    for sc in sorted(by_sc):
        b, y = np.array(by_sc[sc]).T
        axes[1].plot(b * sc, y, "-o", color=HUES[sc], lw=1.8, ms=5,
                     markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3)
        label_end(axes[1], (b * sc)[-1], y[-1], f"{sc} nodes", HUES[sc])
    axes[1].axvline(640, color=INK2, ls="--", lw=1.8, zorder=2)
    axes[1].annotate("$B\\times N \\approx 640$", (640, axes[1].get_ylim()[0]),
                     textcoords="offset points", xytext=(8, 14), color=INK2, fontsize=13)
    style(axes[1], "batch $\\times$ nodes  ($B\\,N$)", "throughput (evaluations/s)",
          "(b) Peaks align at $BN\\approx640$\n(levels differ by network size)")

    # (c) control: core count changes nothing
    for tag, name, col in (("full", "64 cores", "#2a78d6"), ("cores16", "16 cores", "#eb6834")):
        rows = read(os.path.join(a.results, f"cpu_cliff_{tag}.csv"))
        b = np.array([float(r["batch"]) for r in rows])
        y = np.array([float(r["evals_per_s"]) for r in rows])
        axes[2].plot(b, y, "-o", color=col, lw=1.8, ms=5,
                     markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3)
        label_end(axes[2], b[-1], y[-1], name, col)
    style(axes[2], "batch width $B$", "throughput (evaluations/s)",
          "(c) Core count changes nothing\n(10 nodes): not oversubscription")

    for ax in axes:
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi * 2.2)          # room for the direct labels
        _pow2_ticks(ax, lo, hi)

    fig.tight_layout()
    out = os.path.join(a.results, "cpu_working_set.png")
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)  # dpi from paper_style
    print("wrote", out)
    if a.paper:
        import shutil
        shutil.copy(out, os.path.join(a.paper, "cpu_working_set.png"))
        print("copied into", a.paper)

if __name__ == "__main__":
    main()
