"""Overlay GPU vs CPU forward-eval throughput from microbench_batch.py CSVs.

Reads results/<DATE>/forward_throughput_{gpu,cpu}.csv, plots throughput (evals/s)
vs batch size on log-log axes, marks the GPU>CPU crossover. Shows the core
finding: GPU wall-time is flat (throughput ~linear in batch) while CPU plateaus
then collapses.

Usage:  python plot_forward_throughput.py [--outdir DIR]
"""
import os, argparse
from datetime import date
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
import paper_style


def load(outdir, platform):
    path = os.path.join(outdir, f"forward_throughput_{platform}.csv")
    if not os.path.exists(path):
        return None
    d = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    # time_s is carried for the companion table, not the plot. It is the column that
    # makes the table worth printing: GPU wall time is ~flat from batch 1 to 4096, i.e.
    # 4096 simulations cost what one does. A throughput plot shows a rising line and
    # cannot say that the TIME never moved.
    return {"batch": d["batch"], "evals_per_s": d["evals_per_s"], "time_s": d["time_s"]}


def _write_table(gpu, cpu, out, xover):
    r"""Companion table for the throughput figure, from the same CSVs the figure reads."""
    import csv as _csv
    bs = sorted({int(b) for src in (gpu, cpu) if src for b in src["batch"]})

    def pick(src, b, key):
        if not src:
            return None
        for bb, v in zip(src["batch"], src[key]):
            if int(bb) == b:
                return float(v)
        return None

    rows = []
    for b in bs:
        gt, ge = pick(gpu, b, "time_s"), pick(gpu, b, "evals_per_s")
        ct, ce = pick(cpu, b, "time_s"), pick(cpu, b, "evals_per_s")
        rows.append({"batch": b, "gpu_time_s": gt, "gpu_evals_per_s": ge,
                     "cpu_time_s": ct, "cpu_evals_per_s": ce,
                     "speedup": (ge / ce) if (ge and ce) else None})

    csv_path = os.path.join(out, "forward_throughput_table.csv")
    with open(csv_path, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {csv_path}")

    def f(x, nd=3):
        return "--" if x is None else f"{x:.{nd}g}"

    lines = [
        r"\begin{table}[t]\centering",
        r"\caption{Forward-model cost against batch width (10-node model with FC "
        r"feature extraction), the numbers behind Fig.~\ref{fig:throughput}. Batch is "
        r"the number of $G$ values evaluated together under \texttt{vmap}. Note the GPU "
        r"\emph{wall time}, not just its throughput: it is essentially constant from "
        r"batch 1 to 4096, so 4096 simulations cost what a single one does -- the "
        r"batching argument in its strongest form, and the one thing a throughput plot "
        r"cannot show. The CPU instead degrades beyond batch 64. ``--'' = not measured.}",
        r"\label{tab:throughput}\small",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"& \multicolumn{2}{c}{GPU (L4)} & \multicolumn{2}{c}{CPU (64-core)} & \\",
        r"batch & time (s) & evals/s & time (s) & evals/s & GPU/CPU \\",
        r"\midrule",
    ]
    for r in rows:
        sp = "--" if r["speedup"] is None else f"{r['speedup']:.3g}$\\times$"
        lines.append(f"{r['batch']} & {f(r['gpu_time_s'])} & {f(r['gpu_evals_per_s'])} & "
                     f"{f(r['cpu_time_s'])} & {f(r['cpu_evals_per_s'])} & {sp} \\\\")
    lines += [r"\bottomrule"]
    if xover:
        lines += [r"\\[2pt]", rf"\multicolumn{{6}}{{l}}{{\footnotesize Crossover "
                  rf"(GPU overtakes CPU): batch $\approx{xover:.0f}$.}}\\"]
    lines += [r"\end{tabular}", r"\end{table}"]
    tex_path = os.path.join(out, "forward_throughput_table.tex")
    with open(tex_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {tex_path}")


def crossover(gpu, cpu):
    """Batch at which GPU throughput overtakes CPU, by log-log interpolation.

    The geometric mean of the bracketing batches used to be returned here, which
    ignores HOW CLOSE the curves are at each end: on the measured sweep the CPU leads
    by only 7% at B=16 but the GPU leads by 2.1x at B=64, so the crossing sits near 18
    while the geometric mean reports 32 -- most of an octave to the right of where the
    lines visibly meet. Interpolating log2(GPU/CPU) against log2(B) puts the marker
    where the curves actually cross.
    """
    gm = {int(b): t for b, t in zip(gpu["batch"], gpu["evals_per_s"])}
    cm = {int(b): t for b, t in zip(cpu["batch"], cpu["evals_per_s"])}
    common = sorted(set(gm) & set(cm))
    prev = None
    for b in common:
        if gm[b] >= cm[b]:
            if prev is None:
                return float(b)
            lr0, lr1 = np.log2(gm[prev] / cm[prev]), np.log2(gm[b] / cm[b])
            if lr1 == lr0:
                return float(np.sqrt(prev * b))
            f = -lr0 / (lr1 - lr0)                       # fraction of the octave to travel
            return float(2.0 ** (np.log2(prev) + f * (np.log2(b) - np.log2(prev))))
        prev = b
    return None


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--outdir", default=os.path.join(here, "..", "..", "results",
                                                     date.today().isoformat()))
    # Inputs and outputs are separable: the measurement CSVs live in the dated folder of
    # the day they were MEASURED, while regenerated assets belong in today's. Without
    # this, re-rendering an old measurement silently rewrites that old folder.
    ap.add_argument("--out", default=None,
                    help="where to write figure+table (default: --outdir)")
    args = ap.parse_args()
    out_dir = args.out or args.outdir
    os.makedirs(out_dir, exist_ok=True)

    gpu = load(args.outdir, "gpu")
    cpu = load(args.outdir, "cpu")
    if gpu is None and cpu is None:
        raise SystemExit(f"no forward_throughput_*.csv in {args.outdir}")

    # Authored at the size it is INCLUDED at (0.8\linewidth), so nothing is rescaled by
    # \includegraphics and the point sizes below are the point sizes on the page.
    paper_style.use()
    fig, ax = plt.subplots(figsize=paper_style.figsize(0.8, aspect=0.70))
    if gpu is not None:
        ax.plot(gpu["batch"], gpu["evals_per_s"], "o-", color="#295785",
                lw=1.8, ms=5.5, label="GPU (L4)")
    if cpu is not None:
        ax.plot(cpu["batch"], cpu["evals_per_s"], "s-", color="#B5651D",
                lw=1.8, ms=5.5, label="CPU (64-core)")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")

    # Tick where we actually measured. The sweep samples EVEN powers of two
    # (1, 4, 16, ... = 2^0, 2^2, ...), while matplotlib's default base-2 LogLocator
    # thins 2^0..2^14 with a stride of 2 starting at 2^1 -- the odd powers. The two
    # interleave exactly, so every marker landed midway between two labels and the
    # plot read as if the points were off the grid.
    batches = sorted({int(b) for src in (gpu, cpu) if src is not None
                      for b in src["batch"]})
    # TICK the exact powers of two only. Every measured batch is still plotted; the
    # sweep now includes non-powers (96, 192) to resolve the CPU plateau, and labelling
    # those crowds them into their neighbours -- 96 collided with 2^7, 192 with 2^8.
    powers = [b for b in batches if float(np.log2(b)).is_integer()]
    ax.set_xticks(powers)
    ax.set_xticklabels([rf"$2^{{{int(np.log2(b))}}}$" for b in powers])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    xc = None                       # also reported in the companion table below
    if gpu is not None and cpu is not None:
        xc = crossover(gpu, cpu)
        if xc is not None:
            ax.axvline(xc, color="gray", ls="--", lw=1)
            # place the label in axes-fraction y so it never blows up the canvas
            ax.text(xc, 0.04, " crossover", rotation=90, va="bottom", ha="left",
                    color="gray", fontsize=9,
                    transform=ax.get_xaxis_transform())

    ax.set_xlabel("batch size $B$")
    ax.set_ylabel("throughput (forward evals/s)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()

    out = os.path.join(out_dir, "forward_throughput.png")
    fig.savefig(out, dpi=300)          # no bbox_inches='tight' (caused giant canvas)
    print(f"wrote {out}")
    _write_table(gpu, cpu, out_dir, xc)


if __name__ == "__main__":
    main()
