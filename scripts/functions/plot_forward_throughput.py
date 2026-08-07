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
import matplotlib.pyplot as plt


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
    """Geometric-mean batch of the first interval where GPU throughput >= CPU."""
    gm = {int(b): t for b, t in zip(gpu["batch"], gpu["evals_per_s"])}
    cm = {int(b): t for b, t in zip(cpu["batch"], cpu["evals_per_s"])}
    common = sorted(set(gm) & set(cm))
    prev = None
    for b in common:
        if gm[b] >= cm[b]:
            return b if prev is None else float(np.sqrt(prev * b))
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

    fig, ax = plt.subplots(figsize=(7, 5))
    if gpu is not None:
        ax.plot(gpu["batch"], gpu["evals_per_s"], "o-", color="#295785",
                lw=2, ms=7, label="GPU (L4)")
    if cpu is not None:
        ax.plot(cpu["batch"], cpu["evals_per_s"], "s-", color="#B5651D",
                lw=2, ms=7, label="CPU (64-core)")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")

    xc = None                       # also reported in the companion table below
    if gpu is not None and cpu is not None:
        xc = crossover(gpu, cpu)
        if xc is not None:
            ax.axvline(xc, color="gray", ls="--", lw=1)
            # place the label in axes-fraction y so it never blows up the canvas
            ax.text(xc, 0.04, " crossover", rotation=90, va="bottom", ha="left",
                    color="gray", fontsize=9,
                    transform=ax.get_xaxis_transform())

    ax.set_xlabel("batch size  (G-values evaluated together via vmap)")
    ax.set_ylabel("throughput  (forward evals / s)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()

    out = os.path.join(out_dir, "forward_throughput.png")
    fig.savefig(out, dpi=300)          # no bbox_inches='tight' (caused giant canvas)
    print(f"wrote {out}")
    _write_table(gpu, cpu, out_dir, xc)


if __name__ == "__main__":
    main()
