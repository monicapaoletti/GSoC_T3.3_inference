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
    return {"batch": d["batch"], "evals_per_s": d["evals_per_s"]}


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
    args = ap.parse_args()

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

    out = os.path.join(args.outdir, "forward_throughput.png")
    fig.savefig(out, dpi=300)          # no bbox_inches='tight' (caused giant canvas)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
