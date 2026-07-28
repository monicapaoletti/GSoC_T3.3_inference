#!/usr/bin/env python3
"""Overlay JAX-CPU vs JAX-GPU throughput from sim_batch_throughput.py CSVs.

For each t_end that has BOTH backends, plot throughput (sims/s) vs batch size on
a log-log axis. The GPU curve staying high/flat while the CPU curve peaks and
collapses is the visible "GPU wins when you batch" evidence the per-backend
sim figures don't show. Also prints the max GPU/CPU throughput ratio.

Usage:  python plot_sim_batch_throughput.py [--results DIR] [--out DIR]
"""
import os, glob, re, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(results, backend, tend):
    hits = glob.glob(os.path.join(results, "**",
                     f"sim_batch_throughput_{backend}_tend{tend}.csv"), recursive=True)
    if not hits:
        return None
    d = np.genfromtxt(hits[0], delimiter=",", names=True)
    d = np.atleast_1d(d)
    return {"batch": d["batch"], "thr": d["throughput_sims_per_s"]}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(here, "..", "..", "results"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or args.results

    tends = sorted(set(int(re.search(r"_tend(\d+)\.csv", f).group(1))
                   for f in glob.glob(os.path.join(args.results, "**",
                   "sim_batch_throughput_*_tend*.csv"), recursive=True)))
    if not tends:
        raise SystemExit(f"no sim_batch_throughput_*_tend*.csv under {args.results}")

    for tend in tends:
        cpu, gpu = load(args.results, "CPU", tend), load(args.results, "GPU", tend)
        if cpu is None or gpu is None:
            print(f"t_end={tend}: missing {'CPU' if cpu is None else 'GPU'} csv, skipping")
            continue
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.plot(cpu["batch"], cpu["thr"], "o-", color="#B5651D", label="JAX (CPU, 64 cores)")
        ax.plot(gpu["batch"], gpu["thr"], "o-", color="#295785", label="JAX (GPU, L4)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("batch size  (simulations run in parallel)")
        ax.set_ylabel("throughput  (simulations / s)")
        ax.set_title(f"MPR simulation throughput vs batch, t_end={tend}")
        ax.grid(True, which="both", ls=":", alpha=0.4); ax.legend(frameon=False)
        fig.tight_layout()
        f = os.path.join(out, f"sim_batch_throughput_tend{tend}.png")
        fig.savefig(f, dpi=300); plt.close(fig)
        # ratio at the largest batch both backends reached
        n = min(len(cpu["batch"]), len(gpu["batch"]))
        ratio = np.asarray(gpu["thr"])[:n] / np.asarray(cpu["thr"])[:n]
        i = int(ratio.argmax())
        print(f"wrote {f}  (max GPU/CPU throughput {ratio.max():.1f}x at batch={int(gpu['batch'][:n][i])}; "
              f"GPU peak {np.max(gpu['thr']):.0f} sims/s, CPU peak {np.max(cpu['thr']):.0f} sims/s)")


if __name__ == "__main__":
    main()
