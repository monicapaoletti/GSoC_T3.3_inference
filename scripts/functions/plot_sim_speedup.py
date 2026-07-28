#!/usr/bin/env python3
"""Direct JAX-CPU vs JAX-GPU comparison for the MPR simulation benchmark.

The per-backend figures from benchmarking_simuations.py plot JAX-vs-Numba
SEPARATELY (CPU in one figure, GPU in another), so the CPU-vs-GPU speedup is
never on the same axes and looks absent. This script loads the saved .npz files
(benchmarking_{CPU,GPU}_tend<T>.png.npz -> Ns, times_jax, times_numba) and, for
each t_end, overlays JAX-CPU vs JAX-GPU (and Numba) plus a GPU-speedup panel
(t_CPU / t_GPU; >1 = GPU faster). That is where the batched-simulation GPU win
(if any, at large N) shows up.

Usage:  python plot_sim_speedup.py [--results DIR] [--out DIR]
"""
import os, glob, re, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(results, backend, tend):
    hits = glob.glob(os.path.join(results, "**",
                     f"benchmarking_{backend}_tend{tend}.png.npz"), recursive=True)
    if not hits:
        return None
    d = np.load(hits[0])
    return {"Ns": d["Ns"], "jax": d["times_jax"], "numba": d["times_numba"]}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(here, "..", "..", "results"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or args.results

    # discover the t_end values that have BOTH CPU and GPU npz
    tends = sorted(set(int(re.search(r"_tend(\d+)\.png\.npz", f).group(1))
                       for f in glob.glob(os.path.join(args.results, "**",
                                          "benchmarking_*_tend*.png.npz"), recursive=True)))
    if not tends:
        raise SystemExit(f"no benchmarking_*_tend*.png.npz under {args.results}")

    for tend in tends:
        cpu, gpu = load(args.results, "CPU", tend), load(args.results, "GPU", tend)
        if cpu is None or gpu is None:
            print(f"t_end={tend}: missing {'CPU' if cpu is None else 'GPU'} npz, skipping")
            continue
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        # left: wall-time vs N (log-log), JAX-CPU vs JAX-GPU (+ Numba for context)
        ax1.plot(cpu["Ns"], cpu["jax"], "o-", color="#B5651D", label="JAX (CPU)")
        ax1.plot(gpu["Ns"], gpu["jax"], "o-", color="#295785", label="JAX (GPU)")
        ax1.plot(cpu["Ns"], cpu["numba"], "s--", color="gray", alpha=0.7, label="Numba (CPU)")
        ax1.set_xscale("log"); ax1.set_yscale("log")
        ax1.set_xlabel("number of simulations N"); ax1.set_ylabel("wall-time (s)")
        ax1.set_title(f"MPR simulation, t_end={tend}")
        ax1.grid(True, which="both", ls=":", alpha=0.4); ax1.legend(frameon=False)
        # right: GPU speedup = t_CPU / t_GPU (JAX), >1 means GPU faster
        n = min(len(cpu["Ns"]), len(gpu["Ns"]))
        speedup = np.asarray(cpu["jax"])[:n] / np.asarray(gpu["jax"])[:n]
        ax2.plot(gpu["Ns"][:n], speedup, "o-", color="#295785")
        ax2.axhline(1.0, color="k", ls="--", lw=1, alpha=0.6)
        ax2.set_xscale("log")
        ax2.set_xlabel("number of simulations N")
        ax2.set_ylabel("GPU speedup  (t$_{CPU}$ / t$_{GPU}$)")
        ax2.set_title("GPU vs CPU (JAX);  >1 = GPU faster")
        ax2.grid(True, which="both", ls=":", alpha=0.4)
        fig.tight_layout()
        f = os.path.join(out, f"benchmarking_jax_cpu_vs_gpu_tend{tend}.png")
        fig.savefig(f, dpi=300); plt.close(fig)
        print(f"wrote {f}  (max GPU speedup {speedup.max():.2f}x at N={int(gpu['Ns'][:n][speedup.argmax()])})")


if __name__ == "__main__":
    main()
