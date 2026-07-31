#!/usr/bin/env python3
"""Running-time figure for the raw MPR simulation: Numba (CPU) vs JAX (GPU).

Companion to the throughput view (plot_sim_batch_throughput.py). Where that plot
shows simulations/second against batch width, this one shows the quantity a user
actually feels -- WALL-CLOCK SECONDS to run N simulations -- so the two curves can be
read directly as "how long would this take me".

Reads the .npz written by benchmarking_simuations.py (Ns, times_jax, times_numba).
The GPU npz carries both the JAX-GPU timings and the reused Numba baseline, so a
single file is enough for the default one-panel figure.

Usage:
  python plot_sim_runtime.py --npz <benchmarking_GPU_tend30000.png.npz> \
      --t_end 30000 --out DIR [DIR ...]
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Chart-surface identity, shared with make_paper_assets. The blue is #1F5FA8 rather
# than the paper's original #295785: that one fails the chroma floor (0.091, reads
# gray). This pair passes all six palette checks -- CVD separation dE 22.6 (protan),
# normal-vision 28.0, contrast >= 3:1 on a light surface.
C_GPU = "#1F5FA8"
C_CPU = "#B5651D"
DPI = 300


def load(npz):
    d = np.load(npz)
    return (np.asarray(d["Ns"], dtype=float),
            np.asarray(d["times_jax"], dtype=float),
            np.asarray(d["times_numba"], dtype=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="benchmarking_GPU_tend*.png.npz")
    ap.add_argument("--t_end", type=int, default=30000)
    ap.add_argument("--out", nargs="+", required=True,
                    help="one or more output directories (results folder AND paper)")
    ap.add_argument("--name", default=None, help="basename; default sim_runtime_tend<T>.png")
    args = ap.parse_args()

    Ns, t_jax, t_numba = load(args.npz)
    name = args.name or f"sim_runtime_tend{args.t_end}.png"

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    # 2px lines, >=8px markers, recessive grid; 2 series -> legend always present.
    ax.plot(Ns, t_numba, "-o", color=C_CPU, lw=2, ms=8, label="Numba (CPU, all cores)",
            zorder=3, markeredgecolor="white", markeredgewidth=1.2)
    ax.plot(Ns, t_jax, "-o", color=C_GPU, lw=2, ms=8, label="JAX (GPU)",
            zorder=4, markeredgecolor="white", markeredgewidth=1.2)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of simulations")
    ax.set_ylabel("Running time (s)")
    # thin-space the thousands separator INSIDE the math only -- replacing on the whole
    # title also hits the comma after "cost" and leaves a literal backslash-comma.
    tend = f"{args.t_end:,}".replace(",", r"\,")
    ax.set_title(f"MPR simulation cost, $t_{{end}}={tend}$")
    ax.grid(True, which="both", ls="--", alpha=0.3, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="upper left")

    # Direct-label the headline gap at the largest N (selective labelling, not every
    # point): this single number is the claim the figure exists to support.
    i = int(np.argmax(Ns))
    if np.isfinite(t_jax[i]) and np.isfinite(t_numba[i]) and t_jax[i] > 0:
        speedup = t_numba[i] / t_jax[i]
        ax.annotate(f"{speedup:.0f}$\\times$ faster",
                    xy=(Ns[i], np.sqrt(t_jax[i] * t_numba[i])),
                    xytext=(-12, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=11, color="#333333")
        ax.annotate("", xy=(Ns[i], t_jax[i]), xytext=(Ns[i], t_numba[i]),
                    arrowprops=dict(arrowstyle="<->", color="#666666", lw=1.2))

    fig.tight_layout()
    for d in args.out:
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, name)
        fig.savefig(p, dpi=DPI)
        print(f"wrote {p} (dpi={DPI})")
    plt.close(fig)


if __name__ == "__main__":
    main()
