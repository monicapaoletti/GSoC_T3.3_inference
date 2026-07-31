#!/usr/bin/env python3
"""Model-dynamics figure (paper point 7 / extended-abstract Fig. 2A).

For each ground-truth coupling G, simulate the MPR whole-brain model and show, on one
row: (i) example regional firing-rate r(t) and BOLD traces, (ii) the static FC matrix,
(iii) the FCD matrix. This grounds the reader in the dynamical regimes before the
inference results, and reproduces the switching behaviour (bistability -> FCD structure)
described in the model section.

This is the standalone version of the mpr_jax.ipynb cells (`plot` + `plot_fc_fcd`).

Usage:
  python plot_model_dynamics.py                 # default G grid, t_end=300000
  SIM_T_END=100000 python plot_model_dynamics.py --G 0.2 0.5 0.7 --out ../../results
"""
import os, argparse
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mpr_jax
import utils
from FCD_jax import extract_FCD


def simulate(G, eta, t_end, seed=42):
    """Run the MPR model at coupling G and excitability eta; return r, BOLD, time axes."""
    weights = jnp.array(np.loadtxt(utils.DATA_ROOT + "/weights.txt"))
    params = {"G": float(G), "t_end": t_end, "weights": weights / jnp.max(weights),
              "dt": 0.01, "eta": jnp.array([float(eta)]), "rv_decimate": 10,
              "noise_amp": 0.037, "tr": 300.0, "seed": seed}
    sde = mpr_jax.MPR_sde.create(params)
    out = sde.run({}, record_rv=True)
    # rv_d is (time, 2*nn): r and v CONCATENATED along the last axis, not stacked as
    # (time, 2, nn). Indexing [:, 0, :] raises "Too many indices: 2-dimensional array
    # indexed with 3 regular indices". Mirrors the notebook's rv_d[:, :nn] / [:, nn:].
    nn = int(weights.shape[0])
    r = np.asarray(out["rv_d"][:, :nn])           # (time, nodes) firing rate
    bold = np.asarray(out["bold_d"])              # (time, nodes)
    rv_t = np.asarray(out["rv_t"]); bold_t = np.asarray(out["bold_t"])
    return r, rv_t, bold, bold_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--G", type=float, nargs="+", default=[0.2, 0.33, 0.5, 0.7],
                    help="coupling values (swept unless --eta has >1 value); default matches inference")
    ap.add_argument("--eta", type=float, nargs="+", default=[-4.6],
                    help="excitability values; pass >1 (e.g. -5.5 -4.6 -3.7, prior U(-6,-3.5)) "
                         "to sweep eta at fixed G instead of sweeping G")
    ap.add_argument("--t_end", type=int, default=int(os.environ.get("SIM_T_END", 300000)))
    ap.add_argument("--cut", type=int, default=50,
                    help="drop initial BOLD frames (transient + Balloon-Windkessel startup); "
                         "50 matches the mpr_jax.ipynb cells this script reproduces (t_end=300000 -> 1000 frames)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or utils.results_folder()

    # sweep eta (at fixed G) if multiple eta given, else sweep G (at fixed eta)
    if len(args.eta) > 1:
        sweep, fixed = "eta", f"G={args.G[0]}"
        rows = [(args.G[0], e, f"$\\eta$={e}") for e in args.eta]
    else:
        sweep, fixed = "G", f"$\\eta$={args.eta[0]}"
        rows = [(g, args.eta[0], f"G={g}") for g in args.G]

    nrow = len(rows)
    fig, axes = plt.subplots(nrow, 4, figsize=(16, 3.4 * nrow),
                             gridspec_kw={"width_ratios": [2.2, 2.2, 1, 1]})
    axes = np.atleast_2d(axes)
    fig.suptitle(f"MPR dynamics: sweep {sweep} ({fixed} fixed), t_end={args.t_end}", y=1.0)

    for row, (G, eta, rlabel) in enumerate(rows):
        r, rv_t, bold, bold_t = simulate(G, eta, args.t_end)
        # cut once, then feed the SAME trimmed series to both features.
        # extract_FCD expects (nodes, timepoints) -- pass b.T, not b (or set coldata=True).
        b = bold[args.cut:]
        FC = np.corrcoef(b.T)
        # extract_FCD returns (fcd_matrix, corr_vectors, shift) -- unpack it. Wrapping
        # the whole tuple in np.asarray gives "inhomogeneous shape ... detected (3,)".
        FCD, _, _ = extract_FCD(b.T, wwidth=30, maxNwindows=200, olap=0.94, mode="corr")
        FCD = np.asarray(FCD)

        ax = axes[row, 0]                                    # firing rate
        ax.plot(rv_t, r[:, :min(5, r.shape[1])], lw=0.5)
        ax.set_ylabel(f"{rlabel}\nr(t)"); ax.set_xlabel("time")
        if row == 0: ax.set_title("firing rate (5 regions)")

        ax = axes[row, 1]                                    # BOLD
        ax.plot(bold_t, bold[:, :min(5, bold.shape[1])], lw=0.6)
        ax.set_ylabel("BOLD"); ax.set_xlabel("time")
        if row == 0: ax.set_title("BOLD (5 regions)")

        ax = axes[row, 2]                                    # FC
        im = ax.imshow(FC, vmin=-1, vmax=1, cmap="jet"); ax.set_xticks([]); ax.set_yticks([])
        if row == 0: ax.set_title("FC")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[row, 3]                                    # FCD
        im = ax.imshow(FCD, vmin=0, vmax=1, cmap="jet"); ax.set_xticks([]); ax.set_yticks([])
        if row == 0: ax.set_title("FCD")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        print(f"G={G} eta={eta}: r{r.shape} bold{bold.shape} FC{FC.shape} FCD{FCD.shape}")

    fig.tight_layout()
    f = os.path.join(out, f"model_dynamics_sweep{sweep}_tend{args.t_end}.png")
    fig.savefig(f, dpi=200); plt.close(fig)
    print(f"wrote {f}")


if __name__ == "__main__":
    main()
