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


def simulate(G, eta, t_end, seed=42, sc_size=None):
    """Run the MPR model at coupling G and excitability eta; return r, BOLD, time axes.

    sc_size subsets the connectome to its first n nodes. The slice happens BEFORE the
    max-normalisation, exactly as mpr_jax_numpyro does it -- normalising the full matrix
    and then slicing would give a different effective coupling at the same G, so the
    figure would not describe the network the inference actually runs on.
    """
    weights = jnp.array(np.loadtxt(utils.DATA_ROOT + "/weights.txt"))
    if sc_size:
        weights = weights[:sc_size, :sc_size]
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


def _write_stats(stats, out, tag, sweep, fixed, args):
    r"""Emit the figure's companion table as CSV and as a LaTeX fragment.

    The figure shows regimes qualitatively -- "FCD gains structure around G=0.5" is a
    claim about a picture. These are the same claims as numbers, from the same runs, so
    a reader can check what they think they see. Kept beside the figure rather than in
    make_paper_assets because it describes simulations, not benchmark cells: it has no
    master_results.csv row to be derived from.
    """
    import csv
    cols = ["G", "eta", "r_mean", "r_std", "bold_sd",
            "fc_mean", "fc_sd", "fcd_mean", "fcd_sd"]
    csv_path = os.path.join(out, f"model_dynamics_{tag}_stats.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for s in stats:
            w.writerow({c: s[c] for c in cols})
    print(f"wrote {csv_path}")

    swept = "G" if sweep == "G" else r"$\eta$"
    net = f"{args.SC_size}-node" if args.SC_size else "88-node"
    hdr = (r"$G$" if sweep == "G" else r"$\eta$")
    lines = [
        r"\begin{table}[t]\centering",
        rf"\caption{{Summary statistics for the regimes in the model-dynamics figure, "
        rf"computed from the same simulations ({net} network, "
        rf"$t_{{\mathrm{{end}}}}={args.t_end}$, {fixed}). $r$ is the firing rate; "
        rf"BOLD sd is the temporal sd per region averaged over regions. FC mean is "
        rf"overall synchrony and FC sd how differentiated the pairs are. The regime "
        rf"structure is carried by the MEANS: FC mean and FCD mean both peak sharply at "
        rf"the intermediate coupling and fall away on either side, while FCD sd stays "
        rf"nearly constant across every regime and so does not discriminate them. Note "
        rf"the ordering is not monotonic in $G$ -- the strongest coupling has the "
        rf"highest firing rate but near-baseline FC and FCD.}}",
        rf"\label{{tab:dynamics:{sweep}}}\small\setlength{{\tabcolsep}}{{5pt}}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        hdr + r" & $\bar r$ & sd$(r)$ & BOLD sd & FC mean & FC sd & FCD mean & FCD sd \\",
        r"\midrule",
    ]
    for s in stats:
        key = s["G"] if sweep == "G" else s["eta"]
        lines.append(
            f"{key:g} & {s['r_mean']:.3g} & {s['r_std']:.3g} & {s['bold_sd']:.3g} & "
            f"{s['fc_mean']:.3g} & {s['fc_sd']:.3g} & {s['fcd_mean']:.3g} & "
            f"{s['fcd_sd']:.3g} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path = os.path.join(out, f"model_dynamics_{tag}_stats.tex")
    with open(tex_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {tex_path}")


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
    ap.add_argument("--SC_size", type=int, default=None,
                    help="use only the first N nodes (default: the full 88-node matrix). "
                         "Pass 10 to match the benchmark's subnetwork.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    # results_folder() creates today's dated folder; an explicit --out did not exist
    # until the figure tried to save into it, which failed after the simulation had
    # already run (minutes at t_end=300000).
    out = args.out or utils.results_folder()
    os.makedirs(out, exist_ok=True)

    # sweep eta (at fixed G) if multiple eta given, else sweep G (at fixed eta)
    if len(args.eta) > 1:
        sweep, fixed = "eta", f"G={args.G[0]}"
        rows = [(args.G[0], e, f"$\\eta$={e}") for e in args.eta]
    else:
        sweep, fixed = "G", f"$\\eta$={args.eta[0]}"
        rows = [(g, args.eta[0], f"G={g}") for g in args.G]

    # Layout and styling follow the mpr_jax.ipynb `plot_fc_fcd` cell this script is the
    # standalone version of: traces stacked on the LEFT (firing rate above BOLD), FC and
    # FCD as square matrices on the RIGHT, 'hot' colormap, FC on [-0.5, 1] and FCD on
    # [0, 1], every region drawn at lw=0.1, and time in SECONDS (the model integrates in
    # ms, so the axis is divided by 1000).
    nrow = len(rows)
    stats = []                    # one entry per row; written out beside the figure
    step = 10                     # decimate traces for a legible line density
    fig = plt.figure(figsize=(13, 2.35 * nrow))
    # Column 0 is a narrow strip carrying the row label (the swept value). Putting it
    # there rather than as a title over the traces stops it colliding with the previous
    # row's "Time (s)" label, and makes it read as labelling the WHOLE row -- traces,
    # FC and FCD -- which is what it does.
    gs = fig.add_gridspec(2 * nrow, 6, width_ratios=[0.13, 1, 1, 1, 1.05, 1.05],
                          hspace=0.45, wspace=0.5)
    # No suptitle: the manuscript gives every figure a caption, so a title inside the
    # image duplicates it and eats vertical space. The swept value is on each row and
    # the fixed one belongs in the caption.

    for row, (G, eta, rlabel) in enumerate(rows):
        r, rv_t, bold, bold_t = simulate(G, eta, args.t_end, sc_size=args.SC_size)
        # cut once, then feed the SAME trimmed series to both features.
        # extract_FCD expects (nodes, timepoints) -- pass b.T, not b (or set coldata=True).
        b = bold[args.cut:]
        FC = np.corrcoef(b.T)
        # extract_FCD returns (fcd_matrix, corr_vectors, shift) -- unpack it. Wrapping
        # the whole tuple in np.asarray gives "inhomogeneous shape ... detected (3,)".
        FCD, _, _ = extract_FCD(b.T, wwidth=30, maxNwindows=200, olap=0.94, mode="corr")
        FCD = np.asarray(FCD)

        # Firing rate and BOLD are the SAME run -- BOLD is simply subsampled -- so they
        # must span the same window and end together.
        #
        # They do not, as returned: mpr_jax.py builds rv_t as arange(n)*(rv_decimate*dt*10)
        # but bold_t as linspace(0, t_end, n)*10, applying the x10 to a per-sample
        # increment in one case and to a total duration in the other. Measured on one run,
        # rv_t spans 0..30000 while bold_t spans 0..267000 -- a factor of ~10. rv_t is the
        # correct one (t_end=300000 -> 300 s); bold_t carries a spurious extra decade. So
        # the BOLD axis is rebuilt from the firing-rate span rather than trusted. The
        # notebook never hit this because its two panels are independent and autoscale.
        t_s = np.asarray(rv_t)[::step] / 1000.0          # seconds
        T = float(t_s[-1])
        b_x = np.linspace(0.0, T, bold.shape[0])         # same window, subsampled

        ax0 = fig.add_subplot(gs[2 * row:2 * row + 2, 0])
        ax0.axis("off")
        ax0.text(0.5, 0.5, rlabel, rotation=90, ha="center", va="center", fontsize=14)

        ax1 = fig.add_subplot(gs[2 * row, 1:4])
        ax1.plot(t_s, r[::step, :], lw=0.1)
        ax1.set_ylabel("r")
        ax1.set_xlim(0, T)
        ax1.set_xticklabels([])
        for s in ("top", "right"):
            ax1.spines[s].set_visible(False)

        ax2 = fig.add_subplot(gs[2 * row + 1, 1:4])
        ax2.plot(b_x, bold, lw=0.1)
        ax2.set_ylabel("BOLD")
        # Only the bottom row carries the axis label: every row shares the same axis, so
        # repeating it is noise, and at tight row spacing it collides with the next row's
        # firing-rate panel. Tick numbers stay on every row.
        if row == nrow - 1:
            ax2.set_xlabel("Time (s)")
        ax2.set_xlim(0, T)
        for s in ("top", "right"):
            ax2.spines[s].set_visible(False)

        # --- right: FC and FCD, square, notebook colormap and limits ---
        ax3 = fig.add_subplot(gs[2 * row:2 * row + 2, 4])
        im1 = ax3.imshow(FC, vmin=-0.5, vmax=1, cmap="hot")
        ax3.set_title("FC"); ax3.set_xticks([]); ax3.set_yticks([])
        fig.colorbar(im1, ax=ax3, fraction=0.046, pad=0.04)

        ax4 = fig.add_subplot(gs[2 * row:2 * row + 2, 5])
        im2 = ax4.imshow(FCD, vmin=0, vmax=1, cmap="hot")
        ax4.set_title("FCD"); ax4.set_xticks([]); ax4.set_yticks([])
        fig.colorbar(im2, ax=ax4, fraction=0.046, pad=0.04)

        # --- companion statistics, from THIS run ---
        # Computed inside the same loop, from the same arrays the panels are drawn from,
        # so the table cannot drift from the figure. (Recomputing them in a second script
        # is how make_paper_assets' recovery figure came to disagree with its own table.)
        iu = np.triu_indices_from(FC, k=1)          # off-diagonal only: the unit diagonal
        fc_off = FC[iu]                             # would inflate every FC summary
        fcd_off = FCD[np.triu_indices_from(FCD, k=1)]
        stats.append({
            "G": float(G), "eta": float(eta),
            # firing-rate level and its variability across time and regions
            "r_mean": float(np.mean(r)), "r_std": float(np.std(r)),
            # BOLD amplitude: temporal sd per region, averaged over regions
            "bold_sd": float(np.mean(np.std(bold, axis=0))),
            # FC: mean is overall synchrony, sd is how differentiated the pairs are.
            # A saturated network has HIGH mean and LOW sd -- everything correlated with
            # everything -- which is why both are needed to read the FC panel.
            "fc_mean": float(np.mean(fc_off)), "fc_sd": float(np.std(fc_off)),
            # FCD sd is the switching measure: a static regime gives a near-constant FCD
            # (sd ~ 0) whatever its mean, while switching produces the off-diagonal
            # blocks visible in the panel.
            "fcd_mean": float(np.mean(fcd_off)), "fcd_sd": float(np.std(fcd_off)),
        })

        print(f"G={G} eta={eta}: r{r.shape} bold{bold.shape} "
              f"FC{FC.shape} FCD{FCD.shape} ({r.shape[1]} regions plotted)")

    fig.tight_layout()
    tag = f"sweep{sweep}_tend{args.t_end}" + (f"_SC{args.SC_size}" if args.SC_size else "")
    f = os.path.join(out, f"model_dynamics_{tag}.png")
    fig.savefig(f, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {f}")
    _write_stats(stats, out, tag, sweep, fixed, args)


if __name__ == "__main__":
    main()
