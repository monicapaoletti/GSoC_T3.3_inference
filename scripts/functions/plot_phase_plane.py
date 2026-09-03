"""(v, r) phase-plane trajectories: a fit that captures the noise-driven jump vs one that does not.

Meysam's request (his item 4, after Baldy et al. 2024 Fig. 4): show the trajectory in the
(v, r) plane for a method that works beside one that fails. No nullclines or vector field
-- those are single-node objects and this is a whole-brain model.

At G*=0.5, ABC-SMC on FCD recovers Ghat=0.510 while DEMetropolis-Z on FCD returns
Ghat=0.046 (R-hat 1.85, 0% coverage), i.e. an essentially uncoupled network.

TWO VERSIONS, because the scales disagree and the choice is a judgement call:
  --sc_size 88 : the switching regime. The contrast is dramatic, but the Ghat values were
                 fitted on the 10-node subnetwork, so simulation and inference use
                 different network sizes.
  --sc_size 10 : the network the inference actually ran on -- where, as the figure shows,
                 NO coupling in this range produces switching at all.

Usage: python plot_phase_plane.py --sc_size 88 --out results/<date>
"""
import os, argparse
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import utils, mpr_jax, paper_style

# (label, G, colour). The truth is the reference both fits are judged against.
CASES = [(r"observed" "\n" r"$G^\star=0.50$",     0.500, paper_style.INK),
         (r"ABC-SMC, FCD" "\n" r"$\hat G=0.510$", 0.510, "#2a78d6"),
         (r"DEMetropolis-Z, FCD" "\n" r"$\hat G=0.046$", 0.046, "#eb6834")]


def nullclines(ax, r_grid, Ibar, delta=0.7, J=14.5, eta=-4.6, tau=1.0):
    """r- and v-nullclines of the MPR node, at a FIXED network input Ibar.

        dr/dt = 0  ->  v = -delta / (2 pi tau r)
        dv/dt = 0  ->  v = +- sqrt((pi tau r)^2 - J tau r - eta - Ibar)

    Ibar is the coupling term G*sum_j W_ij r_j, which for a coupled node changes as the
    rest of the network moves. Fixing it at its mean is what makes these curves an
    approximation rather than the nullclines of the system actually simulated.
    """
    import numpy as _np
    r = r_grid[r_grid > 1e-3]
    v_r = -delta / (2.0 * _np.pi * tau * r)
    ax.plot(r, v_r, "-", color="#4a3aa7", lw=1.1, alpha=0.9, zorder=5,
            label=r"$\dot r = 0$")
    rad = (_np.pi * tau * r) ** 2 - J * tau * r - eta - Ibar
    ok = rad >= 0
    if ok.any():
        vv = _np.sqrt(rad[ok])
        ax.plot(r[ok], vv, "-", color="#008300", lw=1.1, alpha=0.9, zorder=5,
                label=r"$\dot v = 0$")
        ax.plot(r[ok], -vv, "-", color="#008300", lw=1.1, alpha=0.9, zorder=5)


def simulate(G, sc_size, t_end, eta=-4.6, seed=42):
    w = jnp.array(np.loadtxt(utils.DATA_ROOT + "/weights.txt"))
    if sc_size:
        w = w[:sc_size, :sc_size]
    p = {"G": float(G), "t_end": t_end, "weights": w / jnp.max(w), "dt": 0.01,
         "eta": jnp.array([float(eta)]), "rv_decimate": 10, "noise_amp": 0.037,
         "tr": 300.0, "seed": seed, "clip_mode": "hard"}
    out = mpr_jax.MPR_sde.create(p).run({}, record_rv=True)
    nn = int(w.shape[0])
    rv = np.asarray(out["rv_d"])
    return rv[:, :nn], rv[:, nn:], np.asarray(w / jnp.max(w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc_size", type=int, default=88, help="88 = switching regime; 10 = the benchmark subnetwork")
    ap.add_argument("--t_end", type=int, default=300000)
    ap.add_argument("--node", type=int, default=0)
    ap.add_argument("--thin", type=int, default=3)
    ap.add_argument("--style", choices=["dots", "lines"], default="dots",
                    help="dots = thinned point cloud over the whole run (occupancy); "
                         "lines = a CONTIGUOUS segment drawn as a path (a trajectory in "
                         "the Baldy et al. sense). Thinning and joining cannot be mixed: "
                         "connecting samples that are far apart during a transition draws "
                         "long straight chords the system never traverses.")
    ap.add_argument("--seg", type=int, default=30000,
                    help="lines: length of the contiguous segment, in recorded samples")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nullclines", action="store_true",
                    help="overlay the r- and v-nullclines. For a COUPLED node the "
                         "v-nullcline moves with the network input G*sum_j W_ij r_j, "
                         "which is time varying, so there is no single curve: it is drawn "
                         "at the mean input over the run and is an approximation.")
    ap.add_argument("--paper", default=None)
    a = ap.parse_args()
    sc = None if a.sc_size >= 88 else a.sc_size
    tag = f"{a.sc_size}node"

    paper_style.use(grid=False)
    fig, axes = plt.subplots(1, 3, figsize=paper_style.grid_figsize(1, 3, 1.0, 1.02),
                             sharex=True, sharey=True)

    traj = []
    for (label, G, colour), ax in zip(CASES, axes):
        r, v, W = simulate(G, sc, a.t_end)
        if a.style == "lines":
            # A contiguous stretch, taken from the middle so the transient is past.
            i0 = max(0, (r.shape[0] - a.seg) // 2)
            sl = slice(i0, i0 + a.seg)
        else:
            sl = slice(None, None, a.thin)
        r0, v0 = r[sl, a.node], v[sl, a.node]
        traj.append((r0, v0))
        # The observed run sits faintly behind each fit so the comparison is in-panel.
        r_ref, v_ref = traj[0]
        if a.style == "lines":
            ax.plot(r_ref, v_ref, "-", color="#d5d4cf", lw=0.5, alpha=0.9, zorder=1,
                    rasterized=True, solid_joinstyle="round")
            ax.plot(r0, v0, "-", color=colour, lw=0.5, alpha=0.6, zorder=2,
                    rasterized=True, solid_joinstyle="round")
            ax.plot(r0[0], v0[0], "o", color=colour, ms=5, alpha=0.9, zorder=3)
        else:
            ax.plot(r_ref, v_ref, ".", color="#c9c8c3", ms=0.7, alpha=0.35, zorder=1,
                    rasterized=True)
            ax.plot(r0, v0, ".", color=colour, ms=0.7, alpha=0.30, zorder=2,
                    rasterized=True)
        up = float(np.mean(r[:, a.node] > 1.0))     # over the WHOLE run, not the segment
        ax.set_title(f"{label}\n{up:.0%} of time up", pad=6, fontsize=9.5)
        if a.nullclines:
            # mean network input for this node, from the run itself
            Ibar = float(G * np.mean(r @ W[a.node]))
            grid = np.linspace(1e-3, float(np.nanmax(r[:, a.node])) * 1.1, 400)
            nullclines(ax, grid, Ibar)
            ax.set_ylim(np.nanmin(v[:, a.node]) * 1.15, np.nanmax(v[:, a.node]) * 1.15)
        ax.set_xlabel(r"firing rate $r$")
        ax.axvline(1.0, color=paper_style.GRID, lw=1.0, ls="--", zorder=0)
    axes[0].set_ylabel(r"membrane potential $v$")
    if a.nullclines:
        axes[0].legend(frameon=False, fontsize=8, loc="lower right")

    fig.tight_layout()
    os.makedirs(a.out, exist_ok=True)
    sfx = ("" if a.style == "dots" else "_lines") + ("_nullclines" if a.nullclines else "")
    out = os.path.join(a.out, f"phase_plane_{tag}{sfx}.png")
    fig.savefig(out); plt.close(fig)
    print("wrote", out)
    if a.paper:
        import shutil; shutil.copy(out, a.paper)


if __name__ == "__main__":
    main()
