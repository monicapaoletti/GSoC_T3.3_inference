#!/usr/bin/env python3
r"""(G, eta) regime map: where is the model alive, measured rather than eyeballed.

Choosing `--eta_prior_scale` for the 2-D inference currently rests on looking at sweep
figures. That is the wrong instrument for the question, which is quantitative: how wide,
in eta, is the region where the features carry information about the parameters at all?

METHOD. Evaluate the SAME forward map the sampler uses (make_forward_fn, so features are
extracted identically) on a grid of (G, eta), and reduce each cell to summary scalars.
The whole grid is one `jax.vmap` call: this is the batched axis the paper is about, and
it is why the map belongs on the GPU rather than being looped on a host.

WHICH SCALAR -- and it depends on the horizon, which is easy to get wrong. At
t_end=300000 (the illustrative sweeps) FC MEAN discriminates: 0.0012 to 0.38 across
regimes. At t_end=30000, the horizon the inference actually uses, it does NOT: a 30 s run
leaves ~70 BOLD samples and the shared initial transient dominates them, so every region
pair correlates near 1 whatever the regime (measured: FC mean 0.998 at eta=-5.0 against
0.585 at -4.2, i.e. HIGHEST where the network is silent). The discriminating scalar at
this horizon is FC SD, the dispersion across region pairs -- 0.0012, 0.045, 0.195 over
the same three eta. High mean with near-zero sd is the "everything correlates with
everything, so nothing is informative" signature. FCD sd was rejected outright: it is
0.130-0.140 across every regime at 88 nodes and discriminates nothing anywhere. All four
scalars are recorded; the live-window summary below reads FC sd.

HORIZON. t_end defaults to the INFERENCE horizon (30000), not the 300000 used for the
illustrative sweeps: the question is which priors make the inference informative, so the
map must describe the data the inference actually sees.

Usage:
  python eta_regime_map.py --SC_size 10                 # the network inference runs on
  python eta_regime_map.py --SC_size 88 --batch 32      # full network, smaller chunks
"""
import os, csv, argparse, time
import numpy as np
import jax, jax.numpy as jnp

import utils
import mpr_jax
from mpr_jax_numpyro import make_forward_fn, precompute_shift_and_starts


def build(which_stat, sc_size, t_end, cut, tr, seed):
    weights = np.loadtxt(os.path.join(utils.DATA_ROOT, "weights.txt"))[:sc_size, :sc_size]
    nn = len(weights)
    par = {
        "G": 0.2, "weights": jnp.array(weights) / jnp.max(jnp.array(weights)),
        "t_end": t_end, "dt": 0.01, "eta": jnp.array([-4.6]),
        "rv_decimate": 10, "noise_amp": 0.037, "tr": 300.0, "seed": seed,
        "clip_mode": "hard",
    }
    # T must be the number of BOLD samples the model ACTUALLY produces, measured from a
    # reference run exactly as mpr_jax_numpyro.py:538-540 does. Deriving it as
    # (t_end-cut)//tr gives ~29990 instead of ~80 here, so the FCD window list is three
    # orders of magnitude too long and the call never returns.
    _ref = mpr_jax.MPR_sde.create(par).run({}, record_rv=False, fast_bold=True)["bold_d"]
    T = int(np.asarray(_ref)[cut::tr].shape[0])
    _shift, starts = precompute_shift_and_starts(T, wwidth=30, olap=0.94, stride1=True)
    # grad_method="autodiff" returns the plain forward; no gradients are taken here.
    return make_forward_fn(par, which_stat, cut, tr, starts, nn, grad_horizon=0,
                           fast_bold=True, eps=0.0, grad_method="autodiff",
                           fisher_z=False, keep_negative=True), nn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--G", type=float, nargs="+",
                    default=[0.05, 0.1, 0.2, 0.33, 0.5, 0.7, 0.9, 1.2])
    ap.add_argument("--eta", type=float, nargs="+",
                    default=[-6.0, -5.6, -5.2, -5.0, -4.8, -4.6, -4.4, -4.2,
                             -4.0, -3.8, -3.4, -3.0])
    ap.add_argument("--SC_size", type=int, default=10)
    ap.add_argument("--t_end", type=int, default=30000)
    ap.add_argument("--cut", type=int, default=10)
    ap.add_argument("--tr", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                    help="the live window must survive reseeding or it is one draw")
    ap.add_argument("--batch", type=int, default=0,
                    help="grid points per vmap call; 0 = the whole grid at once")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or utils.results_folder()
    os.makedirs(out, exist_ok=True)

    grid = np.array([(g, -e) for g in args.G for e in args.eta], float)  # [G, eta_mag]
    print(f"[map] {len(args.G)} G x {len(args.eta)} eta = {len(grid)} cells, "
          f"{len(args.seeds)} seeds, SC={args.SC_size}, t_end={args.t_end}", flush=True)

    rows = []
    for stat in ("FC", "FCD"):
        fwd, nn = build(stat, args.SC_size, args.t_end, args.cut, args.tr, args.seeds[0])
        vf = jax.jit(jax.vmap(fwd))
        for seed in args.seeds:
            f, _ = build(stat, args.SC_size, args.t_end, args.cut, args.tr, seed)
            vf = jax.jit(jax.vmap(f))
            step = args.batch or len(grid)
            t0 = time.time()
            outs = []
            for i in range(0, len(grid), step):
                outs.append(np.asarray(vf(jnp.asarray(grid[i:i + step]))))
            F = np.concatenate(outs, axis=0)          # (cells, n_features)
            dt = time.time() - t0
            print(f"[map] {stat} seed={seed}: {F.shape} in {dt:.1f}s", flush=True)
            for (g, em), v in zip(grid, F):
                v = v[np.isfinite(v)]
                rows.append({"which_stat": stat, "seed": seed, "G": float(g),
                             "eta": float(-em),
                             "mean": float(np.mean(v)) if v.size else float("nan"),
                             "sd": float(np.std(v)) if v.size else float("nan")})

    path = os.path.join(out, f"eta_regime_map_SC{args.SC_size}_tend{args.t_end}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["which_stat", "seed", "G", "eta", "mean", "sd"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {path}")

    # The headline the prior choice needs: for each G, the eta range whose FC SD is
    # within 50% of that row's maximum -- an operational "live window". Reading the MEAN
    # here would invert the answer at this horizon (see the module docstring).
    print("\n[map] live eta window per G (FC sd >= 50% of the row max, median over seeds):")
    import collections
    acc = collections.defaultdict(list)
    for r in rows:
        if r["which_stat"] == "FC":
            acc[(r["G"], r["eta"])].append(r["sd"])
    for g in args.G:
        vals = [(e, float(np.median(acc[(g, e)]))) for e in args.eta if acc[(g, e)]]
        if not vals:
            continue
        top = max(v for _, v in vals)
        live = [e for e, v in vals if top > 0 and v >= 0.5 * top]
        span = f"[{min(live):.1f}, {max(live):.1f}]" if live else "none"
        print(f"   G={g:<5} peak FC sd={top:.4g} at eta={max(vals, key=lambda t: t[1])[0]:.1f}"
              f"   live window {span}")


if __name__ == "__main__":
    main()
