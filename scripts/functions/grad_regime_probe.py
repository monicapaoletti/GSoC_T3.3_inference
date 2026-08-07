#!/usr/bin/env python3
r"""Does the dynamical regime cause the gradient failure? Measure it directly.

The manuscript reports gradient-based NUTS as non-viable and attributes it to chaos "in
the biologically interesting regimes". But the benchmark configuration (10 nodes,
eta=-4.6) is NOT in such a regime: measured at t_end=300000, its BOLD sd is flat at
~0.004 and its FCD statistics are constant to 3% across G in {0.2,0.33,0.5,0.7}, while
the 88-node model swings 300x in FC mean over the same range. So the failure was observed
in one regime and explained by another. This script closes that gap.

METHOD. `grad_horizon` bounds how many neural steps the reverse pass flows back through
(mpr_jax.py:203-218); stop_gradient is the identity forward, so the FORWARD PASS IS
BIT-EXACT for every horizon and only the gradient changes. In a chaotic recursion the
backprop through k steps grows like exp(lambda*k), so sweeping the horizon and reading
the slope of log|grad| vs k measures the amplification directly -- a Lyapunov-like
exponent for the actual model, not a proxy. Running that sweep at two excitabilities
answers the question that motivated this:

    eta = -4.6  the value used throughout the benchmark; near-quiescent at 10 nodes
    eta = -4.2  the LIVE window for the 10-node subnetwork (BOLD oscillates, FCD blocks)

If |grad| grows faster at -4.2, chaos is confirmed as the mechanism AND moving to the
live regime would have made gradient-based sampling worse, not better. If it grows no
faster, the chaos explanation does not survive and the step-size collapse has some other
cause -- which would matter before the 2-D campaign.

The finite-difference gradient is reported as a reference. It differentiates the forward
map rather than the recursion, so it is horizon-INDEPENDENT by construction: a flat line
against which the autodiff growth is read.

Usage:
    python grad_regime_probe.py --out ../../results/$(date +%F)
    python grad_regime_probe.py --eta -4.6 -4.2 --which_stat FCD --G 0.2
"""
import os, csv, argparse, time
import numpy as np
import jax, jax.numpy as jnp

import utils
from mpr_jax_numpyro import make_forward_fn, precompute_shift_and_starts


def build(eta, which_stat, sc_size, t_end, cut, tr, grad_horizon, grad_method, seed=42):
    """Forward map at fixed eta. Mirrors mpr_jax_numpyro's own construction so the
    probe measures the same object the sampler differentiates."""
    weights = np.loadtxt(os.path.join(utils.DATA_ROOT, "weights.txt"))[:sc_size, :sc_size]
    nn = len(weights)
    par = {
        "G": 0.2, "weights": jnp.array(weights) / jnp.max(jnp.array(weights)),
        "t_end": t_end, "dt": 0.01, "eta": jnp.array([float(eta)]),
        "rv_decimate": 10, "noise_amp": 0.037, "tr": 300.0, "seed": seed,
        "clip_mode": "hard",
    }
    T = (t_end - cut) // tr
    _shift, starts = precompute_shift_and_starts(T, wwidth=30, olap=0.94, stride1=True)
    return make_forward_fn(par, which_stat, cut, tr, starts, nn,
                           grad_horizon=grad_horizon, fast_bold=True, eps=0.0,
                           grad_method=grad_method, fd_h=1e-2,
                           fisher_z=True, keep_negative=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eta", type=float, nargs="+", default=[-4.6, -4.2])
    ap.add_argument("--G", type=float, default=0.2,
                    help="GROUND-TRUTH coupling: the observed features are generated here")
    ap.add_argument("--G_eval", type=float, default=None,
                    help="coupling at which the gradient is evaluated; default G+0.15. "
                         "Must differ from --G: at the truth the loss is exactly 0 and "
                         "so is its gradient, which measures nothing. A sampler's "
                         "gradients matter where it actually is -- away from the mode.")
    ap.add_argument("--which_stat", default="FCD", choices=["FC", "FCD"])
    ap.add_argument("--SC_size", type=int, default=10)
    ap.add_argument("--t_end", type=int, default=30000)
    ap.add_argument("--cut", type=int, default=10)
    ap.add_argument("--tr", type=int, default=1)
    ap.add_argument("--horizons", type=int, nargs="+", default=[0],
                    help="grad_horizon values; 0 = untruncated full BPTT. Defaults to "
                         "[0] ONLY: measured 2026-08-06, every nonzero horizon returns "
                         "an identically ZERO autodiff gradient (verified at horizons "
                         "10-3000, fast_bold on and off, both clip modes), so the "
                         "horizon axis carries no signal until that is fixed. See the "
                         "module docstring.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                    help="noise realisations; the comparison between etas must survive "
                         "reseeding or it is one draw, not an effect")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or utils.results_folder()
    os.makedirs(out, exist_ok=True)
    g_evals = ([args.G_eval] if args.G_eval is not None
               else [args.G + d for d in (0.05, 0.15, 0.30)])

    rows = []
    for eta in args.eta:
        for seed in args.seeds:
            # Observed data at THIS eta and seed, frozen noise: the loss must be the one
            # a sampler would actually climb. Sharing one obs across etas would compare
            # gradients of different objectives, confounding regime with misspecification.
            f_obs = build(eta, args.which_stat, args.SC_size, args.t_end, args.cut,
                          args.tr, grad_horizon=0, grad_method="autodiff", seed=seed)
            obs = np.asarray(jax.jit(f_obs)(args.G))

            def loss_factory(gh, gm):
                f = build(eta, args.which_stat, args.SC_size, args.t_end, args.cut,
                          args.tr, grad_horizon=gh, grad_method=gm, seed=seed)
                o = jnp.asarray(obs)
                return lambda G: 0.5 * jnp.sum((f(G) - o) ** 2)

            grad_fd = jax.jit(jax.grad(loss_factory(0, "fd")))
            grads_ad = {gh: jax.jit(jax.grad(loss_factory(gh, "autodiff")))
                        for gh in args.horizons}

            for ge in g_evals:
                t0 = time.time()
                try:
                    v = float(grad_fd(ge)); st = "ok"
                except Exception as e:
                    v, st = float("nan"), type(e).__name__
                rows.append({"eta": eta, "seed": seed, "G_eval": ge, "method": "fd",
                             "grad_horizon": "n/a", "grad": v, "abs_grad": abs(v),
                             "seconds": round(time.time() - t0, 1), "status": st})
                print(f"[probe] eta={eta} seed={seed} G={ge:.2f}  fd   "
                      f"grad={v:+.6g} ({st})", flush=True)

                for gh, fn in grads_ad.items():
                    t0 = time.time()
                    try:
                        v = float(fn(ge)); st = "ok"
                    except Exception as e:      # OOM on full BPTT is itself a result
                        v, st = float("nan"), type(e).__name__
                    rows.append({"eta": eta, "seed": seed, "G_eval": ge,
                                 "method": "autodiff", "grad_horizon": gh or "full",
                                 "grad": v, "abs_grad": abs(v),
                                 "seconds": round(time.time() - t0, 1), "status": st})
                    print(f"[probe] eta={eta} seed={seed} G={ge:.2f}  autodiff/"
                          f"{gh or 'full'}  grad={v:+.6g} ({st})", flush=True)

    path = os.path.join(out, f"grad_regime_probe_{args.which_stat}_SC{args.SC_size}"
                             f"_G{args.G}_tend{args.t_end}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["eta", "seed", "G_eval", "method",
                                           "grad_horizon", "grad", "abs_grad",
                                           "seconds", "status"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {path}")

    # THE comparison: does the live regime amplify gradients relative to the
    # near-quiescent one the benchmark actually used?
    print("\n[probe] median |grad| by eta (over seeds x G_eval):")
    for meth in ("fd", "autodiff"):
        print(f"  method={meth}")
        med = {}
        for eta in args.eta:
            vals = [r["abs_grad"] for r in rows
                    if r["eta"] == eta and r["method"] == meth
                    and np.isfinite(r["abs_grad"])]
            if vals:
                med[eta] = float(np.median(vals))
                print(f"    eta={eta}: median |grad| = {med[eta]:.6g}  (n={len(vals)})")
        if len(med) == 2:
            a, b = sorted(med)                     # a = more negative eta (quieter)
            print(f"    ratio  eta={b} / eta={a}  =  {med[b] / med[a]:.3f}x"
                  "   (>1 means the LIVE regime amplifies gradients)")


if __name__ == "__main__":
    main()
