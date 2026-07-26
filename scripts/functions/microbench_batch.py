"""Micro-benchmark: does jax.vmap(forward_fn) over a batch of G-values stay
~flat in wall-time as the batch grows (i.e. does batching fill the idle GPU)?

Isolates the GPU-batching lever from the expensive NUTS trajectory overhead.
Run once on the GPU, once with JAX_PLATFORMS=cpu; each run writes
results/<DATE>/forward_throughput_<platform>.csv. plot_forward_throughput.py
then overlays the two.

Usage:
    python microbench_batch.py [--outdir DIR] [--sc_size 10] [--reps 3]
"""
import os, time, argparse
from datetime import date
import numpy as np
import jax, jax.numpy as jnp

# reuse the repo's real forward-model machinery
import utils, mpr_jax
from mpr_jax_numpyro import make_forward_fn, precompute_shift_and_starts


def build_forward(sc_size=10, cut=10, tr=1, t_end=30000):
    weights = np.loadtxt(os.path.join(utils.DATA_ROOT, "weights.txt"))[:sc_size, :sc_size]
    nn = len(weights)
    SC = jnp.array(weights) / jnp.max(weights)
    params = {
        "G": 0.2, "weights": SC, "t_end": t_end, "dt": 0.01,
        "eta": jnp.array([-4.6]), "rv_decimate": 10, "noise_amp": 0.037,
        "tr": 300.0, "seed": 42, "clip_mode": "hard",
    }
    T = (t_end - cut) // tr
    _shift, starts = precompute_shift_and_starts(T, wwidth=30, olap=0.94, stride1=True)
    return make_forward_fn(
        params, "FC", cut, tr, starts, nn, grad_horizon=100, fast_bold=True,
        eps=0.0, grad_method="fd", fd_h=1e-2, fisher_z=True, keep_negative=True)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.join(here, "..", "..", "results", date.today().isoformat())
    ap.add_argument("--outdir", default=default_out)
    ap.add_argument("--sc_size", type=int, default=10)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 64, 256, 1024])
    ap.add_argument("--timeout_per_batch", type=float, default=200.0,
                    help="skip larger batches once a single rep exceeds this (protects CPU).")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    forward_fn = build_forward(sc_size=args.sc_size)
    platform = jax.devices()[0].platform
    device = str(jax.devices()[0])
    print(f"platform={platform}  device={device}  sc_size={args.sc_size}", flush=True)

    vfwd = jax.jit(jax.vmap(forward_fn))

    def bench(batch):
        G = jnp.linspace(0.15, 0.5, batch)
        r = vfwd(G); r.block_until_ready()          # compile (excluded)
        ts = []
        for _ in range(args.reps):
            t = time.time(); r = vfwd(G); r.block_until_ready(); ts.append(time.time() - t)
        return min(ts)

    rows = []
    print(f"{'batch':>6} {'time_s':>8} {'per_eval_ms':>12} {'evals_per_s':>12}", flush=True)
    for b in args.batches:
        try:
            dt = bench(b)
        except Exception as e:
            print(f"{b:6d}  FAILED: {type(e).__name__}: {e}", flush=True)
            break
        print(f"{b:6d} {dt:8.3f} {dt/b*1000:12.1f} {b/dt:12.1f}", flush=True)
        rows.append((platform, device, args.sc_size, b, dt, dt / b * 1000.0, b / dt))
        if dt > args.timeout_per_batch:
            print(f"  (rep time {dt:.1f}s > {args.timeout_per_batch}s; stopping to protect runtime)",
                  flush=True)
            break

    csv = os.path.join(args.outdir, f"forward_throughput_{platform}.csv")
    with open(csv, "w") as f:
        f.write("platform,device,sc_size,batch,time_s,per_eval_ms,evals_per_s\n")
        for r in rows:
            f.write("{},{},{},{},{:.6f},{:.4f},{:.4f}\n".format(*r))
    print(f"\nwrote {csv}", flush=True)


if __name__ == "__main__":
    main()
