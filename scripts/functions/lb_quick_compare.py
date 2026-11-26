#!/usr/bin/env python
"""
Quick look: ONE Larter-Breakspear simulation from van der Vlag's CUDA kernel and from
lb_jax.py, plotted side by side, to see what each model actually outputs.

Both codes start from the same initial state and, by default, are fed the SAME noise
(through the in-memory patched copy of the kernel from bench_lb_cuda_vs_jax.py; the .c
file is never modified), so any visible difference is a difference in the model, not
in the random numbers. lb_jax is shown twice: as it is (couples QV(V_j)) and with the
kernel's coupling (raw V_j).

Output: one PNG with, per code,
  row 1  window-averaged V of a few nodes vs time
  row 2  all nodes vs time (heat map)
  row 3  FC (correlation of V between nodes)
and a bottom panel with max |JAX - CUDA| over nodes vs time.

Run from scripts/functions/ on a GPU node (needs cupy + nvcc, like the benchmark):
  python lb_quick_compare.py --kernel <larter_breakspear2.c> --lb_jax_dir <dir> \
      --t_ms 2000 --out lb_quick.png
--own_noise lets each code use its own RNG instead (the kernel's curand, unreproducible).
"""
import argparse
import os
import sys

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_lb_cuda_vs_jax as bl  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--lb_jax_dir", default=HERE)
    ap.add_argument("--conn_dir", default=os.path.join(bl.REPO, "data", "connectivity_88"))
    ap.add_argument("--weights_norm", choices=["max", "none"], default="max")
    ap.add_argument("--t_ms", type=float, default=2000.0, help="simulated time (dt in ms)")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--speed", type=float, default=5.0)
    ap.add_argument("--G", type=float, default=0.5)
    ap.add_argument("--nsig", type=float, default=0.03)
    ap.add_argument("--rNMDA", type=float, default=0.25)
    ap.add_argument("--tau_K", type=float, default=1.0)
    ap.add_argument("--phi", type=float, default=0.7)
    ap.add_argument("--d_V", type=float, default=0.65)
    ap.add_argument("--chunk", type=int, default=10, help="steps per averaging window")
    ap.add_argument("--nodes", type=int, nargs="+", default=[0, 10, 44, 70],
                    help="nodes drawn in row 1")
    ap.add_argument("--own_noise", action="store_true",
                    help="each code uses its own RNG (no shared noise, no shared init)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="lb_quick.png")
    a = ap.parse_args()

    sys.path.insert(0, os.path.abspath(a.lb_jax_dir))
    import jax
    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import lb_jax as lb

    W, L = bl.load_connectome(a.conn_dir, a.weights_norm)
    nn = W.shape[0]
    delays, nh = bl.kernel_delays(L, a.speed, a.dt)
    n_steps = int(round(a.t_ms / a.dt)) // a.chunk * a.chunk
    n_chunks = n_steps // a.chunk
    t = (np.arange(n_chunks) + 1) * a.chunk * a.dt                     # ms, window ends

    P = lb.ParLB.create(weights=W, lengths=L, speed=a.speed, dt=a.dt, nsig=a.nsig, G=a.G,
                        rNMDA=a.rNMDA, tau_K=a.tau_K, phi=a.phi, d_V=a.d_V)
    P = P.replace(delay_steps=jnp.asarray(delays), nh=nh)
    cuda = bl.CudaLB(a.kernel, W, L, nh, a.dt, a.speed)

    rng = np.random.default_rng(a.seed)
    V, Wv, Z, eta = bl.shared_init(rng, nn, 1)
    G = jnp.asarray([a.G], jnp.float32)
    runs = {}
    if a.own_noise:
        tk = cuda.run(cuda.params(1, a), n_steps, a.chunk)             # own curand + own init
        key = jax.random.split(jax.random.PRNGKey(a.seed), 1)
        for c in ("as_is", "kernel"):
            f, restore = bl.make_jax(lb, P, a.chunk, n_chunks, c, "key")
            runs[c] = bl.jax_run(f, restore, G, V, Wv, Z, eta, key)[0][0]
    else:
        noise = rng.standard_normal((1, n_steps, nn)).astype(np.float32)
        tk = cuda.run(cuda.params(1, a), n_steps, a.chunk, init=(V, Wv, Z, eta),
                      noise=noise.transpose(1, 2, 0).copy())
        for c in ("as_is", "kernel"):
            f, restore = bl.make_jax(lb, P, a.chunk, n_chunks, c, "given")
            runs[c] = bl.jax_run(f, restore, G, V, Wv, Z, eta,
                                 jnp.asarray(noise.reshape(1, n_chunks, a.chunk, nn)))[0][0]
    tk = np.asarray(tk.get())[:, :, 0]                                  # (chunks, nn)

    panels = [("CUDA kernel (raw V coupling)", tk),
              ("lb_jax as is (QV(V) coupling)", runs["as_is"]),
              ("lb_jax, kernel coupling", runs["kernel"])]
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1.1, 0.8])
    vmin = min(x.min() for _, x in panels)
    vmax = max(x.max() for _, x in panels)
    burn = n_chunks // 10                                               # FC without transient
    for k, (name, x) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        for i in a.nodes:
            ax.plot(t, x[:, i], lw=0.7, label=f"node {i}")
        ax.set(title=name, xlabel="time [ms]", ylabel="V (window mean)", ylim=(vmin, vmax))
        if k == 0:
            ax.legend(fontsize=7, loc="upper right")
        ax = fig.add_subplot(gs[1, k])
        im = ax.imshow(x.T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax,
                       extent=(t[0], t[-1], 0, nn), cmap="viridis")
        ax.set(xlabel="time [ms]", ylabel="node")
        fig.colorbar(im, ax=ax, fraction=0.04)
        ax = fig.add_subplot(gs[2, k])
        fc = np.nan_to_num(np.corrcoef(x[burn:].T))
        im = ax.imshow(fc, vmin=-1, vmax=1, cmap="RdBu_r")
        iu = np.triu_indices(nn, 1)
        ax.set(title=f"FC (mean {fc[iu].mean():.2f})", xlabel="node", ylabel="node")
        fig.colorbar(im, ax=ax, fraction=0.04)
    ax = fig.add_subplot(gs[3, :])
    for c, lab in (("as_is", "lb_jax as is"), ("kernel", "lb_jax, kernel coupling")):
        d = np.abs(runs[c] - tk).max(axis=1)
        ax.semilogy(t, np.maximum(d, 1e-9), lw=0.8, label=lab)
    ax.set(xlabel="time [ms]", ylabel="max over nodes |JAX - CUDA|",
           title="difference from the CUDA kernel"
                 + (" (independent noise: only the statistics should match)" if a.own_noise
                    else " (same initial state and same noise)"))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.suptitle(f"Larter-Breakspear, {nn} nodes, G={a.G}, nsig={a.nsig}, speed={a.speed}, "
                 f"dt={a.dt}, {n_steps * a.dt:.0f} ms, weights {a.weights_norm}-normalised, "
                 f"nh={nh}")
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    np.savez_compressed(os.path.splitext(a.out)[0] + ".npz", t_ms=t, cuda=tk,
                        jax_as_is=runs["as_is"], jax_kernel_coupling=runs["kernel"])
    print(f"[quick] wrote {a.out}")


if __name__ == "__main__":
    main()
