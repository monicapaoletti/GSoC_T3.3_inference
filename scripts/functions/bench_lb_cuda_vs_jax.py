#!/usr/bin/env python
"""
Larter-Breakspear: van der Vlag's CUDA kernel (tvbSBI/tvbgpu/models/larter_breakspear2.c)
against our JAX port (lb_jax.py), on the same GPU. Two questions:

  1. EQUIVALENCE -- do they compute the same thing?
  2. SPEED       -- which is faster, for one simulation and for a batch?

WHY "SAME OUTPUT" NEEDS CONTROLLING. Out of the box the two can never agree trajectory by
trajectory: the kernel draws its noise from curand seeded with clock64() (not even
reproducible run to run), draws its own initial state, and leaves history rows != 0 as
whatever the buffer holds. So the equivalence tests run a PATCHED COPY of the kernel
(built in memory here; the .c file is never modified) that reads its noise from a buffer,
and both codes start from one shared initial state:

  T1 deterministic  nsig=0, shared initial state  -> pure equations + integrator
  T2 shared noise   nsig>0, the SAME noise array fed to both
  T3 statistics     both unmodified, own RNG, long runs -> mean/sd/FC/spectrum agreement

Each of T1/T2 runs lb_jax twice: AS IS, and with its coupling replaced (here, by
monkeypatch -- lb_jax.py is not edited) by the kernel's: lb_jax couples QV(V_j), the
kernel couples raw V_j. If only the patched variant matches, that line is the difference.

In float32 a chaotic system diverges eventually even when both codes implement identical
equations, because the two compilers round differently (fmaf vs separate ops). The
signature of "same equations" is agreement at ~1e-6 for a long stretch before the
divergence, not agreement forever; a wrong equation shows up within a few steps.

The delays use the KERNEL's formula, (int)(L * (1/speed/dt) + 0.5) in float32, for both,
and speed must be > 0: with nh == 1 the kernel updates its state buffer in place (node i
sees the already-updated nodes j < i), which no vectorised code reproduces.

nh = max_delay + 2, NOT + 1. The kernel writes row (t+1) % nh while it reads row
(t-d) % nh; with d = nh-1 those are the SAME row, so a connection at the longest delay
reads the value node j was given earlier in this very step instead of its delayed value.
Verified on a NumPy line-by-line port of the kernel: at nh = max+1 the kernel-coupling
variant differs by ~2e-2 from the first steps; at max+2 it agrees to ~1e-5 for the whole
run. Whether the EBRAINS runs were affected depends on the nh their driver passes.

Weights are max-normalised by default (--weights_norm). The raw 88-node matrix holds
streamline counts (max ~6e3, row sums ~1e4), and with the kernel's linear-in-V coupling
that pins V at the +-20 clamp. Equivalence still holds there, but the dynamics are not
meaningful. Match whatever the EBRAINS driver feeds the kernel.

Needs cupy (pip install --no-deps cupy-cuda12x fastrlock) and nvcc on PATH
(module load cuda/12.2). Run from scripts/functions/ (imports mpr_jax for lb_jax).
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

# JAX would otherwise grab 75% of the card at import and starve the CuPy side.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

NOISE_OLD = "sqrtf(2.0f*theta*dt)*sigma_eta*curand_normal(&crndst)"
NOISE_NEW = "sqrtf(2.0f*theta*dt)*sigma_eta*noise_pwi[((t - t0) * n_node + i_node) * size + id]"
SIG_OLD = "float * __restrict__ tavg_pwi)"
SIG_NEW = ("float * __restrict__ tavg_pwi,\n"
           "        const float * __restrict__ noise_pwi, unsigned int t0)")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kernel", required=True, help="path to larter_breakspear2.c")
    ap.add_argument("--lb_jax_dir", default=HERE,
                    help="directory holding the lb_jax.py to test (default: next to this script)")
    ap.add_argument("--conn_dir", default=os.path.join(REPO, "data", "connectivity_88"))
    ap.add_argument("--weights_norm", choices=["max", "none"], default="max",
                    help="max: W / W.max() (TVB-style); none: raw streamline counts")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--speed", type=float, default=5.0, help="conduction speed; must be > 0")
    ap.add_argument("--G", type=float, default=0.5)
    ap.add_argument("--nsig", type=float, default=0.03)
    ap.add_argument("--rNMDA", type=float, default=0.25)
    ap.add_argument("--tau_K", type=float, default=1.0)
    ap.add_argument("--phi", type=float, default=0.7)
    ap.add_argument("--d_V", type=float, default=0.65)
    ap.add_argument("--chunk", type=int, default=10,
                    help="steps per kernel launch = the tavg averaging window")
    ap.add_argument("--eq_steps", type=int, default=20000, help="T1/T2 length (steps)")
    ap.add_argument("--stat_steps", type=int, default=200000, help="T3 length (steps)")
    ap.add_argument("--stat_reps", type=int, default=8, help="T3 independent runs per code")
    ap.add_argument("--time_steps", type=int, default=10000, help="timing length (steps)")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 16, 256, 1024, 4096])
    ap.add_argument("--jax_cpu_batches", type=int, nargs="*", default=[1, 16],
                    help="also time JAX on the CPU at these batch sizes")
    ap.add_argument("--skip", nargs="*", default=[], choices=["T1", "T2", "T3", "timing"])
    ap.add_argument("--out", default="lb_bench_results")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.speed <= 0:
        ap.error("--speed must be > 0 (see module docstring: nh == 1 is not comparable)")
    for n in (a.eq_steps, a.stat_steps, a.time_steps):
        if n % a.chunk:
            ap.error(f"step counts must be multiples of --chunk ({a.chunk})")
    return a


# --------------------------------------------------------------------------- setup

def load_connectome(conn_dir, norm="none"):
    W = np.loadtxt(os.path.join(conn_dir, "weights.txt"))
    if norm == "max":
        W = W / W.max()
    W = W.astype(np.float32)
    L = np.loadtxt(os.path.join(conn_dir, "tract_lengths.txt")).astype(np.float32)
    assert W.shape == L.shape and W.shape[0] == W.shape[1], (W.shape, L.shape)
    return W, L


def kernel_delays(L, speed, dt):
    """The kernel's own rounding, in float32: (int)(lengths*rec_speed_dt + 0.5f)."""
    rec = np.float32(1.0) / np.float32(speed) / np.float32(dt)
    d = (L * rec + np.float32(0.5)).astype(np.int32)          # positive -> trunc == floor
    return d, int(d.max()) + 2   # +2, not +1: see the nh note in the module docstring


def shared_init(rng, nn, B):
    V = (0.001 * rng.standard_normal((B, nn))).astype(np.float32)
    W = (0.001 * rng.standard_normal((B, nn))).astype(np.float32)
    Z = (0.001 * rng.standard_normal((B, nn))).astype(np.float32)
    eta = np.zeros((B, nn), np.float32)
    return V, W, Z, eta


# --------------------------------------------------------------------------- CUDA side

class CudaLB:
    def __init__(self, kernel_path, W, L, delay_nh, dt, speed):
        import cupy as cp
        self.cp = cp
        src = open(kernel_path).read()
        for old in (NOISE_OLD, SIG_OLD):
            if src.count(old) != 1:
                raise RuntimeError(f"kernel source changed: expected exactly one {old!r}")
        # The kernels are not extern "C", so their names are C++-mangled and could not be
        # looked up by name. Add the linkage in these in-memory copies only; it changes
        # nothing about the generated code.
        for k in ("larter_breakspear", "bold_update"):
            src = src.replace(f"__global__ void {k}(", f'extern "C" __global__ void {k}(')
        cc = cp.cuda.Device().compute_capability
        opts = (f"-arch=sm_{cc}",)
        t0 = time.perf_counter()
        self.f_orig = cp.RawModule(code=src, backend="nvcc", options=opts) \
            .get_function("larter_breakspear")
        patched = src.replace(SIG_OLD, SIG_NEW).replace(NOISE_OLD, NOISE_NEW)
        self.f_noise = cp.RawModule(code=patched, backend="nvcc", options=opts) \
            .get_function("larter_breakspear")
        self.compile_s = time.perf_counter() - t0          # both modules, nvcc included
        self.nn = W.shape[0]
        self.nh = delay_nh
        self.dt, self.speed = dt, speed
        self.W = cp.asarray(W.ravel())                     # [i*n + j], i = target node
        self.L = cp.asarray(L.ravel())

    def params(self, B, a, G=None):
        G = np.full(B, a.G, np.float32) if G is None else np.asarray(G, np.float32)
        p = np.stack([G, np.full(B, a.nsig), np.full(B, a.rNMDA), np.full(B, a.speed),
                      np.full(B, a.tau_K), np.full(B, a.phi), np.full(B, a.d_V)])
        return self.cp.asarray(p.astype(np.float32))       # (7, B): params(i) = p[i*B + id]

    def run(self, params, n_steps, chunk, init=None, noise=None, sync=True):
        """Returns tavg per launch, shape (n_steps//chunk, nn, B), on the device.

        init=None  -> kernel draws its own initial state (i_step starts at 0).
        init given -> every history row holds it, and i_step starts at nh (a non-zero
                      multiple of nh), so the kernel's own init block is skipped and step
                      t reads row t % nh = 0 first, exactly like lb_jax's buffer.
        noise      -> (n_steps, nn, B) float32, routed to the patched kernel.
        """
        cp = self.cp
        nn, nh, B = self.nn, self.nh, params.shape[1]
        state = cp.zeros((nh, 4, nn, B), np.float32)
        if init is not None:
            for k, x in enumerate(init):                    # V, W, Z, eta -> (nn, B)
                state[:, k] = cp.asarray(x.T)[None]
            i0 = nh
        else:
            i0 = 0
        tavg = cp.zeros((nn, B), np.float32)
        out = cp.empty((n_steps // chunk, nn, B), np.float32)
        threads = 32
        grid = ((B + threads - 1) // threads, 1)
        noise_d = cp.asarray(noise) if noise is not None else None
        for c in range(n_steps // chunk):
            args = [np.uint32(i0 + c * chunk), np.uint32(nn), np.uint32(nh), np.uint32(chunk),
                    np.uint32(B), np.float32(self.dt), np.float32(self.speed), np.int32(0),
                    self.W, self.L, params, state, tavg]
            if noise_d is None:
                self.f_orig(grid, (threads, 1, 1), tuple(args))
            else:
                self.f_noise(grid, (threads, 1, 1), tuple(args + [noise_d, np.uint32(i0)]))
            out[c] = tavg
        if sync:
            cp.cuda.Device().synchronize()
        return out


# --------------------------------------------------------------------------- JAX side

def make_jax(lb, P, chunk, n_chunks, coupling, rng_mode):
    """jit(vmap(sim)) over a batch. Mirrors the kernel's output: the mean of V over each
    `chunk`-step window. rng_mode: 'given' (noise array in) or 'key' (drawn in the scan,
    like the kernel's in-loop curand -- needed at large batch, where a pre-drawn
    (steps, batch, nodes) array would not fit)."""
    import jax
    import jax.numpy as jnp

    orig = lb._coupling
    if coupling == "kernel":
        def _coupling_kernel(V_hist, t_idx, P_):
            read_idx = jnp.mod(t_idx - P_.delay_steps, P_.nh)
            cols = jnp.broadcast_to(jnp.arange(P_.nn), read_idx.shape)
            return P_.G * jnp.sum(P_.weights * V_hist[read_idx, cols], axis=1)
        lb._coupling = _coupling_kernel

    def sim(G, V0, W0, Z0, eta0, noise_or_key):
        Pg = P.replace(G=G)
        Vh = jnp.tile(V0[None, :], (Pg.nh, 1))
        carry0 = (V0, W0, Z0, eta0, Vh, jnp.array(0, jnp.int32))

        def window(state, xs):
            lbc, key = state
            def step(s, z):
                c, acc, k = s
                if rng_mode == "key":
                    k, sub = jax.random.split(k)
                    z = jax.random.normal(sub, (Pg.nn,))
                c, V = lb.lb_step(c, z, Pg)
                return (c, acc + V / chunk, k), None
            init = (lbc, jnp.zeros(Pg.nn, jnp.float32), key)
            (lbc, acc, key), _ = jax.lax.scan(step, init, xs, length=chunk)
            return (lbc, key), acc

        if rng_mode == "key":
            state0, xs = (carry0, noise_or_key), None
        else:
            state0, xs = (carry0, jax.random.PRNGKey(0)), noise_or_key   # (n_chunks, chunk, nn)
        _, tav = jax.lax.scan(window, state0, xs, length=n_chunks)
        return tav                                                       # (n_chunks, nn)

    f = jax.jit(jax.vmap(sim))
    return f, (lambda: setattr(lb, "_coupling", orig))


def jax_run(f, restore, *args):
    import jax
    try:
        t0 = time.perf_counter()
        out = jax.block_until_ready(f(*args))
        first = time.perf_counter() - t0
        t0 = time.perf_counter()
        out = jax.block_until_ready(f(*args))
        run = time.perf_counter() - t0
    finally:
        restore()
    return np.asarray(out), first - run, run         # compile ~= first call - steady call


# --------------------------------------------------------------------------- tests

def compare_traces(tj, tk, a):
    """tj, tk: (B, n_chunks, nn). Max |diff| over nodes, per window."""
    d = np.abs(tj - tk).max(axis=(0, 2))
    scale = float(np.std(tk))
    over = np.nonzero(d > 1e-3 * max(scale, 1e-12))[0]
    first = int(over[0]) if over.size else None
    ms_per_window = a.chunk * a.dt
    return {
        "signal_sd": scale,
        "max_abs_diff_first_10_windows": float(d[:10].max()),
        "max_abs_diff_first_1pct": float(d[: max(1, len(d) // 100)].max()),
        "max_abs_diff_overall": float(d.max()),
        "first_window_diff_gt_1e-3_sd": first,
        "agree_until_ms": (first * ms_per_window if first is not None
                           else len(d) * ms_per_window),
        "run_ms": len(d) * ms_per_window,
    }


def equivalence(test, lb, cuda, P_noisy, P_quiet, a, nn, results):
    import jax.numpy as jnp
    rng = np.random.default_rng(a.seed)
    B = 2
    Gs = np.array([a.G, 0.5 * a.G], np.float32)        # two work items, two parameter values
    V, W, Z, eta = shared_init(rng, nn, B)
    n_chunks = a.eq_steps // a.chunk
    noise = rng.standard_normal((B, a.eq_steps, nn)).astype(np.float32)
    if test == "T1":
        P, nsig = P_quiet, 0.0
    else:
        P, nsig = P_noisy, a.nsig

    a_k = argparse.Namespace(**{**vars(a), "nsig": nsig})
    tk = cuda.run(cuda.params(B, a_k, Gs), a.eq_steps, a.chunk,
                  init=(V, W, Z, eta), noise=noise.transpose(1, 2, 0).copy())
    tk = np.asarray(tk.get()).transpose(2, 0, 1)                        # (B, chunks, nn)

    res = {}
    for coupling in ("as_is", "kernel"):
        f, restore = make_jax(lb, P, a.chunk, n_chunks, coupling, "given")
        tj, _, _ = jax_run(f, restore, jnp.asarray(Gs), V, W, Z, eta,
                           jnp.asarray(noise.reshape(B, n_chunks, a.chunk, nn)))
        res[f"lb_jax_{coupling}_coupling"] = compare_traces(tj, tk, a)
        np.savez_compressed(os.path.join(a.out, f"{test}_{coupling}.npz"), jax=tj, cuda=tk)
    results[test] = res


def statistics(lb, cuda, P, a, nn, results):
    import jax
    import jax.numpy as jnp
    B, n_chunks = a.stat_reps, a.stat_steps // a.chunk
    burn = n_chunks // 10
    tk = cuda.run(cuda.params(B, a), a.stat_steps, a.chunk)           # own curand + own init
    tk = np.asarray(tk.get()).transpose(2, 0, 1)[:, burn:]
    rng = np.random.default_rng(a.seed + 1)
    V, W, Z, eta = shared_init(rng, nn, B)
    keys = jax.random.split(jax.random.PRNGKey(a.seed + 1), B)
    out = {}
    for coupling in ("as_is", "kernel"):
        f, restore = make_jax(lb, P, a.chunk, n_chunks, coupling, "key")
        tj, _, _ = jax_run(f, restore, jnp.full(B, a.G, jnp.float32), V, W, Z, eta, keys)
        tj = tj[:, burn:]
        out[f"lb_jax_{coupling}_coupling"] = summary_stats(tj, tk, a)
    results["T3"] = out


def summary_stats(tj, tk, a):
    def fc(x):  # mean FC over replicates
        return np.mean([np.corrcoef(r.T) for r in x], axis=0)
    def peak_hz(x):
        sig = x.mean(axis=2) - x.mean(axis=(1, 2))[:, None]
        ps = np.mean(np.abs(np.fft.rfft(sig, axis=1)) ** 2, axis=0)
        fr = np.fft.rfftfreq(sig.shape[1], d=a.chunk * a.dt / 1000.0)  # dt assumed in ms
        return float(fr[1:][np.argmax(ps[1:])])
    iu = np.triu_indices(tk.shape[2], 1)
    fj, fk = fc(tj), fc(tk)
    return {
        "mean_V": [float(tj.mean()), float(tk.mean())],
        "sd_V": [float(tj.std()), float(tk.std())],
        "node_mean_corr": float(np.corrcoef(tj.mean(axis=(0, 1)), tk.mean(axis=(0, 1)))[0, 1]),
        "FC_corr": float(np.corrcoef(np.nan_to_num(fj[iu]), np.nan_to_num(fk[iu]))[0, 1]),
        "FC_mean": [float(np.nanmean(fj[iu])), float(np.nanmean(fk[iu]))],
        "peak_freq_hz_if_dt_in_ms": [peak_hz(tj), peak_hz(tk)],
        "order": "[jax, cuda]",
    }


def timing(lb, cuda, P, a, nn, results):
    import jax
    import jax.numpy as jnp
    n_chunks = a.time_steps // a.chunk
    rows = []
    rng = np.random.default_rng(a.seed + 2)
    for B in a.batches:
        V, W, Z, eta = shared_init(rng, nn, B)
        p = cuda.params(B, a)
        cuda.run(p, a.chunk, a.chunk)                                  # warm-up launch
        t0 = time.perf_counter()
        cuda.run(p, a.time_steps, a.chunk)
        tc = time.perf_counter() - t0
        keys = jax.random.split(jax.random.PRNGKey(0), B)
        f, restore = make_jax(lb, P, a.chunk, n_chunks, "kernel", "key")
        _, comp, tj = jax_run(f, restore, jnp.full(B, a.G, jnp.float32), V, W, Z, eta, keys)
        row = {"batch": B, "cuda_s": tc, "jax_gpu_s": tj, "jax_gpu_compile_s": comp}
        if B in a.jax_cpu_batches:
            try:
                cpu = jax.devices("cpu")[0]
                with jax.default_device(cpu):
                    f, restore = make_jax(lb, P, a.chunk, n_chunks, "kernel", "key")
                    args = [jax.device_put(x, cpu) for x in
                            (jnp.full(B, a.G, jnp.float32), V, W, Z, eta, keys)]
                    _, compc, tcpu = jax_run(f, restore, *args)
                row.update(jax_cpu_s=tcpu, jax_cpu_compile_s=compc)
            except Exception as e:                   # a CPU-timing failure must not sink the GPU rows
                row["jax_cpu_error"] = repr(e)[:300]
        for k in ("cuda_s", "jax_gpu_s", "jax_cpu_s"):
            if k in row:
                row[k.replace("_s", "_steps_per_s")] = B * a.time_steps / row[k]
        rows.append(row)
        print(f"[timing] B={B:5d}  cuda {tc:8.2f}s  jax-gpu {tj:8.2f}s"
              + (f"  jax-cpu {row['jax_cpu_s']:8.2f}s" if "jax_cpu_s" in row else ""), flush=True)
    results["timing"] = {"steps": a.time_steps, "cuda_compile_s_both_modules": cuda.compile_s,
                         "rows": rows}


# --------------------------------------------------------------------------- main

def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, HERE)                      # mpr_jax
    sys.path.insert(0, os.path.abspath(a.lb_jax_dir))
    import jax
    import cupy as cp
    import lb_jax as lb

    W, L = load_connectome(a.conn_dir, a.weights_norm)
    nn = W.shape[0]
    delays, nh = kernel_delays(L, a.speed, a.dt)
    if nh < 2:
        raise SystemExit("all delays are 0 steps (nh == 1): lower --speed or --dt")

    def mkP(nsig):
        P = lb.ParLB.create(weights=W, lengths=L, speed=a.speed, dt=a.dt, nsig=nsig, G=a.G,
                            rNMDA=a.rNMDA, tau_K=a.tau_K, phi=a.phi, d_V=a.d_V)
        return P.replace(delay_steps=jax.numpy.asarray(delays), nh=nh)
    P_noisy, P_quiet = mkP(a.nsig), mkP(0.0)

    cuda = CudaLB(a.kernel, W, L, nh, a.dt, a.speed)
    results = {
        "env": {"gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                "jax": jax.__version__, "cupy": cp.__version__, "python": platform.python_version(),
                "lb_jax": os.path.abspath(lb.__file__), "kernel": os.path.abspath(a.kernel)},
        "config": {**vars(a), "nn": nn, "nh": nh, "max_delay_steps": int(delays.max())},
    }
    print(f"[setup] nn={nn} nh={nh} on {results['env']['gpu']}; nvcc compile "
          f"{cuda.compile_s:.1f}s", flush=True)

    if "T1" not in a.skip:
        equivalence("T1", lb, cuda, P_noisy, P_quiet, a, nn, results)
    if "T2" not in a.skip:
        equivalence("T2", lb, cuda, P_noisy, P_quiet, a, nn, results)
    for t in ("T1", "T2"):
        for k, v in results.get(t, {}).items():
            print(f"[{t}] {k:28s} first-10-windows max|diff| {v['max_abs_diff_first_10_windows']:.2e}"
                  f"  agree until {v['agree_until_ms']:.1f} of {v['run_ms']:.0f} ms", flush=True)
    if "T3" not in a.skip:
        statistics(lb, cuda, P_noisy, a, nn, results)
        for k, v in results["T3"].items():
            print(f"[T3] {k:28s} FC corr {v['FC_corr']:.3f}  mean V jax/cuda "
                  f"{v['mean_V'][0]:.4f}/{v['mean_V'][1]:.4f}", flush=True)
    if "timing" not in a.skip:
        timing(lb, cuda, P_noisy, a, nn, results)

    path = os.path.join(a.out, "lb_bench.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=1, default=str)
    print(f"[done] {path}")


if __name__ == "__main__":
    main()
