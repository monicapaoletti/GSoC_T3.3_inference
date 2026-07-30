# Updated benchmark script with GPU-aware JAX batching and stateless Numba runs

import time
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import os

import mpr  # Numba version
import mpr_jax
import utils

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

from concurrent.futures import ProcessPoolExecutor

def run_single_simulation(params, G, seed):
    # fresh instance per worker (safe)
    local_sde = mpr.MPR_sde(params)
    return local_sde.run({"G": G, "seed": seed})


# -----------------------------
# JAX functions
# -----------------------------
def create_sde_jax(params):
    return mpr_jax.MPR_sde.create(params)


def build_batched_run(sde):
    """jit(vmap) over a batch of (seed, G) -> one full simulation each.

    CRITICAL: return a scalar REDUCTION of the actual simulation output (sum of the
    BOLD trace). Returning a constant (e.g. 0) lets XLA dead-code-eliminate the whole
    scan, so the timing measures nothing. Same trap already fixed in
    sim_batch_throughput.py (commit 1a95403).

    Built ONCE and reused: defining the jitted function inside the batch loop creates a
    fresh function object per batch, so jax.jit re-enters its compile path every time and
    the "timing" becomes compile/dispatch bookkeeping instead of simulation work.
    """
    @jax.jit
    def batched(seeds, gs):
        def single(seed, g):
            out = sde.run({"G": g, "seed": seed})
            return jnp.nansum(out["bold_d"])
        return jax.vmap(single)(seeds, gs)
    return batched


def benchmark_sde_run_vmap_chunked(run, sde, N, batch_size=200, warmed=None):
    """Run the model N times via jit(vmap) in batches of `batch_size`; return seconds.

    Compilation is excluded (each distinct batch shape is warmed up untimed first) and
    every timed call is wrapped in jax.block_until_ready, so the result reflects real
    device work rather than JAX's asynchronous dispatch returning a future immediately.
    """
    warmed = warmed if warmed is not None else set()
    seeds = jnp.arange(sde.P.seed, sde.P.seed + N, dtype=jnp.int32)
    g_values = jnp.linspace(0.0, 1.0, N)

    elapsed = 0.0
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        seeds_batch = seeds[start:end]
        g_batch = g_values[start:end]

        n = end - start
        if n not in warmed:                       # compile once per batch shape, untimed
            jax.block_until_ready(run(seeds_batch, g_batch))
            warmed.add(n)

        start_time = time.time()
        jax.block_until_ready(run(seeds_batch, g_batch))
        elapsed += time.time() - start_time
    return elapsed

# -----------------------------
# Numba functions (stateless safe calls)
# -----------------------------
def create_sde_numba(params):
    return mpr.MPR_sde(params)

def benchmark_sde_run_loop(sde_template, params, N):
    """
    Faster Numba benchmark: reuse one compiled SDE and avoid reconstructing
    Python objects each loop. We mutate only the 'seed' and G.
    """
    g_values = np.linspace(0.0, 1.0, N)
    start = time.time()

    # reuse the object
    sde = sde_template

    for i in range(N):
        seed = params["seed"] + i
        sde.P.G = g_values[i]
        sde.P.seed = seed
        sde.P.t_end = params['t_end']
        sde.run({"G": g_values[i], "seed": seed})

    return time.time() - start

def run_single_simulation_star(args):
    params, G, seed = args
    return run_single_simulation(params, G, seed)


def benchmark_sde_run_parallel(params, N, max_workers=None):
    G_values = np.linspace(0.0, 1.0, N)
    seeds = params["seed"] + np.arange(N)

    start = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        results = list(
            pool.map(
                run_single_simulation_star,
                [(params, G_values[i], seeds[i]) for i in range(N)]
            )
        )

    return time.time() - start


# -----------------------------
# Plotting
# -----------------------------
def plot_benchmark(Ns, times_jax, times_numba, resultspath, cpugpu):
    Ns = np.array(Ns, dtype=int)
    times_jax = np.array(times_jax, dtype=float)
    times_numba = np.array(times_numba, dtype=float)

    Ns_log = np.log10(Ns)

    plt.figure(figsize=(8, 6))
    plt.plot(Ns_log, times_jax, marker='o', label="JAX")
    plt.plot(Ns_log, times_numba, marker='s', label="Numba")

    plt.xlabel(r"Number of simulations ($10^n$)")
    plt.ylabel("Time (s)")
    plt.yscale("log")
    plt.title(f"Benchmarking JAX vs Numba MPR model, tested on {cpugpu}")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.6)

    filename = f"benchmarking_{cpugpu}_tend{os.environ.get('SIM_T_END','1000')}.png"
    np.savez(os.path.join(resultspath, filename + '.npz'),
             Ns=Ns, times_jax=times_jax, times_numba=times_numba)
    plt.savefig(os.path.join(resultspath, filename), dpi=300)
    plt.close()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    t_end = int(os.environ.get("SIM_T_END", 1000))

    print("JAX devices:", jax.devices())
    device_type = "GPU" if any(d.platform == "gpu" for d in jax.devices()) else "CPU"
    print(device_type)

    # -----------------------------
    # JAX setup
    # -----------------------------
    weights_jax = jnp.array(np.loadtxt(utils.DATA_ROOT + "/weights.txt"))
    params_jax = {
        "G": 0.33,
        "t_end": t_end,
        "weights": weights_jax / jnp.max(weights_jax),
        "dt": 0.01,
        "eta": jnp.array([-4.6]),
        "rv_decimate": 10,
        "noise_amp": 0.037,
        "tr": 300.0,
        "seed": 42,
    }
    sde_jax = create_sde_jax(params_jax)

    # Warmup JAX
    sde_jax.run({"seed": params_jax["seed"]})
    print("JAX warmup done!")

    # -----------------------------
    # Numba setup (stateless runs)
    # -----------------------------
    weights_np = np.loadtxt(utils.DATA_ROOT + "/weights.txt")
    params_numba = {
        "G": 0.33,
        "t_end": t_end,
        "weights": weights_np / np.max(weights_np),
        "dt": 0.01,
        "eta": np.array([-4.6]),
        "rv_decimate": 10,
        "noise_amp": 0.037,
        "tr": 300.0,
        "seed": 42,
    }

    # Reuse a previous Numba arm instead of re-running it (it dominates the runtime and
    # is unaffected by the JAX timing fix, so its old numbers are still valid).
    #   SIM_REUSE_NUMBA=/path/to/benchmarking_CPU_tend30000.png.npz
    # The stored Ns must match the Ns being run, otherwise the two curves would be
    # plotted against different x values.
    reuse_numba = os.environ.get("SIM_REUSE_NUMBA", "")

    if not reuse_numba:
        # Warmup Numba
        sde_numba = create_sde_numba(params_numba)
        sde_numba.run({"seed": params_numba["seed"]})
        print("Numba warmup done!")
    else:
        print(f"Reusing Numba timings from {reuse_numba} (skipping Numba benchmark)")

    # -----------------------------
    # Benchmark
    # -----------------------------
    Ns = [int(x) for x in os.environ.get("SIM_NS", "10,100,1000,10000,100000").split(",")]
    # vmap width per chunk. 200 under-feeds the GPU: the t_end=1000 batch sweep
    # (sim_batch_throughput) showed GPU wall time flat out to batch 1024 and only
    # saturating at 4096, so a wider chunk is close to free on device. Keep it
    # IDENTICAL between the CPU and GPU runs or the two curves aren't comparable.
    batch_size = int(os.environ.get("SIM_BATCH_SIZE", 200))

    times_jax = []
    times_numba = []

    run_jax = build_batched_run(sde_jax)   # build ONCE, reuse across every N and batch
    warmed = set()                         # batch shapes already compiled (untimed)
    for N in Ns:
        elapsed_jax = benchmark_sde_run_vmap_chunked(run_jax, sde_jax, N,
                                                     batch_size=batch_size, warmed=warmed)
        times_jax.append(elapsed_jax)
        print(f"JAX repeated {N} times: {elapsed_jax:.4f}s")

    #for N in Ns:
    #    elapsed_numba = benchmark_sde_run_loop(sde_numba, params_numba, N)
    #    times_numba.append(elapsed_numba)
    #    print(f"Numba repeated {N} times: {elapsed_numba:.4f}s")

    if reuse_numba:
        prev = np.load(reuse_numba)
        prev_Ns = list(np.asarray(prev["Ns"]).astype(int))
        if prev_Ns != list(Ns):
            raise SystemExit(
                f"SIM_REUSE_NUMBA Ns mismatch: stored {prev_Ns} != requested {list(Ns)}.\n"
                f"Re-run with SIM_NS={','.join(map(str, prev_Ns))}, or drop SIM_REUSE_NUMBA "
                f"to re-measure Numba."
            )
        times_numba = list(np.asarray(prev["times_numba"], dtype=float))
        for N, t in zip(Ns, times_numba):
            print(f"Numba (reused) repeated {N} times: {t:.4f}s")
    else:
        for N in Ns:
            elapsed_numba = benchmark_sde_run_parallel(params_numba, N, max_workers=os.cpu_count())
            times_numba.append(elapsed_numba)
            print(f"Numba parallel repeated {N} times: {elapsed_numba:.4f}s")


    # -----------------------------
    # Plot
    # -----------------------------
    resultspath = utils.results_folder()
    plot_benchmark(Ns, times_jax, times_numba, resultspath, device_type)
