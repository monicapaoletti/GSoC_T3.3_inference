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

def benchmark_sde_run_vmap_chunked(sde, N, batch_size=200):
    """
    Run sde.run({}) N times using vmap in batches.
    This version is optimized for both CPU and GPU.
    """
    seeds = jnp.arange(sde.P.seed, sde.P.seed + N, dtype=jnp.int32)
    g_values = jnp.linspace(0.0, 1.0, N)

    elapsed = 0.0
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        seeds_batch = seeds[start:end]
        g_batch = g_values[start:end]

        @jax.jit
        def vmap_run(seeds_local, g_local):
            def single_run(seed, g):
                sde.run({"G": g, "seed": seed})
                return 0
            return jax.vmap(single_run)(seeds_local, g_local)

        start_time = time.time()
        vmap_run(seeds_batch, g_batch)
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

    filename = f'benchmarking_{cpugpu}.png'
    np.savez(os.path.join(resultspath, filename + '.npz'),
             Ns=Ns, times_jax=times_jax, times_numba=times_numba)
    plt.savefig(os.path.join(resultspath, filename), dpi=300)
    plt.close()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    t_end = 1_000

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

    # Warmup Numba
    sde_numba = create_sde_numba(params_numba)
    sde_numba.run({"seed": params_numba["seed"]})
    print("Numba warmup done!")

    # -----------------------------
    # Benchmark
    # -----------------------------
    Ns = [10, 100, 1000, 10000, 100000]
    batch_size = 200  # optimized for GPU batching

    times_jax = []
    times_numba = []

    for N in Ns:
        elapsed_jax = benchmark_sde_run_vmap_chunked(sde_jax, N, batch_size=batch_size)
        times_jax.append(elapsed_jax)
        print(f"JAX repeated {N} times: {elapsed_jax:.4f}s")

    #for N in Ns:
    #    elapsed_numba = benchmark_sde_run_loop(sde_numba, params_numba, N)
    #    times_numba.append(elapsed_numba)
    #    print(f"Numba repeated {N} times: {elapsed_numba:.4f}s")

    for N in Ns:
        elapsed_numba = benchmark_sde_run_parallel(params_numba, N, max_workers=os.cpu_count())
        times_numba.append(elapsed_numba)
        print(f"Numba parallel repeated {N} times: {elapsed_numba:.4f}s")


    # -----------------------------
    # Plot
    # -----------------------------
    resultspath = utils.results_folder()
    plot_benchmark(Ns, times_jax, times_numba, resultspath, device_type)
