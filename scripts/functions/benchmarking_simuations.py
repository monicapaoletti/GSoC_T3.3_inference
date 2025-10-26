"""
run multiple simulations from the mpr model both in JAX and in NUMBA in order to do benchmarking both on CPUs and GPUs
"""
import time
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import os

import mpr # numba version
import mpr_jax
import utils


# JAX functions

def create_sde_jax(params):
    return mpr_jax.MPR_sde.create(params)

def benchmark_sde_run_vmap(sde, N):
    """
    Run sde.run({}) N times in parallel using vmap, return elapsed time.
    Only time is returned; results are discarded.
    """
    seeds = jnp.arange(sde.P.seed, sde.P.seed + N, dtype=jnp.int32)
    g_values = jnp.arange(0,N)/N

    def single_run(seed,g):
        # Pass seed as JAX scalar, not Python int
        data = sde.run({"G": g, "seed": seed})
        return 0  # dummy output to satisfy vmap

    vmap_run = jax.vmap(single_run)

    start = time.time()
    vmap_run(seeds, g_values)
    elapsed = time.time() - start

    return elapsed



# NUMBA functions

def create_sde_numba(params):
    return mpr.MPR_sde(params)

def benchmark_sde_run_loop(sde, N):
    """
    Run sde.run({}) N times sequentially for Numba version.
    Only time is returned; results are discarded.
    """
    g_values = np.arange(0,N)/N
    start = time.time()
    for i in range(N):
        sde.run({"G": g_values[i],"seed": sde.P.seed + i})
    elapsed = time.time() - start
    return elapsed


# plot function

def plot_benchmark(Ns, times_jax, times_numba, resultspath, cpugpu):
    """
    Plot benchmark results for JAX and Numba.

    Args:
        Ns (list or array): Number of simulations.
        times_jax (list or array): Times for JAX runs.
        times_numba (list or array): Times for Numba runs.
        filename (str): Name of the file to save.
    """
    Ns = np.array(Ns, dtype=int)
    times_jax = np.array(times_jax, dtype=float)
    times_numba = np.array(times_numba, dtype=float)

    # Convert Ns to log10 scale
    Ns_log = np.log10(Ns)

    plt.figure(figsize=(8, 6))
    plt.plot(Ns_log, times_jax, marker='o', label="JAX")
    plt.plot(Ns_log, times_numba, marker='s', label="Numba")

    plt.xlabel(r"Number of simulations ($10^n$)")
    plt.ylabel("Time (s)")
    plt.yscale("log")
    plt.title(f"Benchmarking  simulation cost of JAX vs Numba mpr model, tested on {cpugpu}")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.6)

    results_folder = utils.results_folder()
    filename = f'benchmarking_{cpugpu}_100m.png'
    np.savez(os.path.join(resultspath, filename + '.npz'), Ns=Ns, times_jax=times_jax, times_numba=times_numba)
    plt.savefig(os.path.join(resultspath, filename), dpi=300)
    plt.close()



if __name__ == "__main__":

    t_end = 10_000

    print('check for jax.devices() = '+str(jax.devices()))

    if any(device.platform == "gpu" for device in jax.devices()):
        device_type = "GPU"
    else:
        device_type = "CPU"

    print(device_type)

    # JAX simulations

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

    # optional warmup
    sde_jax.run({"seed": params_jax["seed"]})
    print("Warmup done!")
    #elapsed = benchmark_sde_run_vmap(sde_jax, 10)
    #print("Test done!")

    Ns = [10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
    #Ns = [1]

    times_jax = []
    times_numba = []

    for N in Ns:
        elapsed = benchmark_sde_run_vmap(sde_jax, N)
        times_jax.append(elapsed)
        print(f"JAX sde.run() repeated {N} times took {elapsed:.4f} seconds")


    # NUMBA simulations

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

    sde_numba = create_sde_numba(params_numba)

    # optional warmup
    sde_numba.run({"seed": params_numba["seed"]})
    print("Warmup done!")

    for N in Ns:
        elapsed = benchmark_sde_run_loop(sde_numba, N)
        times_numba.append(elapsed)
        print(f"Numba sde.run() repeated {N} times took {elapsed:.4f} seconds")


    resultspath = utils.results_folder()

    plot_benchmark(Ns, times_jax, times_numba, resultspath, device_type)
