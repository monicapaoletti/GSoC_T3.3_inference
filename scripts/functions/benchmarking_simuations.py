import time
import jax.numpy as jnp
import jax
import numpy as np
import mpr # numba version
import mpr_jax
import utils  # Assuming utils.DATA_ROOT is defined there



# JAX functions

def create_sde_jax(params):
    return mpr_jax.MPR_sde.create(params)

def benchmark_sde_run_vmap(sde, N):
    """
    Run sde.run({}) N times in parallel using vmap, return elapsed time.
    Only time is returned; results are discarded.
    """
    seeds = jnp.arange(sde.P.seed, sde.P.seed + N, dtype=jnp.int32)

    def single_run(seed):
        # Pass seed as JAX scalar, not Python int
        data = sde.run({"seed": seed})
        return 0  # dummy output to satisfy vmap

    vmap_run = jax.vmap(single_run)

    start = time.time()
    vmap_run(seeds)
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
    start = time.time()
    for i in range(N):
        sde.run({"seed": sde.P.seed + i})
    elapsed = time.time() - start
    return elapsed


if __name__ == "__main__":

    t_end = 10_000

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

    Ns = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
    #Ns = [1]
    for N in Ns:
        elapsed = benchmark_sde_run_vmap(sde_jax, N)
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

    Ns = [10, 100, 1000, 10000, 100000, 10000000]
    for N in Ns:
        elapsed = benchmark_sde_run_loop(sde_numba, N)
        print(f"Numba sde.run() repeated {N} times took {elapsed:.4f} seconds")