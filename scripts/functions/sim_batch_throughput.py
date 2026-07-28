#!/usr/bin/env python3
"""Batch-size sweep for the raw MPR simulation: JAX-CPU vs JAX-GPU throughput.

The standard sim benchmark (benchmarking_simuations.py) caps the vmap batch at
200, which is far too small to fill an L4 GPU, so its CPU and GPU figures look
almost identical. This script instead SWEEPS the batch size (how many
independent simulations run in one jax.jit(jax.vmap(...)) call) and measures
throughput = batch / wall-time. That is where the GPU advantage shows: GPU
throughput stays high/flat as the batch grows (it fills the idle device),
while the CPU peaks early and then collapses.

Run once per backend, then plot_sim_batch_throughput.py overlays them:
    JAX_PLATFORMS=cpu            python3 sim_batch_throughput.py     # -> ..._CPU.csv
    XLA_PYTHON_CLIENT_PREALLOCATE=false python3 sim_batch_throughput.py  # -> ..._GPU.csv

Env knobs: SIM_T_END (default 1000), SIM_BATCHES (comma list, default
"1,16,64,256,1024,4096,16384"), SIM_REPS (timed reps after warmup, default 3).
"""
import os, time, csv
import numpy as np
import jax, jax.numpy as jnp
import mpr_jax
import utils


def build_batched_run(sde):
    """jit(vmap) over a batch of (seed, G) -> one full simulation each.

    CRITICAL: return a scalar REDUCTION of the actual simulation output
    (sum of the BOLD trace). Returning a constant (e.g. 0) lets XLA
    dead-code-eliminate the whole scan, so the timing measures nothing.
    """
    @jax.jit
    def batched(seeds, gs):
        def single(seed, g):
            out = sde.run({"G": g, "seed": seed})
            return jnp.nansum(out["bold_d"])
        return jax.vmap(single)(seeds, gs)
    return batched


def main():
    t_end = int(os.environ.get("SIM_T_END", 1000))
    batches = [int(x) for x in os.environ.get(
        "SIM_BATCHES", "1,16,64,256,1024,4096,16384").split(",")]
    reps = int(os.environ.get("SIM_REPS", 3))

    devs = jax.devices()
    backend = "GPU" if any(d.platform == "gpu" for d in devs) else "CPU"
    print(f"JAX devices: {devs}  -> backend={backend}, t_end={t_end}")

    weights = jnp.array(np.loadtxt(utils.DATA_ROOT + "/weights.txt"))
    params = {"G": 0.33, "t_end": t_end, "weights": weights / jnp.max(weights),
              "dt": 0.01, "eta": jnp.array([-4.6]), "rv_decimate": 10,
              "noise_amp": 0.037, "tr": 300.0, "seed": 42}
    sde = mpr_jax.MPR_sde.create(params)
    run = build_batched_run(sde)

    rows = []
    for B in batches:
        seeds = jnp.arange(42, 42 + B, dtype=jnp.int32)
        gs = jnp.linspace(0.0, 1.0, B)
        try:
            jax.block_until_ready(run(seeds, gs))          # warmup / compile
            best = np.inf
            for _ in range(reps):
                t0 = time.time()
                jax.block_until_ready(run(seeds, gs))
                best = min(best, time.time() - t0)
            thr = B / best
            rows.append((B, best, thr))
            print(f"  batch={B:>6}  time={best:8.4f}s  throughput={thr:9.2f} sims/s")
        except Exception as e:                              # OOM at large batch on GPU
            print(f"  batch={B:>6}  FAILED ({type(e).__name__}: {str(e)[:60]}) -> stopping sweep")
            break

    out = utils.results_folder()
    fn = os.path.join(out, f"sim_batch_throughput_{backend}_tend{t_end}.csv")
    with open(fn, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["batch", "time_s", "throughput_sims_per_s"])
        w.writerows(rows)
    print(f"wrote {fn}")


if __name__ == "__main__":
    main()
