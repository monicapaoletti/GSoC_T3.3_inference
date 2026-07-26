"""Two-GPU work-queue runner for the doslis GPU batching benchmark.

doslis has NO SLURM, so this is a tiny scheduler: it runs up to 2 GPU jobs at
once (one pinned to each L4 via CUDA_VISIBLE_DEVICES) plus a small CPU lane
(JAX_PLATFORMS=cpu, no GPU contention). Jobs are ordered forward -> smc -> nuts
so the cheap, fast results land first and the multi-hour NUTS cells grind last.

Launch it inside tmux so it survives ssh disconnects, e.g.:
    tmux new-session -d -s bench \
      'source /CNSdata/mpaolett/env.sh && \
       python run_gpu_pool.py --suite forward smc nuts > pool.log 2>&1'

Each GPU job sets XLA_PYTHON_CLIENT_PREALLOCATE=false (only ~5.8 GB free per L4).
Per-job logs go to results/<DATE>/out/<name>.log.
"""
import argparse, os, subprocess, time
from datetime import date

FUNCS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(FUNCS_DIR, "..", ".."))

# shared model/config knobs; small 10-node model for CPU-comparability.
# G and which_stat (FC/FCD) are threaded per-run so the same runner drives both
# the multi-G and the FC-vs-FCD campaigns.
COMMON = ["--grad_method", "fd", "--grad_horizon", "100",
          "--SC_type", "data", "--SC_size", "10", "--fast_bold",
          "--cut", "10", "--tr", "1", "--t_end", "30000", "--seed", "42"]


def results_dir():
    d = os.path.join(REPO, "results", date.today().isoformat())
    os.makedirs(os.path.join(d, "out"), exist_ok=True)
    return d


def forward_jobs(save):
    return [
        ("forward_gpu", "gpu", ["microbench_batch.py", "--outdir", save]),
        ("forward_cpu", "cpu", ["microbench_batch.py", "--outdir", save,
                                "--timeout_per_batch", "60"]),
    ]


def smc_jobs(save, n_stages, n_mcmc, g, stat, gpu_only=False):
    jobs = []
    GS = ["--G", str(g), "--which_stat", stat]
    for flavor in ["smc_lik", "smc_abc"]:
        # GPU: full particle sweep (batched axis -> ~free on GPU)
        for npart in [64, 256, 1024, 4096]:
            jobs.append((f"{flavor}_gpu_np{npart}_G{g}_{stat}", "gpu",
                         ["mpr_jax_numpyro.py", "--sampler", flavor,
                          "--n_particles", str(npart), "--n_stages", str(n_stages),
                          "--n_mcmc", str(n_mcmc), "--save_dir", save] + GS + COMMON))
        # CPU: only the smaller particle counts (large batch collapses on CPU).
        # Skipped for the multi-G/FCD campaigns (--gpu_only): CPU np256 ~5.6h would
        # bottleneck every cell; the CPU baseline is already established at G=0.2 FC.
        if not gpu_only:
            for npart in [64, 256]:
                jobs.append((f"{flavor}_cpu_np{npart}_G{g}_{stat}", "cpu",
                             ["mpr_jax_numpyro.py", "--sampler", flavor,
                              "--n_particles", str(npart), "--n_stages", str(n_stages),
                              "--n_mcmc", str(n_mcmc), "--save_dir", save] + GS + COMMON))
    return jobs


def nuts_jobs(save, n_warmup, n_samples, g, stat, gpu_only=False):
    jobs = []
    GS = ["--G", str(g), "--which_stat", stat]
    # GPU: vectorized chains sweep (n_chains is the batched axis)
    for nc in [1, 8, 32, 128]:
        jobs.append((f"nuts_gpu_nc{nc}_G{g}_{stat}", "gpu",
                     ["mpr_jax_numpyro.py", "--sampler", "nuts",
                      "--chain_method", "vectorized", "--n_chains", str(nc),
                      "--n_warmup", str(n_warmup), "--n_samples", str(n_samples),
                      "--save_dir", save] + GS + COMMON))
    # CPU baseline: only low chain counts (multi-chain CPU NUTS is impractical)
    if not gpu_only:
        for nc in [1, 8]:
            jobs.append((f"nuts_cpu_nc{nc}_G{g}_{stat}", "cpu",
                         ["mpr_jax_numpyro.py", "--sampler", "nuts",
                          "--chain_method", "parallel", "--n_chains", str(nc),
                          "--n_warmup", str(n_warmup), "--n_samples", str(n_samples),
                          "--save_dir", save] + GS + COMMON))
    return jobs


def run(jobs, save, gpu_slots=(0, 1), cpu_cap=1, poll=3.0):
    out = os.path.join(save, "out")
    free_gpu = list(gpu_slots)
    cpu_running = 0
    pending = list(jobs)
    running = []  # (name, popen, slot, logfh)

    def launch(job):
        nonlocal cpu_running
        name, backend, argv = job
        env = dict(os.environ)
        fh = open(os.path.join(out, name + ".log"), "w")
        if backend == "gpu":
            slot = free_gpu.pop(0)
            env["CUDA_VISIBLE_DEVICES"] = str(slot)
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        else:
            slot = None
            env["JAX_PLATFORMS"] = "cpu"
            cpu_running += 1
        p = subprocess.Popen(["python3"] + argv, cwd=FUNCS_DIR, env=env,
                             stdout=fh, stderr=subprocess.STDOUT)
        tag = backend if slot is None else f"gpu{slot}"
        print(f"[{time.strftime('%H:%M:%S')}] launch {name} ({tag}) pid={p.pid}", flush=True)
        return (name, p, slot, fh)

    while pending or running:
        moved = True
        while moved:
            moved = False
            for i, job in enumerate(pending):
                _, backend, _ = job
                if backend == "gpu" and free_gpu:
                    running.append(launch(pending.pop(i))); moved = True; break
                if backend == "cpu" and cpu_running < cpu_cap:
                    running.append(launch(pending.pop(i))); moved = True; break
        time.sleep(poll)
        for entry in running[:]:
            name, p, slot, fh = entry
            if p.poll() is not None:
                fh.close()
                print(f"[{time.strftime('%H:%M:%S')}] done  {name} exit={p.returncode}", flush=True)
                running.remove(entry)
                if slot is not None:
                    free_gpu.append(slot)
                else:
                    cpu_running -= 1
    print(f"[{time.strftime('%H:%M:%S')}] ALL JOBS DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", nargs="+", required=True,
                    choices=["forward", "smc", "nuts"])
    ap.add_argument("--n_warmup", type=int, default=1000)
    ap.add_argument("--n_samples", type=int, default=1000)
    ap.add_argument("--n_stages", type=int, default=50)
    ap.add_argument("--n_mcmc", type=int, default=5)
    ap.add_argument("--cpu_cap", type=int, default=1)
    ap.add_argument("--G", type=float, default=0.2,
                    help="true G for this run (multi-G campaign loops the runner over G).")
    ap.add_argument("--gpu_only", action="store_true",
                    help="skip CPU baseline cells (multi-G/FCD campaign: GPU-only to avoid the "
                         "slow CPU np256 bottleneck; CPU baseline already done at G=0.2 FC).")
    ap.add_argument("--which_stat", type=str, default="FC", choices=["FC", "FCD"],
                    help="summary statistic for the SMC/NUTS jobs (FC or FCD).")
    args = ap.parse_args()

    save = results_dir()
    jobs = []
    if "forward" in args.suite:      # forward throughput is G-/stat-independent -> run once
        jobs += forward_jobs(save)
    if "smc" in args.suite:
        jobs += smc_jobs(save, args.n_stages, args.n_mcmc, args.G, args.which_stat, args.gpu_only)
    if "nuts" in args.suite:
        jobs += nuts_jobs(save, args.n_warmup, args.n_samples, args.G, args.which_stat, args.gpu_only)

    print(f"scheduling {len(jobs)} jobs -> {save}/out/  (GPUs={len(range(2))}, cpu_cap={args.cpu_cap})", flush=True)
    run(jobs, save, cpu_cap=args.cpu_cap)


if __name__ == "__main__":
    main()
