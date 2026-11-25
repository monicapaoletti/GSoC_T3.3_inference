"""Two-GPU work-queue runner for the doslis GPU batching benchmark.

doslis has NO SLURM, so this is a tiny scheduler: it runs one GPU job per visible
device at once (each pinned via CUDA_VISIBLE_DEVICES -- see visible_gpus(), which
is also what makes it correct inside a 1-GPU Slurm allocation) plus a small CPU lane
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


def forward_jobs(save, sc_size=10):
    return [
        ("forward_gpu", "gpu", ["microbench_batch.py", "--outdir", save,
                                "--sc_size", str(sc_size)]),
        ("forward_cpu", "cpu", ["microbench_batch.py", "--outdir", save,
                                "--sc_size", str(sc_size),
                                "--timeout_per_batch", "60"]),
    ]


# The batched axis is swept over these by default (n_particles for SMC, n_chains for the
# chain-vmapped MCMC). Overridable per run via --batch_sizes so a single lost point can be
# re-run on its own instead of redoing the three beside it that already landed.
BATCHES = [64, 256, 1024, 4096]


def smc_jobs(save, n_stages, n_mcmc, g, stat, gpu_only=False, flavors=None, smc_move="rw",
             batches=None, adapt_rw=False):
    jobs = []
    GS = ["--G", str(g), "--which_stat", stat]
    # tempered DE-MC is a separate algorithm -> separate job name and (via the driver)
    # a separate sampler label in the results, so it never collides with the rw cells.
    MOVE = [] if smc_move == "rw" else ["--smc_move", smc_move]
    # Off by default so the 10-node matrix and the UGent n=88 matrix stay reproducible.
    # It is recorded in each benchmark row (adapt_rw column), but NOT in the filename, so
    # an adaptive and a fixed-step run must go to different --results_dir.
    if adapt_rw:
        MOVE = MOVE + ["--adapt_rw"]
    sfx = "" if smc_move == "rw" else f"_{smc_move}"
    for flavor in (flavors or ["smc_lik", "smc_abc"]):
        # GPU: full particle sweep (batched axis -> ~free on GPU)
        for npart in (batches or BATCHES):
            jobs.append((f"{flavor}{sfx}_gpu_np{npart}_G{g}_{stat}", "gpu",
                         ["mpr_jax_numpyro.py", "--sampler", flavor,
                          "--n_particles", str(npart), "--n_stages", str(n_stages),
                          "--n_mcmc", str(n_mcmc), "--save_dir", save] + MOVE + GS + COMMON))
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


def mcmc_jobs(save, n_tune, n_draws, g, stat, gpu_only=False, samplers=None,
              batches=None):
    """GPU chain-vmapped gradient-free samplers (rwmh/demc/slice): n_chains is the
    batched axis, swept like SMC's n_particles. slice is eval-heavy per step, so
    keep n_tune/n_draws modest (and typically run it only at one G)."""
    jobs = []
    GS = ["--G", str(g), "--which_stat", stat]
    for sampler in (samplers or ["rwmh", "demc", "slice"]):
        for nc in (batches or BATCHES):
            jobs.append((f"{sampler}_gpu_nc{nc}_G{g}_{stat}", "gpu",
                         ["mpr_jax_numpyro.py", "--sampler", sampler, "--n_chains", str(nc),
                          "--n_warmup", str(n_tune), "--n_samples", str(n_draws),
                          "--save_dir", save] + GS + COMMON))
        if not gpu_only:                    # one small-nc CPU baseline per sampler
            jobs.append((f"{sampler}_cpu_nc64_G{g}_{stat}", "cpu",
                         ["mpr_jax_numpyro.py", "--sampler", sampler, "--n_chains", "64",
                          "--n_warmup", str(n_tune), "--n_samples", str(n_draws),
                          "--save_dir", save] + GS + COMMON))
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


def visible_gpus():
    """The device ids this process may hand out, one entry per GPU it actually owns.

    A child is pinned with CUDA_VISIBLE_DEVICES=<slot>, and that variable is absolute:
    setting it in the child REPLACES ours rather than indexing into it, so the ids have
    to be the ones we can see, not 0..n-1. Under Slurm the allocation is already masked
    into CUDA_VISIBLE_DEVICES (one entry per --gpus-per-node), so a 1-GPU job yields a
    single slot. The old hardcoded (0, 1) assumed doslis's two L4s and, in a 1-GPU Slurm
    job, sent every second cell to a device that was not there -- each died in ~6 s with
    "FAILED_PRECONDITION: No visible GPU devices" while the job still exited 0.
    """
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if vis:
        return tuple(d.strip() for d in vis.split(",") if d.strip())
    n = int(os.environ.get("SLURM_GPUS_ON_NODE", 0) or 0)
    return tuple(str(i) for i in range(n)) if n else ("0", "1")  # doslis: 2 L4s, no Slurm


def run(jobs, save, gpu_slots=("0",), cpu_cap=1, poll=3.0):
    out = os.path.join(save, "out")
    free_gpu = list(gpu_slots)
    cpu_running = 0
    pending = list(jobs)
    running = []  # (name, popen, slot, logfh)
    failed = []   # (name, returncode)

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
                verdict = "done " if p.returncode == 0 else "FAILED"
                if p.returncode != 0:
                    failed.append((name, p.returncode))
                print(f"[{time.strftime('%H:%M:%S')}] {verdict} {name} exit={p.returncode}",
                      flush=True)
                running.remove(entry)
                if slot is not None:
                    free_gpu.append(slot)
                else:
                    cpu_running -= 1
    print(f"[{time.strftime('%H:%M:%S')}] ALL JOBS DONE "
          f"({len(jobs) - len(failed)}/{len(jobs)} ok)", flush=True)
    # A cell that dies leaves no output file, and a pool that still exits 0 lets Slurm
    # record the whole job COMPLETED -- which is how the first n=88 benchmark reported
    # success on 16 of 80 runs. Fail loudly instead.
    for name, rc in failed:
        print(f"  FAILED {name} (exit {rc}) -> {os.path.join(out, name + '.log')}", flush=True)
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", nargs="+", required=True,
                    choices=["forward", "smc", "mcmc", "nuts"])
    ap.add_argument("--n_warmup", type=int, default=1000)
    ap.add_argument("--n_samples", type=int, default=1000)
    ap.add_argument("--smc_move", type=str, default="rw", choices=["rw", "demc"],
                    help="within-stage SMC kernel; 'demc' = tempered DE-MC, recorded as a "
                         "separate sampler row (<flavor>_demc)")
    ap.add_argument("--adapt_rw", action="store_true",
                    help="pass --adapt_rw to every SMC cell: tune the random-walk scale "
                         "toward 0.234 acceptance. Without it the n=88 SMC/FC cloud "
                         "collapses (sd 0). Not comparable with runs made without it.")
    ap.add_argument("--n_stages", type=int, default=50)
    ap.add_argument("--n_mcmc", type=int, default=5)
    ap.add_argument("--cpu_cap", type=int, default=1)
    ap.add_argument("--gpu_concurrency", type=int, default=1,
                    help="GPU jobs to run PER DEVICE concurrently. Each job is tiny "
                         "(~250 MiB, GPU under-utilized), so several fit and fill the "
                         "idle device -> higher matrix throughput at no memory cost.")
    ap.add_argument("--G", type=float, default=0.2,
                    help="true G for this run (multi-G campaign loops the runner over G).")
    ap.add_argument("--gpu_only", action="store_true",
                    help="skip CPU baseline cells (multi-G/FCD campaign: GPU-only to avoid the "
                         "slow CPU np256 bottleneck; CPU baseline already done at G=0.2 FC).")
    ap.add_argument("--which_stat", type=str, default="FC", choices=["FC", "FCD"],
                    help="summary statistic for the SMC/NUTS jobs (FC or FCD).")
    ap.add_argument("--mcmc_samplers", nargs="+", default=None,
                    choices=["rwmh", "demc", "slice"],
                    help="subset of chain-vmapped samplers for the mcmc suite "
                         "(default all three; e.g. run slice only at one G).")
    ap.add_argument("--smc_samplers", nargs="+", default=None,
                    choices=["smc_lik", "smc_abc"],
                    help="subset of SMC flavors for the smc suite (default both; "
                         "e.g. smc_abc only for the fixed-epsilon re-run).")
    ap.add_argument("--SC_size", type=int, default=10,
                    help="nodes in the network. 10 is the CPU-comparable small model the "
                         "paper's matrix uses; 88 is the whole-brain connectome. Every "
                         "cell in one campaign must share it -- see below.")
    ap.add_argument("--batch_sizes", nargs="+", type=int, default=None,
                    help="override the swept batched axis (n_particles for smc, n_chains "
                         "for mcmc). Default 64 256 1024 4096. Use it to re-run a single "
                         "lost point without redoing the ones beside it that landed.")
    ap.add_argument("--results_dir", default=None,
                    help="pin the output directory (avoids the midnight date-rollover "
                         "splitting a long campaign across two results/<date> folders).")
    args = ap.parse_args()

    # SC_size is baked into COMMON rather than threaded per job the way G and which_stat
    # are: those vary WITHIN a campaign and are part of what it compares, whereas the
    # network is what the comparison is held fixed against. Rewriting it once, here,
    # makes it impossible for one cell in a results dir to disagree with another.
    COMMON[COMMON.index("--SC_size") + 1] = str(args.SC_size)

    save = args.results_dir if args.results_dir else results_dir()
    os.makedirs(os.path.join(save, "out"), exist_ok=True)
    jobs = []
    if "forward" in args.suite:      # forward throughput is G-/stat-independent -> run once
        jobs += forward_jobs(save, args.SC_size)
    if "smc" in args.suite:
        jobs += smc_jobs(save, args.n_stages, args.n_mcmc, args.G, args.which_stat,
                         args.gpu_only, args.smc_samplers, args.smc_move,
                         batches=args.batch_sizes, adapt_rw=args.adapt_rw)
    if "mcmc" in args.suite:
        jobs += mcmc_jobs(save, args.n_warmup, args.n_samples, args.G, args.which_stat,
                          args.gpu_only, args.mcmc_samplers, batches=args.batch_sizes)
    if "nuts" in args.suite:
        jobs += nuts_jobs(save, args.n_warmup, args.n_samples, args.G, args.which_stat, args.gpu_only)

    gpu_slots = tuple(d for d in visible_gpus() for _ in range(max(1, args.gpu_concurrency)))
    print(f"scheduling {len(jobs)} jobs -> {save}/out/  (gpu_slots={gpu_slots}, "
          f"cpu_cap={args.cpu_cap})", flush=True)
    raise SystemExit(run(jobs, save, gpu_slots=gpu_slots, cpu_cap=args.cpu_cap))


if __name__ == "__main__":
    main()
