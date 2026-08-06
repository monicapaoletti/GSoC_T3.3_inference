#!/usr/bin/env python3
"""Simulation-based calibration (SBC, Talts et al. 2018) for the MPR samplers.

SBC is the check the benchmark table cannot make. Every metric we currently report
-- |dG|, 94% HDI coverage, R-hat -- is evaluated at ONE ground truth per cell, so it
says whether a sampler found the right answer there, not whether its posteriors are
calibrated. SBC tests the latter: if the model and sampler are correct, then drawing
G ~ prior, simulating data at that G, and refitting must leave the RANK of the true G
among the posterior draws uniformly distributed. Systematic deviations are diagnostic
-- U-shaped ranks mean the posterior is too narrow (overconfident), a hump means too
wide, and a slope means bias.

Each replicate is a full, unmodified run of mpr_jax_numpyro.py, launched as a
subprocess with its own prior-drawn --G and its own --seed. Driving the real pipeline
rather than reimplementing it means the thing under test is exactly the thing we
benchmark, and JAX's persistent compilation cache makes the repeated launches cheap
after the first.

Usage (GPU, the sensible place -- an smc_lik fit is ~12 min there against ~14 h on CPU):
  python sbc_run.py --L 100 --sampler smc_lik --which_stat FCD \
      --out_dir ../../results/sbc --n_particles 4096

Re-running resumes: replicates whose draws file already exists are skipped.
"""
import argparse, os, subprocess, sys, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INNER = os.path.join(HERE, "mpr_jax_numpyro.py")


def parse_args():
    p = argparse.ArgumentParser(description="Simulation-based calibration driver")
    p.add_argument("--L", type=int, default=100, help="number of SBC replicates")
    p.add_argument("--sampler", default="smc_lik")
    p.add_argument("--which_stat", default="FCD", choices=["FC", "FCD"])
    p.add_argument("--out_dir", default=os.path.join(HERE, "..", "..", "results", "sbc"))
    p.add_argument("--scale", type=float, default=1.0,
                   help="prior scale; MUST match the inner run's --scale, since SBC "
                        "draws the ground truth from that same prior")
    p.add_argument("--sbc_seed", type=int, default=20260805,
                   help="seed for the prior draws themselves (not the simulations)")
    # inner-run configuration, kept identical to the benchmark cells
    p.add_argument("--n_particles", type=int, default=4096)
    p.add_argument("--n_stages", type=int, default=50)
    p.add_argument("--n_mcmc", type=int, default=5)
    p.add_argument("--SC_type", default="data")
    p.add_argument("--SC_size", type=int, default=10)
    p.add_argument("--t_end", type=int, default=30000)
    p.add_argument("--tr", type=int, default=1)
    p.add_argument("--cut", type=int, default=10)
    p.add_argument("--fcd_band", type=int, default=0)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dry_run", action="store_true",
                   help="print the replicate commands and the prior draws, run nothing")
    return p.parse_args()


def prior_draws(L, scale, seed):
    """Ground truths drawn from the SAME prior the model uses: HalfNormal(scale).

    numpy's half-normal is |N(0, scale)|, which is what numpyro's HalfNormal(scale)
    is too -- keeping these consistent is the whole validity condition for SBC.
    """
    rng = np.random.default_rng(seed)
    return np.abs(rng.normal(0.0, scale, size=L))


def draws_path(out_dir, G, seed, a):
    """Mirror the inner script's TAG so we can find what it wrote (and resume)."""
    backend = "gpu" if os.environ.get("JAX_PLATFORMS", "") != "cpu" else "cpu"
    tag = (f"G{G}_cut{a.cut}_tr{a.tr}_seed{seed}_tend{a.t_end}_ns20_nc1_"
           f"SC_{a.SC_size}_sampler_{a.sampler}_which_stat_{a.which_stat}_"
           f"{backend}_cmvectorized_np{a.n_particles}")
    exact = os.path.join(out_dir, f"draws_{tag}.npz")
    if os.path.exists(exact):
        return exact
    # the tag embeds n_samples/backend, which we do not control precisely from here;
    # fall back to matching on the fields that uniquely identify this replicate.
    hits = glob.glob(os.path.join(out_dir, f"draws_G{G}_*seed{seed}_*{a.sampler}_"
                                           f"which_stat_{a.which_stat}_*.npz"))
    return hits[0] if hits else None


def run_replicate(i, G, a):
    seed = 1000 + i
    have = draws_path(a.out_dir, G, seed, a)
    if have:
        print(f"[sbc] {i+1}/{a.L} G={G:.4f} -- already done, skipping")
        return have
    cmd = [a.python, INNER,
           "--sampler", a.sampler, "--which_stat", a.which_stat,
           "--G", str(G), "--seed", str(seed),
           "--save_dir", a.out_dir, "--save_draws",
           "--n_particles", str(a.n_particles),
           "--n_stages", str(a.n_stages), "--n_mcmc", str(a.n_mcmc),
           "--SC_type", a.SC_type, "--SC_size", str(a.SC_size),
           "--t_end", str(a.t_end), "--tr", str(a.tr), "--cut", str(a.cut),
           "--scale", str(a.scale), "--fast_bold"]
    if a.fcd_band:
        cmd += ["--fcd_band", str(a.fcd_band)]
    if a.dry_run:
        print("  " + " ".join(cmd))
        return None
    print(f"[sbc] {i+1}/{a.L} G={G:.4f} seed={seed} ...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[sbc] REPLICATE FAILED (rc={r.returncode})\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
        return None
    return draws_path(a.out_dir, G, seed, a)


def rank_of(true_value, draws):
    """SBC rank statistic: how many posterior draws fall below the ground truth.

    Ties are broken at random rather than always counting < or <=; with a discrete or
    duplicated particle cloud (SMC resampling produces exact duplicates) a fixed
    convention biases the rank downward and fakes a calibration failure.
    """
    d = np.asarray(draws).ravel()
    d = d[np.isfinite(d)]
    if d.size == 0:
        return None, 0
    below = int(np.sum(d < true_value))
    ties = int(np.sum(d == true_value))
    if ties:
        below += int(np.random.default_rng(0).integers(0, ties + 1))
    return below, d.size


def main():
    a = parse_args()
    a.out_dir = os.path.abspath(a.out_dir)
    os.makedirs(a.out_dir, exist_ok=True)
    Gs = prior_draws(a.L, a.scale, a.sbc_seed)
    print(f"[sbc] {a.L} replicates, {a.sampler}/{a.which_stat}, prior HalfNormal({a.scale})")
    print(f"[sbc] ground truths: min={Gs.min():.3f} median={np.median(Gs):.3f} max={Gs.max():.3f}")
    print(f"[sbc] out_dir={a.out_dir}")

    ranks, sizes, used_G = [], [], []
    for i, G in enumerate(Gs):
        G = float(np.round(G, 6))
        path = run_replicate(i, G, a)
        if a.dry_run or path is None:
            continue
        z = np.load(path)
        r, n = rank_of(float(z["G_true"]), z["G"])
        if r is None:
            print(f"[sbc] replicate {i+1} produced no finite draws; skipped")
            continue
        ranks.append(r); sizes.append(n); used_G.append(G)
        print(f"[sbc]   rank {r}/{n}")

    if a.dry_run:
        return
    if not ranks:
        raise SystemExit("[sbc] no replicates completed -- nothing to summarise")

    ranks = np.asarray(ranks); sizes = np.asarray(sizes)
    out = os.path.join(a.out_dir, f"sbc_ranks_{a.sampler}_{a.which_stat}.npz")
    np.savez(out, ranks=ranks, sizes=sizes, G_true=np.asarray(used_G))
    print(f"\n[sbc] {len(ranks)} replicates -> {out}")

    # Normalised ranks should be Uniform(0,1) under correct calibration.
    u = ranks / np.maximum(sizes, 1)
    try:
        from scipy import stats
        ks = stats.kstest(u, "uniform")
        print(f"[sbc] KS test against Uniform(0,1): D={ks.statistic:.4f} p={ks.pvalue:.4f}")
        print("[sbc] " + ("no evidence of miscalibration" if ks.pvalue > 0.05
                          else "MISCALIBRATED at the 5% level -- inspect the rank histogram"))
    except Exception as e:
        print(f"[sbc] scipy unavailable for the KS test ({e}); ranks saved regardless")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
        nb = 20
        ax[0].hist(u, bins=nb, range=(0, 1), color="#2a78d6", edgecolor="white")
        exp = len(u) / nb
        # binomial 99% band for the count in each equal-width bin
        sd = np.sqrt(len(u) * (1/nb) * (1 - 1/nb))
        ax[0].axhline(exp, color="k", lw=1)
        ax[0].axhspan(exp - 2.576*sd, exp + 2.576*sd, color="k", alpha=0.12, lw=0)
        ax[0].set_xlabel("normalised rank"); ax[0].set_ylabel("count")
        ax[0].set_title(f"SBC rank histogram ({a.sampler}, {a.which_stat}, L={len(u)})")
        xs = np.sort(u); ecdf = np.arange(1, len(xs)+1) / len(xs)
        ax[1].plot(xs, ecdf - xs, color="#eb6834")
        ax[1].axhline(0, color="k", lw=1)
        ax[1].set_xlabel("normalised rank"); ax[1].set_ylabel("ECDF - uniform")
        ax[1].set_title("ECDF difference")
        png = os.path.join(a.out_dir, f"sbc_{a.sampler}_{a.which_stat}.png")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        print(f"[sbc] plot -> {png}")
    except Exception as e:
        print(f"[sbc] plotting failed ({type(e).__name__}: {e}); ranks are saved")


if __name__ == "__main__":
    main()
