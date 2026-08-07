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
    p.add_argument("--i_start", type=int, default=0,
                   help="first replicate index to RUN (inclusive). Lets several "
                        "instances split the work across GPUs. The prior draws are "
                        "generated for the full --L before slicing, so replicate k is "
                        "the same ground truth whichever instance runs it.")
    p.add_argument("--i_end", type=int, default=None,
                   help="last replicate index to run (exclusive); default --L.")
    p.add_argument("--infer_eta", action="store_true",
                   help="2-D SBC: draw a (G, eta) PAIR per replicate and infer both. "
                        "Ranks are recorded for each parameter separately -- a joint "
                        "posterior can be well calibrated in one coordinate and not the "
                        "other, and averaging them would hide exactly that.")
    p.add_argument("--eta_prior_scale", type=float, default=0.1,
                   help="sigma of LogNormal(log 4.6, sigma) on |eta|; a MULTIPLICATIVE "
                        "spread, so 0.1 spans |eta| in about [3.8, 5.6] at 95%% while "
                        "the package default 0.5 spans [1.7, 12.5]. Wide priors make an "
                        "SBC pass VACUOUS here: most truths land where the data is "
                        "uninformative, the posterior returns the prior, and the ranks "
                        "are uniform by construction rather than by calibration.")
    p.add_argument("--eta_center", type=float, default=4.6,
                   help="median |eta| of that prior; must match ETA_MAG_DEFAULT in the "
                        "inference or the SBC draws from a different prior than it fits.")
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


def eta_prior_draws(L, center, sigma, seed):
    """|eta| ~ LogNormal(log center, sigma), the SAME prior the inference uses when
    --infer_eta is on. Drawing the truth from a different prior than the one being fitted
    is not a stricter test, it is an invalid one: SBC's uniformity result assumes they
    match."""
    rng = np.random.default_rng(seed + 977)      # offset: independent of the G stream
    return np.exp(rng.normal(np.log(center), sigma, size=L))


def draws_path(out_dir, G, seed, a, eta=None):
    """Mirror the inner script's TAG so we can find what it wrote (and resume)."""
    backend = "gpu" if os.environ.get("JAX_PLATFORMS", "") != "cpu" else "cpu"
    # mirrors the inference's tag: eta appears only when it differs from the default
    _et = "" if (eta is None or abs(float(eta) - (-4.6)) < 1e-9) else f"_eta{eta}"
    tag = (f"G{G}{_et}_cut{a.cut}_tr{a.tr}_seed{seed}_tend{a.t_end}_ns20_nc1_"
           f"SC_{a.SC_size}_sampler_{a.sampler}_which_stat_{a.which_stat}_"
           f"{backend}_cmvectorized_np{a.n_particles}")
    exact = os.path.join(out_dir, f"draws_{tag}.npz")
    if os.path.exists(exact):
        return exact
    # the tag embeds n_samples/backend, which we do not control precisely from here;
    # fall back to matching on the fields that uniquely identify this replicate.
    # The eta component must be part of the pattern, not left to the wildcard. Without
    # it "draws_G0.2_*seed1000_*" matches BOTH the 1-D file and any 2-D file at the same
    # G and seed, so a 2-D replicate would be reported "already done" and silently reuse
    # a 1-D result -- and a 1-D re-run could pick up a 2-D one.
    hits = glob.glob(os.path.join(out_dir, f"draws_G{G}{_et}_cut*seed{seed}_*{a.sampler}_"
                                           f"which_stat_{a.which_stat}_*.npz"))
    return hits[0] if hits else None


def run_replicate(i, G, a, eta=None):
    seed = 1000 + i
    have = draws_path(a.out_dir, G, seed, a, eta=eta)
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
    if eta is not None:
        cmd += ["--eta", str(eta), "--infer_eta",
                "--eta_prior_scale", str(a.eta_prior_scale)]
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
    return draws_path(a.out_dir, G, seed, a, eta=eta)


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

    i_end = a.L if a.i_end is None else a.i_end
    if (a.i_start, i_end) != (0, a.L):
        print(f"[sbc] this instance runs replicates [{a.i_start}, {i_end})")

    Es = (eta_prior_draws(a.L, a.eta_center, a.eta_prior_scale, a.sbc_seed)
          if a.infer_eta else None)
    if Es is not None:
        print(f"[sbc] 2-D: |eta| ~ LogNormal(log {a.eta_center}, {a.eta_prior_scale}) "
              f"-> min={Es.min():.3f} median={np.median(Es):.3f} max={Es.max():.3f}")

    # ranks are kept PER PARAMETER: a joint posterior can be calibrated in G and not in
    # eta, and one pooled statistic would average that away.
    ranks, sizes, used_G = [], [], []
    eta_ranks, eta_sizes, used_eta = [], [], []
    for i, G in enumerate(Gs):
        G = float(np.round(G, 6))
        eta = float(np.round(-Es[i], 6)) if Es is not None else None
        if a.i_start <= i < i_end:
            path = run_replicate(i, G, a, eta=eta)
        else:
            # outside this instance's range: do not run it, but still pick up its draws
            # if a sibling instance has finished it, so a final full-range pass
            # aggregates every replicate without recomputing anything.
            path = draws_path(a.out_dir, G, 1000 + i, a, eta=eta)
        if a.dry_run or path is None:
            continue
        z = np.load(path)
        r, n = rank_of(float(z["G_true"]), z["G"])
        if r is None:
            print(f"[sbc] replicate {i+1} produced no finite draws; skipped")
            continue
        ranks.append(r); sizes.append(n); used_G.append(G)
        msg = f"[sbc]   rank {r}/{n}"
        if Es is not None and "eta_mag" in getattr(z, "files", []):
            # the inference samples the MAGNITUDE, so rank the truth as a magnitude too
            er, en = rank_of(abs(float(eta)), z["eta_mag"])
            if er is not None:
                eta_ranks.append(er); eta_sizes.append(en); used_eta.append(abs(eta))
                msg += f" | eta rank {er}/{en}"
        print(msg)

    if a.dry_run:
        return
    if not ranks:
        raise SystemExit("[sbc] no replicates completed -- nothing to summarise")
    if len(ranks) < a.L:
        print(f"[sbc] NOTE: {len(ranks)}/{a.L} replicates present. Re-run with the full "
              f"range once every instance has finished to aggregate them all.")

    ranks = np.asarray(ranks); sizes = np.asarray(sizes)
    suffix = "_2d" if a.infer_eta else ""
    out = os.path.join(a.out_dir, f"sbc_ranks_{a.sampler}_{a.which_stat}{suffix}.npz")
    payload = dict(ranks=ranks, sizes=sizes, G_true=np.asarray(used_G))
    if eta_ranks:
        payload.update(eta_ranks=np.asarray(eta_ranks), eta_sizes=np.asarray(eta_sizes),
                       eta_true=np.asarray(used_eta))
    np.savez(out, **payload)
    print(f"\n[sbc] {len(ranks)} replicates -> {out}")
    # The 2-D file is named separately so it never overwrites the 1-D campaign's ranks
    # for the same sampler and feature.

    # Normalised ranks should be Uniform(0,1) under correct calibration.
    u = ranks / np.maximum(sizes, 1)
    try:
        from scipy import stats
        for nm, uu in ([("G", u)] + ([("eta_mag",
                        np.asarray(eta_ranks) / np.maximum(np.asarray(eta_sizes), 1))]
                       if eta_ranks else [])):
            ks = stats.kstest(uu, "uniform")
            print(f"[sbc] {nm}: KS against Uniform(0,1) D={ks.statistic:.4f} "
                  f"p={ks.pvalue:.4f} (n={len(uu)})")
            print("[sbc]   " + ("no evidence of miscalibration" if ks.pvalue > 0.05
                               else "MISCALIBRATED at the 5% level -- inspect the ranks"))
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
