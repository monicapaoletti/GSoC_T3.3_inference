#!/usr/bin/env python
"""
Calibration diagnostics for an SBC campaign, from the draws it already saved. No new
simulations.

Why this exists: the KS test on SBC ranks reacts mainly to a SHIFT of the rank
distribution (bias) and barely to a symmetric U shape (intervals too narrow). The 2-D FCD
campaigns pass KS while their 94% intervals cover ~0.78, so KS alone is not a calibration
verdict. These are the checks that see what KS misses:

  1. expected coverage curve   empirical coverage of central intervals at every level
                               (5%..99%) against nominal, with a binomial 95% band
  2. ECDF difference + simultaneous 95% band (Saeilynoja et al. 2022): ECDF(u) - u over
                               the whole curve at once, so a U shape shows as an S
  3. rank dispersion           var(u) / (1/12): > 1 means overconfident (U shape),
                               < 1 underconfident (hump); bootstrap 95% CI
  4. u and z against the true G, and coverage by range of true G
  5. TARP (Lemos et al. 2023)  joint calibration of (G, |eta|) for 2-D campaigns

u is the normalised SBC rank: the fraction of posterior draws below the truth (ties broken
at random, as in sbc_run.rank_of).

Usage (myenv python, from scripts/functions/):
  python calib_diagnostics.py --out <dir> <sbc_dir> [<sbc_dir> ...]
Each <sbc_dir> holds draws_*.npz from sbc_run.py. Writes one PNG per (dir, sampler) and a
summary CSV.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

LEVELS = np.linspace(0.05, 0.99, 48)
G_BINS = [0, 0.2, 0.35, 0.5, 0.7, np.inf]


def load_campaign(d, sampler):
    rows = []
    rng = np.random.default_rng(0)
    for f in sorted(glob.glob(os.path.join(d, f"draws_*_sampler_{sampler}_*.npz"))):
        z = np.load(f)
        rec = {"file": os.path.basename(f), "G_true": float(z["G_true"])}
        draws = {"G": np.asarray(z["G"]).ravel()}
        truths = {"G": float(z["G_true"])}
        if "eta_mag" in z.files:
            draws["eta"] = np.asarray(z["eta_mag"]).ravel()
            truths["eta"] = abs(float(np.asarray(z["theta_true"])[1]))
        for p, x in draws.items():
            x = x[np.isfinite(x)]
            t = truths[p]
            below, ties = np.sum(x < t), np.sum(x == t)
            rec[f"u_{p}"] = (below + (rng.integers(0, ties + 1) if ties else 0)) / x.size
            rec[f"mean_{p}"], rec[f"sd_{p}"] = x.mean(), x.std()
            rec[f"z_{p}"] = (x.mean() - t) / max(x.std(), 1e-300)
            rec[f"true_{p}"] = t
        if "n_unique" in z.files:
            rec["unique_frac"] = float(z["n_unique"]) / draws["G"].size
        rec["_draws"] = draws
        rows.append(rec)
    return rows


def coverage_curve(u):
    """Central-interval coverage: the truth sits inside the central level-L interval iff
    |u - 0.5| <= L/2."""
    return np.array([np.mean(np.abs(u - 0.5) <= L / 2) for L in LEVELS])


def simultaneous_band(n, K=200, M=4000, conf=0.95, seed=1):
    """Simultaneous band for ECDF(u)-u on a K-point grid, by Monte Carlo: find the
    pointwise binomial level gamma such that a fraction `conf` of uniform ECDFs stays
    inside the band EVERYWHERE (the Saeilynoja et al. construction, done by simulation)."""
    from scipy.stats import binom
    zg = np.linspace(0, 1, K + 2)[1:-1]
    sims = np.sort(np.random.default_rng(seed).uniform(size=(M, n)), axis=1)
    counts = np.stack([np.searchsorted(s, zg, side="right") for s in sims])
    lo_g, hi_g = 1e-6, 1 - conf
    for _ in range(40):
        g = 0.5 * (lo_g + hi_g)
        lo, hi = binom.ppf(g / 2, n, zg), binom.ppf(1 - g / 2, n, zg)
        inside = np.mean(np.all((counts >= lo) & (counts <= hi), axis=1))
        lo_g, hi_g = (g, hi_g) if inside > conf else (lo_g, g)
    g = lo_g
    return zg, binom.ppf(g / 2, n, zg) / n - zg, binom.ppf(1 - g / 2, n, zg) / n - zg


def dispersion(u, B=4000, seed=2):
    r = np.var(u) * 12
    bs = np.random.default_rng(seed).choice(u, size=(B, u.size))
    return r, np.percentile(np.var(bs, axis=1) * 12, [2.5, 97.5])


def tarp(rows, n_ref=None, seed=3):
    """TARP expected coverage for the joint (G, |eta|) posterior. Parameters are put on a
    common scale by the spread of the truths; one random reference point per replicate,
    drawn uniformly over the box the truths span."""
    rng = np.random.default_rng(seed)
    T = np.array([[r["true_G"], r["true_eta"]] for r in rows])
    lo, hi = T.min(0), T.max(0)
    sc = hi - lo
    f = []
    for r, t in zip(rows, T):
        S = np.stack([r["_draws"]["G"], r["_draws"]["eta"]], 1)
        S = S[np.all(np.isfinite(S), 1)]
        ref = lo + rng.uniform(size=2) * sc
        d_s = np.linalg.norm((S - ref) / sc, axis=1)
        d_t = np.linalg.norm((t - ref) / sc)
        f.append(np.mean(d_s < d_t))
    f = np.asarray(f)
    return np.array([np.mean(f < a) for a in LEVELS])


def binom_band(n):
    from scipy.stats import binom
    return (binom.ppf(0.025, n, LEVELS) / n, binom.ppf(0.975, n, LEVELS) / n)


def plot_campaign(name, rows, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import kstest
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "_draws"} for r in rows])
    params = [p for p in ("G", "eta") if f"u_{p}" in df]
    n = len(df)
    two_d = "eta" in params
    ncol = 4 + (1 if two_d else 0)
    fig, axs = plt.subplots(len(params), ncol, figsize=(4.2 * ncol, 3.8 * len(params)),
                            squeeze=False)
    blo, bhi = binom_band(n)
    zg, elo, ehi = simultaneous_band(n)
    summary = []
    for i, p in enumerate(params):
        u = df[f"u_{p}"].to_numpy()
        cov = coverage_curve(u)
        disp, (dlo, dhi) = dispersion(u)
        ks = kstest(u, "uniform").pvalue
        ecdf = np.searchsorted(np.sort(u), zg, side="right") / n - zg
        outside = bool(np.any((ecdf < elo) | (ecdf > ehi)))
        lab = "G" if p == "G" else "|eta|"
        ax = axs[i, 0]
        ax.fill_between(LEVELS, blo, bhi, color="0.85", label="95% band (calibrated)")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.plot(LEVELS, cov, "o-", ms=3, label="empirical")
        ax.set(xlabel="nominal coverage", ylabel="empirical coverage",
               title=f"{lab}: expected coverage")
        ax.legend(fontsize=7)
        ax = axs[i, 1]
        # band as two step lines, not a fill: it is discrete (binomial) and its jagged
        # edges drawn as a fill look far wider than they are near u = 0 and u = 1
        ax.step(zg, elo, where="mid", color="0.5", lw=0.8, label="simultaneous 95%")
        ax.step(zg, ehi, where="mid", color="0.5", lw=0.8)
        ax.axhline(0, color="k", lw=0.8)
        ax.plot(zg, ecdf, lw=1.5)
        bad = (ecdf < elo) | (ecdf > ehi)
        ax.plot(zg[bad], ecdf[bad], "o", color="C3", ms=3, label="outside band")
        ax.legend(fontsize=7)
        ax.set(xlabel="u (normalised rank)", ylabel="ECDF(u) - u",
               title=f"{lab}: ECDF diff ({'OUTSIDE' if outside else 'inside'} band), KS p={ks:.2g}")
        ax = axs[i, 2]
        ax.hist(u, bins=10, range=(0, 1), color="C0", alpha=0.8)
        ax.axhline(n / 10, color="k", ls="--", lw=0.8)
        ax.set(xlabel="u", ylabel="count",
               title=f"{lab}: var(u)*12 = {disp:.2f} [{dlo:.2f}, {dhi:.2f}]")
        ax = axs[i, 3]
        sc = ax.scatter(df["true_G"], df[f"z_{p}"], c=df[f"u_{p}"], cmap="coolwarm",
                        vmin=0, vmax=1, s=14)
        ax.axhline(0, color="k", lw=0.8)
        for y in (-2, 2):
            ax.axhline(y, color="0.5", ls=":", lw=0.8)
        ax.set_yscale("symlog", linthresh=3)
        ax.set(xlabel="true G", ylabel="z = (mean - true)/sd", title=f"{lab}: z vs true G")
        fig.colorbar(sc, ax=ax, label="u", fraction=0.05)
        binned = df.assign(b=pd.cut(df["true_G"], G_BINS)).groupby("b", observed=True)
        by_bin = {str(k): (len(g), float(np.mean(np.abs(g[f"u_{p}"] - 0.5) <= 0.47)))
                  for k, g in binned}
        summary.append(dict(campaign=name, param=lab, n=n, ks_p=ks, ecdf_outside_band=outside,
                            var_ratio=disp, var_ratio_lo=dlo, var_ratio_hi=dhi,
                            cov50=coverage_curve(u)[np.argmin(np.abs(LEVELS - 0.5))],
                            cov80=coverage_curve(u)[np.argmin(np.abs(LEVELS - 0.8))],
                            cov94=float(np.mean(np.abs(u - 0.5) <= 0.47)),
                            median_z=float(np.median(df[f"z_{p}"])),
                            cov94_by_trueG="; ".join(f"{k}: {c:.2f} (n={m})"
                                                     for k, (m, c) in by_bin.items())))
    if two_d:
        ecp = tarp(rows)
        for i in range(len(params)):
            ax = axs[i, 4]
            if i == 0:
                ax.fill_between(LEVELS, blo, bhi, color="0.85")
                ax.plot([0, 1], [0, 1], "k--", lw=0.8)
                ax.plot(LEVELS, ecp, "o-", ms=3, color="C3")
                ax.set(xlabel="credibility level", ylabel="expected coverage",
                       title="TARP, joint (G, |eta|)")
            else:
                ax.axis("off")
        for s in summary:
            s["tarp_at_0.94"] = float(ecp[np.argmin(np.abs(LEVELS - 0.94))])
    fig.suptitle(f"{name}  (n = {n} replicates)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samplers", nargs="+", default=["smc_lik", "smc_abc"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    allsum = []
    for d in a.dirs:
        for s in a.samplers:
            rows = load_campaign(d, s)
            if not rows:
                continue
            name = f"{os.path.basename(os.path.normpath(d))}  {s}"
            png = os.path.join(a.out, f"calib_{os.path.basename(os.path.normpath(d))}_{s}.png")
            allsum += plot_campaign(name, rows, png)
            print(f"[calib] {name}: {len(rows)} replicates -> {png}")
    df = pd.DataFrame(allsum)
    csv = os.path.join(a.out, "calibration_summary.csv")
    df.to_csv(csv, index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 120):
        print(df.drop(columns=["var_ratio_lo", "var_ratio_hi"]).round(3).to_string(index=False))
    print(f"[calib] wrote {csv}")


if __name__ == "__main__":
    main()
