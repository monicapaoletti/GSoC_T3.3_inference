#!/usr/bin/env python3
"""
Grid-posterior SBC for the 1-D (G only) problem: the exact posterior, no sampler.

WHY. In the SBC runs the likelihood is exact by construction: the forward model uses the
same noise seed as the observed data (frozen noise), and --noisy_obs adds exactly the
N(0, obs_err) the likelihood assumes. So if the posterior were computed exactly, SBC would
pass -- unless the setup itself depends on the truth. Two suspects are left:
  (a) the SMC sampler (rugged frozen-noise likelihood at high G), and
  (b) the feature standardisation, which by default is computed at each replicate's OWN
      true G (feature_reference_scatter), so the likelihood changes with the truth.
With one parameter the exact posterior is cheap: evaluate the likelihood on a dense grid
of G and multiply by the HalfNormal prior. This driver does that for the same truths and
seeds as sbc_run.py, twice -- standardisation at the truth (what every SBC campaign ran)
and pinned with --std_ref_G -- so:
  grid calibrated, SMC not          -> the sampler is the cause
  grid miscalibrated, pinned fixes  -> the truth-dependent standardisation is the cause
  grid miscalibrated either way     -> something else in the setup (look again)

Each replicate is two calls of the unmodified mpr_jax_numpyro.py --profile_grid: a coarse
grid over the whole prior range, then a fine grid around the coarse maximum (FC
posteriors are ~1e-5 wide, far below any affordable full-range spacing). The posterior is
the fine grid inside its window and the coarse one outside it.

Usage (one replicate; the Leonardo array runs one per task):
  python sbc_grid.py run --which_stat FCD --i 17 --out_dir <dir> [--std_ref_G 0.33]
Aggregate (after all tasks; also compares with the SMC draws if --smc_dir is given):
  python sbc_grid.py aggregate --which_stat FCD --out_dir <dir> [--smc_dir <sbc dir>]
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sbc_run import prior_draws  # noqa: E402  same truths as the SMC campaigns

INNER = os.path.join(HERE, "mpr_jax_numpyro.py")
G_BINS = [0, 0.2, 0.35, 0.5, 0.7, np.inf]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("mode", choices=["run", "aggregate"])
    p.add_argument("--which_stat", required=True, choices=["FC", "FCD"])
    p.add_argument("--out_dir", required=True)
    p.add_argument("--i", type=int, help="replicate index (run mode)")
    p.add_argument("--L", type=int, default=100)
    p.add_argument("--scale", type=float, default=0.5, help="HalfNormal prior scale, as in "
                   "the n=88 SBC campaigns (sbc_n88_submit.sh --scale 0.5)")
    p.add_argument("--sbc_seed", type=int, default=20260805)
    p.add_argument("--variants", nargs="+", default=["truth", "pinned"],
                   help="truth = standardisation at the replicate's true G (as run); "
                        "pinned = at --std_ref_G")
    p.add_argument("--std_ref_G", type=float, default=0.33)
    p.add_argument("--n_coarse", type=int, default=4001)
    p.add_argument("--gmax", type=float, default=2.0, help="P(G>2) = 6e-5 under "
                   "HalfNormal(0.5)")
    p.add_argument("--n_fine", type=int, default=4001)
    p.add_argument("--fine_halfwidth", type=int, default=20,
                   help="fine window = coarse argmax +- this many coarse spacings")
    p.add_argument("--smc_dir", default=None)
    p.add_argument("--python", default=sys.executable)
    return p.parse_args()


def truth(a, i):
    return float(np.round(prior_draws(a.L, a.scale, a.sbc_seed)[i], 6)), 1000 + i


def profile(a, G, seed, gmin, gmax, n, variant, sub):
    d = os.path.join(a.out_dir, "profiles", variant, sub)
    os.makedirs(d, exist_ok=True)
    pat = os.path.join(d, f"profile_G{G}_*seed{seed}_*which_stat_{a.which_stat}_*.csv")
    if not glob.glob(pat):
        cmd = [a.python, INNER, "--sampler", "smc_lik", "--which_stat", a.which_stat,
               "--G", str(G), "--seed", str(seed), "--save_dir", d,
               "--n_particles", "4096", "--SC_type", "data", "--SC_size", "88",
               "--t_end", "30000", "--tr", "1", "--cut", "10", "--scale", str(a.scale),
               "--fast_bold", "--profile_grid", str(n),
               "--profile_gmin", repr(gmin), "--profile_gmax", repr(gmax)]
        if variant == "pinned":
            cmd += ["--std_ref_G", str(a.std_ref_G)]
        print("[grid] " + " ".join(cmd[2:]), flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        print("\n".join(l for l in r.stdout.splitlines() if l.startswith("[profile]")))
        if r.returncode != 0:
            raise SystemExit(f"[grid] inner run failed rc={r.returncode}\n{r.stdout[-2000:]}\n"
                             f"{r.stderr[-2000:]}")
    import pandas as pd
    df = pd.read_csv(glob.glob(pat)[0])
    return df["G"].to_numpy(float), df["loglik"].to_numpy(float)


def run_one(a, i):
    G, seed = truth(a, i)
    for v in a.variants:
        out = os.path.join(a.out_dir, f"grid_{a.which_stat}_{v}_i{i:03d}.npz")
        if os.path.exists(out):
            print(f"[grid] {out} exists, skipping"); continue
        gc, lc = profile(a, G, seed, 1e-4, a.gmax, a.n_coarse, v, "coarse")
        lc = np.where(np.isfinite(lc), lc, -np.inf)
        lp = lc + log_prior(gc, a.scale)
        h = (gc[1] - gc[0]) * a.fine_halfwidth
        gm = gc[np.argmax(lp)]
        lo, hi = max(1e-6, gm - h), min(a.gmax, gm + h)
        gf, lf = profile(a, G, seed, lo, hi, a.n_fine, v, "fine")
        lf = np.where(np.isfinite(lf), lf, -np.inf)
        # fine grid inside its window, coarse outside. The two are the SAME function
        # evaluated at different points (identical obs, seed, standardisation), so they
        # splice without any offset.
        keep = (gc < gf[0]) | (gc > gf[-1])
        g = np.concatenate([gc[keep], gf]); ll = np.concatenate([lc[keep], lf])
        o = np.argsort(g); g, ll = g[o], ll[o]
        np.savez(out, G=g, loglik=ll, G_true=G, seed=seed, variant=v,
                 std_ref_G=(a.std_ref_G if v == "pinned" else np.nan), scale=a.scale)
        s = summarise(g, ll, G, a.scale)
        print(f"[grid] i={i} G_true={G:.6f} {v}: mean={s['mean']:.6f} sd={s['sd']:.3g} "
              f"u={s['u']:.3f} map={s['map']:.6f}  fine window [{lo:.5f}, {hi:.5f}]")


def log_prior(g, scale):
    return -0.5 * (g / scale) ** 2          # HalfNormal on G >= 0, up to a constant


def summarise(g, ll, G_true, scale):
    lp = ll + log_prior(g, scale)
    w = np.exp(lp - lp.max())
    # trapezoid CDF on a non-uniform grid
    seg = 0.5 * (w[1:] + w[:-1]) * np.diff(g)
    cdf = np.concatenate([[0.0], np.cumsum(seg)]); Z = cdf[-1]; cdf /= Z
    pdf = w / Z
    mean = np.trapz(g * pdf, g); sd = np.sqrt(max(np.trapz((g - mean) ** 2 * pdf, g), 0))
    u = float(np.interp(G_true, g, cdf))
    # mass in the single most probable cell: if ~1, the grid does not resolve the peak
    top = float(seg.max() / Z)
    q = lambda p: float(np.interp(p, cdf, g))
    return dict(mean=mean, sd=sd, u=u, map=float(g[np.argmax(lp)]), top_cell_mass=top,
                q03=q(0.03), q97=q(0.97))


def aggregate(a):
    import pandas as pd
    from scipy.stats import kstest
    rows = []
    for f in sorted(glob.glob(os.path.join(a.out_dir, f"grid_{a.which_stat}_*_i*.npz"))):
        z = np.load(f)
        s = summarise(z["G"], z["loglik"], float(z["G_true"]), float(z["scale"]))
        s.update(variant=str(z["variant"]), G_true=float(z["G_true"]), seed=int(z["seed"]))
        if a.smc_dir:
            hits = glob.glob(os.path.join(a.smc_dir, f"draws_G{s['G_true']}_cut*seed{s['seed']}_"
                                                     f"*smc_lik_which_stat_{a.which_stat}_*.npz"))
            if hits:
                d = np.load(hits[0])["G"].ravel()
                s.update(smc_mean=d.mean(), smc_sd=d.std(), smc_u=float(np.mean(d < s["G_true"])),
                         smc_minus_grid_in_grid_sd=(d.mean() - s["mean"]) / max(s["sd"], 1e-300),
                         smc_sd_over_grid_sd=d.std() / max(s["sd"], 1e-300))
        rows.append(s)
    if not rows:
        raise SystemExit("[grid] nothing to aggregate")
    df = pd.DataFrame(rows)
    csv = os.path.join(a.out_dir, f"grid_sbc_{a.which_stat}.csv")
    df.to_csv(csv, index=False)
    print(f"[grid] {len(df)} rows -> {csv}")
    for v, g in df.groupby("variant"):
        u = g["u"].to_numpy()
        cov = np.mean((u > 0.03) & (u < 0.97))
        print(f"\n== {a.which_stat} grid posterior, standardisation '{v}' (n={len(g)})")
        print(f"   KS p={kstest(u, 'uniform').pvalue:.3g}  cov94={cov:.2f}  "
              f"var(u)*12={np.var(u) * 12:.2f}  median top-cell mass={g['top_cell_mass'].median():.3f}")
        b = g.assign(b=pd.cut(g["G_true"], G_BINS)).groupby("b", observed=True)
        print("   cov94 by true G: " + "; ".join(
            f"{k}: {np.mean((x['u'] > 0.03) & (x['u'] < 0.97)):.2f} (n={len(x)})" for k, x in b))
        if "smc_u" in g and g["smc_u"].notna().any():
            su = g["smc_u"].dropna().to_numpy()
            print(f"   SMC on the same truths: KS p={kstest(su, 'uniform').pvalue:.3g} "
                  f"cov94={np.mean((su > 0.03) & (su < 0.97)):.2f}; "
                  f"median |SMC mean - grid mean| = "
                  f"{g['smc_minus_grid_in_grid_sd'].abs().median():.2f} grid sd, "
                  f"median SMC sd / grid sd = {g['smc_sd_over_grid_sd'].median():.2f}")
    if (df["top_cell_mass"] > 0.2).any():
        print(f"\n[grid] WARNING: {int((df['top_cell_mass'] > 0.2).sum())} posteriors put >20% "
              f"of their mass in one grid cell -- unresolved; raise --n_fine or narrow the window")


def main():
    a = parse_args()
    a.out_dir = os.path.abspath(a.out_dir)
    os.makedirs(a.out_dir, exist_ok=True)
    if a.mode == "run":
        if a.i is None:
            raise SystemExit("--i is required in run mode")
        run_one(a, a.i)
    else:
        aggregate(a)


if __name__ == "__main__":
    main()
