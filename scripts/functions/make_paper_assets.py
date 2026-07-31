#!/usr/bin/env python3
"""Aggregate all benchmark CSVs into the paper's assets: a master results table,
LaTeX tables, and figures. Closes the loop results -> manuscript, reproducibly.

Reads every ``benchmark_*.csv`` (the one-row run summaries written by
mpr_jax_numpyro.py / mpr_jax_pymc.py) found under the given results directories --
across date folders and clusters -- and emits:
  * master_results.csv          : the aggregated, tidy table (one row per cell)
  * tables/benchmark_table.tex  : accuracy + performance table for the paper
  * tables/settings_table.tex    : per-sampler budgets (from benchmark_config.yaml)
  * figures/recovery_vs_G.png    : posterior recovery across regimes
  * figures/throughput_ess.png   : ESS/sec vs batch, GPU vs CPU

Usage:
  python make_paper_assets.py --results DIR [DIR ...] --config CONFIG.yaml --out PAPERDIR
  # e.g. after pulling CSVs locally:
  python make_paper_assets.py --results ../../results --out ../../paper
"""
import argparse, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# columns we try to read from the run-row benchmark CSVs (missing ones are tolerated)
_COLS = ["sampler", "sampler_raw", "framework", "platform", "which_stat", "G_true", "n_chains",
         "n_particles", "runtime_sec", "max_r_hat", "min_ess_bulk",
         "min_ess_bulk_per_sec", "rmse_param_mean", "mean_accept", "coverage_94hdi"]

COLORS = {"gpu": "#295785", "cpu": "#B5651D"}

# Validated categorical order, shared with plot_ess_scaling so a sampler keeps the same
# hue across every figure in the paper. All six palette checks pass in light mode.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SAMPLER_ORDER = ["smc_abc", "smc_lik", "demc", "rwmh", "slice", "demcz"]


# mpr_jax_pymc.py records the sampler as a DISPLAY string (one even contains LaTeX),
# and writes neither `platform` nor `framework`. mpr_jax_numpyro.py records canonical
# slugs plus both fields. Without reconciling them the pymc cells cannot join the JAX
# cells at all -- "Slice Sampler" and "slice" are the same algorithm -- and every
# pymc row silently drops out of any platform-split table or figure.
_SAMPLER_ALIASES = {
    "Metropolis": "rwmh",
    "DEMetropolis": "demc",
    "DE MetropolisZ": "demcz",
    "Slice Sampler": "slice",
    "SMC with Likelihood": "smc_lik",
    r"SMC, $\epsilon$ = 10": "smc_abc",
}


def _normalize_provenance(df):
    """Canonicalise sampler names and backfill platform/framework across frameworks."""
    if "sampler" in df:
        df["sampler_raw"] = df["sampler"]
        df["sampler"] = df["sampler"].replace(_SAMPLER_ALIASES)
    # every pymc cell in this project ran on the ulysses CPU cluster; the JAX cells
    # always write their own platform, so a missing value identifies pymc unambiguously.
    if "framework" not in df:
        df["framework"] = np.nan
    if "platform" not in df:
        df["platform"] = np.nan
    is_pymc = df["framework"].isna()
    df.loc[is_pymc, "framework"] = "pymc"
    df.loc[is_pymc & df["platform"].isna(), "platform"] = "cpu"
    return df


def load_master(results_dirs):
    files = []
    for d in results_dirs:
        files += glob.glob(os.path.join(d, "**", "benchmark_*.csv"), recursive=True)
    files = [f for f in files if "_params_" not in f]
    rows = []
    for f in files:
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if d.empty:
            continue
        r = d.iloc[0].to_dict()
        r["_file"] = os.path.basename(f)
        rows.append(r)
    if not rows:
        raise SystemExit("no benchmark_*.csv found under: " + ", ".join(results_dirs))
    df = pd.DataFrame(rows)
    df = _normalize_provenance(df)
    # tidy: unified batch column (particles for SMC, chains otherwise), abs error
    # The batched axis is n_particles for SMC and n_chains for MCMC. It must be chosen
    # BY SAMPLER FAMILY, not by "whichever is non-null": --n_particles has a default of
    # 1000 that MCMC runs never use but still record, so a fillna() picks 1000 for every
    # MCMC cell and silently collapses all four chain counts (64/256/1024/4096) onto a
    # single x value.
    if "n_particles" in df or "n_chains" in df:
        npart = df["n_particles"] if "n_particles" in df else np.nan
        nchain = df["n_chains"] if "n_chains" in df else np.nan
        is_smc = df.get("sampler", pd.Series("", index=df.index)).astype(str).str.startswith("smc")
        df["batch"] = np.where(is_smc, npart, nchain)
        df["batch"] = pd.to_numeric(df["batch"], errors="coerce")
    df["abs_err"] = df.get("rmse_param_mean")
    df["ess_per_sec"] = df.get("min_ess_bulk_per_sec")
    keep = [c for c in _COLS + ["batch", "abs_err", "ess_per_sec", "_file"] if c in df]
    return df[keep].copy()


# Cells whose numbers WILL change once queued re-runs land. Marked with * in the table
# and spelled out in a footnote, so a provisional number is never mistaken for a final
# one. Keyed by (sampler, platform); remove an entry once its re-run is in.
_PROVISIONAL = {
    ("demc", "gpu"): r"200/500 budget; 1000/1000 re-run in progress",
    ("slice", "gpu"): r"15/30 budget, $\widehat{R}\approx2.95$ (not converged); "
                      r"50/100 re-run at a cheaper bracket in progress",
    ("smc_lik", "cpu"): r"$G^\star=0.7$ cells still running",
}


def _provisional_note(sub):
    """LaTeX footnote listing only the provisional groups actually present in `sub`."""
    present = {(r.get("sampler"), r.get("platform")) for _, r in sub.iterrows()}
    items = [(k, v) for k, v in _PROVISIONAL.items() if k in present]
    if not items:
        return []
    bits = [f"{_tt(s)} ({p}): {why}" for (s, p), why in sorted(items)]
    return [
        r"\\[2pt]",
        r"\multicolumn{7}{p{0.95\linewidth}}{\footnotesize $^{*}$Provisional --- final "
        r"results arriving. " + "; ".join(bits) + r".}\\",
    ]


def _tt(s):
    r"""Sampler slug as \texttt, with _ escaped -- a bare underscore in text mode is a
    LaTeX error ("Missing $ inserted"), and every SMC slug contains one."""
    return r"\texttt{" + str(s).replace("_", r"\_") + "}"


def _fmt(x, nd=3):
    """Format a number, rendering NaN/inf as -- (SMC cells have no R-hat: one 'chain')."""
    try:
        v = float(x)
    except Exception:
        return "--"
    return f"{v:.{nd}g}" if np.isfinite(v) else "--"


def latex_benchmark_table(df, out_tex, G=0.2, label="tab:benchmark"):
    """Accuracy + performance at a chosen true G: representative (largest) batch per
    (sampler, platform, which_stat)."""
    sub = df[np.isclose(df["G_true"].astype(float), G)] if "G_true" in df else df
    if sub.empty:
        sub = df
    # pick the largest batch per group
    sub = sub.sort_values("batch").groupby(
        ["sampler", "platform", "which_stat"], dropna=False).tail(1)
    sub = sub.sort_values(["sampler", "which_stat", "platform"])
    lines = [
        r"\begin{table}[t]\centering",
        r"\caption{Recovery accuracy and cost per cell (true $G^\star=%s$, largest "
        r"batch). $|\Delta G|=|\hat G-G^\star|$; ESS/s is budget-normalized throughput.}"
        % _fmt(G, 2),
        rf"\label{{{label}}}\small",
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Sampler & Backend & Feat. & $|\Delta G|$ & runtime (s) & ESS/s & $\widehat{R}$ \\",
        r"\midrule",
    ]
    for _, r in sub.iterrows():
        samp = r.get("sampler", "")
        star = "$^{*}$" if (samp, r.get("platform")) in _PROVISIONAL else ""
        lines.append(
            f"{_tt(samp)}{star} & {r.get('platform','')} & {r.get('which_stat','')} & "
            f"{_fmt(r.get('abs_err'))} & {_fmt(r.get('runtime_sec'),4)} & "
            f"{_fmt(r.get('ess_per_sec'),3)} & {_fmt(r.get('max_r_hat'),3)} \\\\")
    lines += [r"\bottomrule"] + _provisional_note(sub) + [r"\end{tabular}", r"\end{table}"]
    with open(out_tex, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out_tex} ({len(sub)} rows)")


def latex_settings_table(config, out_tex):
    try:
        import yaml
        with open(config) as fh:
            cfg = yaml.safe_load(fh)
    except Exception as e:
        print(f"(settings table skipped: {e})")
        return
    lines = [
        r"\begin{table}[t]\centering",
        r"\caption{Per-sampler budgets. The batched axis is the on-device \texttt{vmap} "
        r"dimension; comparison uses ESS/s since budgets differ.}",
        r"\label{tab:settings}\small",
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Sampler & Family & Batched axis (values) & Budget \\",
        r"\midrule",
    ]
    for name, s in cfg.get("samplers", {}).items():
        axis = s.get("batched_axis", "--")
        vals = s.get(axis, "")
        budget = []
        for k in ("n_stages", "n_mcmc", "n_warmup", "n_samples"):
            if k in s:
                budget.append(f"{k}={s[k]}")
        lines.append(
            f"{name} & {s.get('family','')} & {axis} {vals} & {', '.join(budget)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(out_tex, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out_tex}")


def fig_recovery_vs_G(df, out_png):
    """Posterior recovery |Delta G| vs true G.

    COLOUR = ALGORITHM, LINESTYLE = (feature x backend). Each sampler can appear up to
    four times -- {FC,FCD} x {CPU,GPU} -- and the previous version left colour to
    matplotlib's default cycle, which repeats after ten entries and assigned a fresh
    colour to every (sampler, feature, backend) triple. Different algorithms therefore
    shared a colour while one algorithm's four lines were four different colours,
    i.e. exactly backwards. Colour now follows the entity; the four variants of one
    algorithm are separated by dash pattern, with the headline case (FCD on GPU) solid.
    """
    d = df.dropna(subset=["G_true", "abs_err"])
    if d.empty:
        print("(recovery figure skipped: no data)"); return

    colors = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(SAMPLER_ORDER)}
    # (feature, platform) -> dash pattern. GPU heavier, CPU lighter.
    STYLES = {("FCD", "gpu"): "-", ("FC", "gpu"): "--",
              ("FCD", "cpu"): "-.", ("FC", "cpu"): ":"}

    # FACETED 2x2. On a single axes this is 6 algorithms x 4 (feature,backend) variants
    # = up to 24 crossing lines: the encoding is unambiguous but unreadable. One panel
    # per variant leaves <=6 lines each, so COLOUR ALONE identifies the algorithm and no
    # dash pattern has to be decoded. The linestyle is retained per panel so a single
    # extracted panel still says which variant it is.
    combos = [("FC", "gpu"), ("FCD", "gpu"), ("FC", "cpu"), ("FCD", "cpu")]
    combos = [c for c in combos
              if not d[(d["which_stat"] == c[0]) & (d["platform"] == c[1])].empty]
    ncol = 2
    nrow = int(np.ceil(len(combos) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.9 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, (stat, plat) in zip(axes, combos):
        ls = STYLES[(stat, plat)]
        for samp in SAMPLER_ORDER:
            g = d[(d["sampler"] == samp) & (d["which_stat"] == stat)
                  & (d["platform"] == plat)]
            if g.empty:
                continue
            # one point per G: the widest batch available for that cell
            g = g.sort_values("batch").groupby("G_true").tail(1).sort_values("G_true")
            ax.plot(g["G_true"].astype(float), g["abs_err"].astype(float), ls,
                    color=colors.get(samp, "#777777"), marker="o", ms=5.5, lw=1.8,
                    alpha=0.95, markeredgecolor="white", markeredgewidth=0.8)
        ax.set_yscale("log")
        ax.set_title(f"{stat}, {plat.upper()}", fontsize=10)
        ax.grid(True, ls=":", alpha=0.4)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[len(combos):]:
        ax.set_visible(False)

    for ax in axes[len(combos) - ncol:len(combos)]:
        ax.set_xlabel(r"true $G^\star$")
    for i in range(0, len(combos), ncol):
        axes[i].set_ylabel(r"$|\hat G - G^\star|$")

    from matplotlib.lines import Line2D
    alg = [Line2D([], [], color=colors[s], lw=2, marker="o", ms=5, label=s)
           for s in SAMPLER_ORDER if (d["sampler"] == s).any()]
    fig.legend(handles=alg, frameon=False, fontsize=9, ncol=len(alg),
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Posterior recovery across regimes", y=1.0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out_png}")


def fig_throughput(df, out_png):
    """ESS/sec vs batch, GPU vs CPU (log-log)."""
    d = df.dropna(subset=["batch", "ess_per_sec"])
    if d.empty:
        print("(throughput figure skipped: no data)"); return
    fig, ax = plt.subplots(figsize=(7, 5))
    for plat, g in d.groupby("platform"):
        g = g.sort_values("batch")
        ax.scatter(g["batch"].astype(float), g["ess_per_sec"].astype(float),
                   color=COLORS.get(plat, "#444"), label=plat, alpha=0.7)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("batch (particles / chains)"); ax.set_ylabel("ESS / sec")
    ax.set_title("Throughput vs batch, GPU vs CPU")
    ax.grid(True, which="both", ls=":", alpha=0.4); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out_png, dpi=300); plt.close(fig)
    print(f"wrote {out_png}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+",
                    default=[os.path.join(here, "..", "..", "results")])
    ap.add_argument("--config",
                    default=os.path.join(here, "..", "..", "config", "benchmark_config.yaml"))
    ap.add_argument("--out", default=os.path.join(here, "..", "..", "paper"))
    ap.add_argument("--table_G", type=float, default=0.2)
    ap.add_argument("--table_label", default="tab:benchmark",
                    help="LaTeX label for the benchmark table; set to match the "
                         "\\ref already used in main.tex")
    args = ap.parse_args()

    tables = os.path.join(args.out, "tables"); os.makedirs(tables, exist_ok=True)
    figs = os.path.join(args.out, "figures"); os.makedirs(figs, exist_ok=True)

    df = load_master(args.results)
    master = os.path.join(args.out, "master_results.csv")
    df.to_csv(master, index=False)
    print(f"wrote {master} ({len(df)} cells)")

    latex_benchmark_table(df, os.path.join(tables, "benchmark_table.tex"),
                          G=args.table_G, label=args.table_label)
    latex_settings_table(args.config, os.path.join(tables, "settings_table.tex"))
    fig_recovery_vs_G(df, os.path.join(figs, "recovery_vs_G.png"))
    fig_throughput(df, os.path.join(figs, "throughput_ess.png"))
    print("done.")


if __name__ == "__main__":
    main()
