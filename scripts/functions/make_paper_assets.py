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
_COLS = ["sampler", "sampler_raw", "framework", "platform", "which_stat", "G_true",
         "G_hat", "G_sd", "G_lo", "G_hi", "SC_size", "t_end", "n_chains", "n_draws", "n_warmup",
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
    # benchmark_comparison.csv is a multi-row AGGREGATE of several cells, not a run row.
    # The iloc[0] below would keep only its first sampler and silently discard the rest,
    # injecting a row that also duplicates a per-cell file we already read.
    files = [f for f in files if os.path.basename(f) != "benchmark_comparison.csv"]
    # One cell can sit under several results dirs -- a date dir may hold an earlier copy
    # of a sweep that a later dir supersedes (verified byte-identical). Recursive globbing
    # would count it once per copy, which biases the per-group pick below and any median.
    seen, uniq = set(), []
    for f in sorted(files):
        b = os.path.basename(f)
        if b not in seen:
            seen.add(b)
            uniq.append(f)
    files = uniq
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
        # mtime is preserved by rsync -a off the clusters, so it dates the run itself.
        # It is the final tie-break: "the last version of this cell that we produced".
        r["_mtime"] = os.path.getmtime(f)
        # The run also writes summary_<TAG>.csv beside benchmark_<TAG>.csv, holding the
        # ArviZ summary. The benchmark row records only rmse_param_mean = |Ghat - G*|,
        # which is unsigned -- so the posterior mean itself has to come from here if we
        # want to show recovery against the identity rather than an absolute error.
        sm = os.path.join(os.path.dirname(f),
                          os.path.basename(f).replace("benchmark_", "summary_", 1))
        if os.path.exists(sm):
            try:
                s = pd.read_csv(sm)
                s = s[s["parameter"].astype(str) == "G"] if "parameter" in s else s
                if not s.empty:
                    r["G_hat"] = float(s.iloc[0].get("mean", np.nan))
                    r["G_sd"] = float(s.iloc[0].get("sd", np.nan))
                    r["G_lo"] = float(s.iloc[0].get("hdi_3%", np.nan))
                    r["G_hi"] = float(s.iloc[0].get("hdi_97%", np.nan))
            except Exception:
                pass
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
        # pymc runs its SMC with n_draws as the particle population and never writes
        # n_particles, so those cells came out batch=NaN. NaN sorts last in pandas, so
        # such a row won (or lost) the per-group pick below by sort position rather than
        # on merit -- next to a numpyro CPU cell of 256 particles it silently won as if
        # it were the widest batch. Fall back to n_draws so the comparison is real.
        if "n_draws" in df:
            _nd = pd.to_numeric(df["n_draws"], errors="coerce")
            df["batch"] = df["batch"].where(~(is_smc & df["batch"].isna()), _nd)
    df["abs_err"] = df.get("rmse_param_mean")
    df["ess_per_sec"] = df.get("min_ess_bulk_per_sec")
    keep = [c for c in _COLS + ["batch", "abs_err", "ess_per_sec", "_file", "_mtime"]
            if c in df]
    return df[keep].copy()


# Cells the reader must not take at face value: either a queued re-run will change the
# number, or the run landed but did not converge. Marked with * in the table and spelled
# out in a footnote. Keyed by (sampler, platform); drop an entry once it is settled.
#
# Status 2026-08-03: the demc 1000/1000 GPU re-run landed ($\widehat{R}\le1.22$) and the
# smc_lik CPU $G^\star=0.7$ cells finished, so both entries are gone. The slice 50/100
# re-run also landed, but quadrupling the budget moved $\widehat{R}$ only 2.96 -> 2.56:
# it stays flagged, now as a converged-failure rather than a pending result.
_CAVEATS = {
    ("slice", "gpu"): r"does not converge even at budget parity. Given the same "
                      r"$1000/1000$ as DE-MC and RWMH --- $20\times$ the draws of the "
                      r"earlier $50/100$ cells and $31$\,h per cell --- $\widehat{R}$ "
                      r"falls only from $2.56$ to $2.17$ (FCD) and $1.79$ to $1.27$ "
                      r"(FC). The earlier budget confound is therefore resolved against "
                      r"budget: this is the proposal",
}


def _provisional_note(sub, ncols=7):
    """LaTeX footnote listing only the caveated groups actually present in `sub`.

    ncols must match the enclosing tabular: a \multicolumn wider or narrower than the
    table is a LaTeX alignment error, and the two tables here have different widths.
    """
    present = {(r.get("sampler"), r.get("platform")) for _, r in sub.iterrows()}
    items = [(k, v) for k, v in _CAVEATS.items() if k in present]
    if not items:
        return []
    bits = [f"{_tt(s)} ({p}): {why}" for (s, p), why in sorted(items)]
    return [
        r"\\[2pt]",
        rf"\multicolumn{{{ncols}}}{{p{{0.95\linewidth}}}}{{\footnotesize $^{{*}}$Not to "
        r"be read as a settled number. " + "; ".join(bits) + r".}\\",
    ]


def _tt(s):
    r"""Sampler slug as \texttt, with _ escaped -- a bare underscore in text mode is a
    LaTeX error ("Missing $ inserted"), and every SMC slug contains one."""
    return r"\texttt{" + str(s).replace("_", r"\_") + "}"


# The benchmark tables describe ONE configuration: the 10-node subnetwork at
# t_end=30000. The 88-node and long-t_end scaling runs share G*=0.2 with it, so without
# this filter they land in the same (sampler, platform, feature) group and compete for
# the per-cell pick -- silently mixing network sizes into a table that never says which
# network it is. They are reported separately instead.
BENCH_SC_SIZE = 10
BENCH_T_END = 30000


def bench_subset(df):
    """Rows belonging to the main benchmark configuration."""
    out = df
    if "SC_size" in out:
        out = out[out["SC_size"].astype(float).fillna(BENCH_SC_SIZE) == BENCH_SC_SIZE]
    if "t_end" in out:
        out = out[out["t_end"].astype(float).fillna(BENCH_T_END) == BENCH_T_END]
    return out if len(out) else df


def _latest_per_cell(sub, keys):
    """One representative run per cell: widest batch, then longest budget, then newest.

    Every step is needed. Batch alone is the reporting convention (a cell is swept over
    batch widths), but a cell is also re-run at several BUDGETS at the same width -- a
    short exploratory pass and a longer re-run that supersedes it. Sorting on batch alone
    left that tie to glob order, which kept the superseded short run, so a landed re-run
    never reached the table. mtime settles anything budget cannot.

    na_position='first' matters: a row missing one of these keys must never win the pick
    by NaN sort position, which is exactly how a pymc SMC cell (no n_particles recorded)
    used to beat a real numpyro cell beside it.
    """
    by = [c for c in ("batch", "n_draws", "n_warmup", "_mtime") if c in sub]
    return sub.sort_values(by, na_position="first").groupby(keys, dropna=False).tail(1)


def _fmt(x, nd=3):
    """Format a number, rendering NaN/inf as -- (SMC cells have no R-hat: one 'chain')."""
    try:
        v = float(x)
    except Exception:
        return "--"
    return f"{v:.{nd}g}" if np.isfinite(v) else "--"


def _fmt_rhat(x, nd=3):
    r"""R-hat, distinguishing "not measured" from "diverged".

    _fmt collapses NaN and inf to the same "--", but for R-hat they mean opposite
    things: NaN is a single-population SMC cell that has no R-hat to report, while an
    infinite R-hat is a chain that blew up. Printing the failure as "--" hides it behind
    the caption's "not run" legend, so infinities are shown as such.
    """
    try:
        v = float(x)
    except Exception:
        return "--"
    if np.isnan(v):
        return "--"
    return f"{v:.{nd}g}" if np.isfinite(v) else r"$\infty$"


def latex_benchmark_table(df, out_tex, G=0.2, label="tab:benchmark"):
    """Accuracy + performance at a chosen true G: representative (largest) batch per
    (sampler, platform, which_stat)."""
    df = bench_subset(df)
    sub = df[np.isclose(df["G_true"].astype(float), G)] if "G_true" in df else df
    if sub.empty:
        sub = df
    sub = _latest_per_cell(sub, ["sampler", "platform", "which_stat"])
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
        star = "$^{*}$" if (samp, r.get("platform")) in _CAVEATS else ""
        lines.append(
            f"{_tt(samp)}{star} & {r.get('platform','')} & {r.get('which_stat','')} & "
            f"{_fmt(r.get('abs_err'))} & {_fmt(r.get('runtime_sec'),4)} & "
            f"{_fmt(r.get('ess_per_sec'),3)} & {_fmt_rhat(r.get('max_r_hat'),3)} \\\\")
    lines += [r"\bottomrule"] + _provisional_note(sub) + [r"\end{tabular}", r"\end{table}"]
    with open(out_tex, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out_tex} ({len(sub)} rows)")


def latex_full_grid_table(df, out_tex, label="tab:fullgrid"):
    r"""Every cell we have: sampler x {FC,FCD} x {cpu,gpu} x every $G^\star$ tested.

    tab:accuracy shows one $G^\star$ so it stays readable in the body; this is the
    complete grid, one row per (sampler, backend, feature) and one column block per
    $G^\star$, each holding the LATEST version of that cell (see _latest_per_cell).
    A cell we never ran prints as "--", so gaps in the sweep are visible rather than
    inferred from a missing row.
    """
    df = bench_subset(df)
    Gs = sorted(df["G_true"].dropna().astype(float).unique())
    sub = _latest_per_cell(df, ["sampler", "platform", "which_stat", "G_true"])
    idx = sub.set_index(["sampler", "platform", "which_stat",
                         sub["G_true"].astype(float).round(4)])
    cells = sorted({(r.sampler, r.platform, r.which_stat) for r in sub.itertuples()})

    lines = [
        r"\begin{table*}[t]\centering",
        r"\caption{Every measured cell, at the latest version of each. Rows are "
        r"sampler $\times$ backend $\times$ feature; each $G^\star$ block gives "
        r"$|\Delta G|=|\hat G-G^\star|$ and $\widehat{R}$ (``--'' = not run, or no "
        r"$\widehat{R}$ for single-population SMC). Representative batch per cell: "
        r"widest batch, longest budget, most recent run.}",
        # 3 + 2*len(Gs) columns overflows a one-column elsarticle body at \small with
        # default padding (~100pt too wide at four G values), so trim both.
        rf"\label{{{label}}}\footnotesize\setlength{{\tabcolsep}}{{3.5pt}}",
        r"\begin{tabular}{lll" + "rr" * len(Gs) + "}",
        r"\toprule",
        r"& & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$G^\star={_fmt(G,2)}$}}"
                               for G in Gs) + r" \\",
        r"Sampler & Backend & Feat. & " +
        " & ".join([r"$|\Delta G|$ & $\widehat{R}$"] * len(Gs)) + r" \\",
        r"\midrule",
    ]
    for samp, plat, feat in cells:
        star = "$^{*}$" if (samp, plat) in _CAVEATS else ""
        vals = []
        for G in Gs:
            try:
                r = idx.loc[(samp, plat, feat, round(float(G), 4))]
                if isinstance(r, pd.DataFrame):
                    r = r.iloc[-1]
                vals += [_fmt(r.get("abs_err")), _fmt_rhat(r.get("max_r_hat"), 3)]
            except KeyError:
                vals += ["--", "--"]
        lines.append(f"{_tt(samp)}{star} & {plat} & {feat} & " + " & ".join(vals) + r" \\")
    lines += ([r"\bottomrule"] + _provisional_note(sub, ncols=3 + 2 * len(Gs))
              + [r"\end{tabular}", r"\end{table*}"])
    with open(out_tex, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out_tex} ({len(cells)} rows x {len(Gs)} G values)")


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
    # Plot the ESTIMATE against the truth, not the absolute error. |Ghat - G*| = 0.1 is
    # unreadable on its own -- it is 50% of G*=0.2 but 14% of G*=0.7 -- whereas distance
    # from the identity line is directly interpretable, and the sign shows whether a
    # sampler over- or under-estimates the coupling.
    ycol = "G_hat" if ("G_hat" in df and df["G_hat"].notna().any()) else "abs_err"
    d = df.dropna(subset=["G_true", ycol])
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

    gs = sorted(d["G_true"].dropna().astype(float).unique())
    lim = (min(gs) - 0.08, max(gs) + 0.12)
    for ax, (stat, plat) in zip(axes, combos):
        ls = STYLES[(stat, plat)]
        # identity: perfect recovery. Distance from it IS the error, to scale.
        ax.plot(lim, lim, "-", color="#999999", lw=1.2, zorder=1)
        # one point per G: the widest batch available for that cell
        series = {}
        for samp in SAMPLER_ORDER:
            g = d[(d["sampler"] == samp) & (d["which_stat"] == stat)
                  & (d["platform"] == plat)]
            if g.empty:
                continue
            series[samp] = (g.sort_values("batch").groupby("G_true").tail(1)
                             .sort_values("G_true"))

        # No +/-1 sd shading. It was tried and removed: the poorly-converged cells have
        # posterior sd comparable to the prior width (GPU rwmh on FCD at G*=0.2: sd 0.82
        # about a mean of 0.37), so their bands cover the entire panel and bury the tight
        # SMC bands regardless of draw order or alpha. The dispersion is reported per
        # cell in master_results.csv (G_sd, G_lo, G_hi) and via R-hat in Table 1, where
        # it is legible; here it destroyed the figure it was meant to enrich.
        # Nudge each sampler by a small x-offset. Several samplers agree to within a
        # line width -- smc_lik and smc_abc are nearly identical on FCD -- and drawn at
        # the true x they hide one another completely, so a reader sees five lines where
        # there are six. The offset is a presentation device only: it is ~1% of the G
        # range, far below any difference being claimed, and the markers still sit at
        # their own G. Stated in the caption.
        present = list(series)
        span = 0.008
        for i, samp in enumerate(present):
            g = series[samp]
            dx = (i - (len(present) - 1) / 2.0) * span
            ax.plot(g["G_true"].astype(float) + dx, g[ycol].astype(float), ls,
                    color=colors.get(samp, "#777777"), marker="o", ms=5.5, lw=1.8,
                    alpha=0.95, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        ax.set_title(f"{stat}, {plat.upper()}", fontsize=10)
        ax.set_xlim(*lim); ax.set_ylim(*lim)
        ax.set_xticks(gs); ax.set_yticks(gs)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, ls=":", alpha=0.4)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[len(combos):]:
        ax.set_visible(False)

    ylab = r"posterior mean $\hat G$" if ycol == "G_hat" else r"$|\hat G - G^\star|$"
    for ax in axes[len(combos) - ncol:len(combos)]:
        ax.set_xlabel(r"true $G^\star$")
    for i in range(0, len(combos), ncol):
        axes[i].set_ylabel(ylab)

    from matplotlib.lines import Line2D
    alg = [Line2D([], [], color=colors[s], lw=2, marker="o", ms=5, label=s)
           for s in SAMPLER_ORDER if (d["sampler"] == s).any()]
    fig.legend(handles=alg, frameon=False, fontsize=9, ncol=len(alg),
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
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
    latex_full_grid_table(df, os.path.join(tables, "full_grid_table.tex"))
    latex_settings_table(args.config, os.path.join(tables, "settings_table.tex"))
    fig_recovery_vs_G(df, os.path.join(figs, "recovery_vs_G.png"))
    fig_throughput(df, os.path.join(figs, "throughput_ess.png"))
    print("done.")


if __name__ == "__main__":
    main()
