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
import argparse, glob, os, re
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
# 7th slot (#7d4bc4) added for smc_abc_demc. Validated with the dataviz palette checker
# at 7 slots, light surface: lightness band, chroma floor, CVD separation and
# normal-vision floor all PASS, and the purple does not become the worst adjacent pair
# (that remains the pre-existing #eda100/#1baf7a at dE 9.1 protan).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7d4bc4"]
SAMPLER_ORDER = ["smc_abc", "smc_lik", "demc", "rwmh", "slice", "demcz", "smc_abc_demc"]


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
    # NB: epsilon-bearing ABC labels are handled by _canon_abc below, not here -- the
    # literal key only ever matched epsilon=10.
}


def _canon_abc(name):
    r"""Map pymc's ABC display label to a slug that ENCODES whether epsilon was calibrated.

    pymc writes the sampler as "SMC, $\epsilon$ = <value>". Keying the alias table on the
    literal "= 10" meant any other epsilon matched nothing and silently dropped out of
    every table and figure -- so a calibrated re-run would have been invisible.

    Calibrated and uncalibrated runs must NOT collapse to one slug: they are different
    configurations of the same sampler and would then compete for the same cell, with the
    winner decided by batch width rather than by which one the text is discussing. The
    ABC kernel is -0.5*((obs-sim)/eps)^2, identical in form to a Gaussian log-likelihood
    of sd eps, so eps == obs_err (here 1.0) is the calibrated choice and anything looser
    is a distinct, weaker configuration.
    """
    m = re.match(r"^SMC,\s*\$?\\?epsilon\$?\s*=\s*([0-9.]+)$", str(name).strip())
    if not m:
        return None
    eps = float(m.group(1))
    return "smc_abc" if abs(eps - 1.0) < 1e-9 else f"smc_abc_eps{m.group(1).rstrip('.')}"


def _normalize_provenance(df):
    """Canonicalise sampler names and backfill platform/framework across frameworks."""
    if "sampler" in df:
        df["sampler_raw"] = df["sampler"]
        df["sampler"] = df["sampler"].replace(_SAMPLER_ALIASES)
        # epsilon-bearing ABC labels carry their calibration in the slug
        _abc = df["sampler"].map(_canon_abc)
        df["sampler"] = _abc.where(_abc.notna(), df["sampler"])
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


def latex_master_table(df, out_tex, cost_G=0.2, label="tab:accuracy"):
    r"""THE benchmark table: accuracy across every $G^\star$ AND cost, in one place.

    Supersedes the tab:accuracy / tab:fullgrid pair. Those had identical rows and split
    the columns between them -- one showed cost at a single $G^\star$, the other showed
    accuracy at all of them -- so every benchmark update meant regenerating two tables
    and keeping two captions honest. Here each row is one (sampler, backend, feature)
    cell, with a $|\Delta G|$/$\widehat{R}$ block per $G^\star$ followed by cost.

    Cost is reported at ONE coupling (`cost_G`), because runtime and ESS/s are a property
    of the budget, not of $G^\star$, and repeating them per block would quadruple the
    width to restate the same number. The caption says which coupling, so this is a stated
    convention rather than a silent one.

    Runtime is in kiloseconds: `3.551e+04` is four characters wider than `35.5` in every
    row, and at 13 columns that difference decides whether the table fits the page.
    """
    df = bench_subset(df)
    Gs = sorted(df["G_true"].dropna().astype(float).unique())
    sub = _latest_per_cell(df, ["sampler", "platform", "which_stat", "G_true"])
    idx = sub.set_index(["sampler", "platform", "which_stat",
                         sub["G_true"].astype(float).round(4)])
    cells = sorted({(r.sampler, r.platform, r.which_stat) for r in sub.itertuples()})
    ncols = 3 + 2 * len(Gs) + 2

    lines = [
        r"\begin{table*}[t]\centering",
        r"\caption{Gradient-free benchmark: every measured cell, at the latest version "
        r"of each. Rows are sampler $\times$ backend $\times$ feature; each $G^\star$ "
        r"block gives $|\Delta G|=|\hat G-G^\star|$ and $\widehat{R}$. Cost (runtime, "
        r"ESS/s) is reported at $G^\star=%s$, since it is set by the budget rather than "
        r"by the coupling. ``--'' = not run, or no $\widehat{R}$ for single-population "
        r"SMC. Representative run per cell: widest batch, longest budget, most recent.}"
        % _fmt(cost_G, 2),
        rf"\label{{{label}}}\footnotesize\setlength{{\tabcolsep}}{{3pt}}",
        r"\begin{tabular}{lll" + "rr" * len(Gs) + "rr}",
        r"\toprule",
        r"& & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$G^\star={_fmt(G,2)}$}}"
                               for G in Gs)
        + rf" & \multicolumn{{2}}{{c}}{{Cost at $G^\star={_fmt(cost_G,2)}$}} \\",
        r"Sampler & Backend & Feat. & " +
        " & ".join([r"$|\Delta G|$ & $\widehat{R}$"] * len(Gs)) +
        r" & runtime (ks) & ESS/s \\",
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
        # cost block: same cell, at the reference coupling only
        try:
            rc = idx.loc[(samp, plat, feat, round(float(cost_G), 4))]
            if isinstance(rc, pd.DataFrame):
                rc = rc.iloc[-1]
            rt = rc.get("runtime_sec")
            rt_ks = float(rt) / 1000.0 if rt is not None and np.isfinite(
                pd.to_numeric(rt, errors="coerce")) else np.nan
            vals += [_fmt(rt_ks, 3), _fmt(rc.get("ess_per_sec"), 3)]
        except KeyError:
            vals += ["--", "--"]
        lines.append(f"{_tt(samp)}{star} & {plat} & {feat} & " + " & ".join(vals) + r" \\")
    lines += ([r"\bottomrule"] + _provisional_note(sub, ncols=ncols)
              + [r"\end{tabular}", r"\end{table*}"])
    with open(out_tex, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out_tex} ({len(cells)} rows x {len(Gs)} G values + cost)")


_METRICS = [
    # (header, csv column, formatter). THE column list: add or remove a line and the
    # subtables follow, no other edit needed.
    #
    # Selected 2026-08-06 from the full menu the runs record. Dropped, with reasons:
    #   G_hat          -- near-derivable from G* and |dG|; only the SIGN of the bias is
    #                     lost, and no claim currently rests on it. (Reporting a SIGNED
    #                     dG instead of |dG| would carry both in one column, at the cost
    #                     of changing a convention the caption and prose already fix.)
    #   coverage_94hdi -- 259 of 263 benchmark cells identical (all check); with one run
    #                     per cell it is a single yes/no, not a coverage RATE.
    #   min_ess_bulk   -- Spearman 0.79 with ess_per_sec, which is the budget-normalised
    #                     one the text actually argues from.
    #
    # HDI is kept DESPITE Spearman 0.88 with sd: the ratio HDI/sd has median 2.81 against
    # the Gaussian 3.75, and 130 of 260 cells sit >25% off that, so these posteriors are
    # not Gaussian and HDI is not a rescaled sd -- it reports shape.
    # runtime is kept because ESS/s is a RATE and cannot reconstruct a duration; the
    # paper's cost argument quotes durations ("31 h per cell", "0.75-2.1 ks vs 15-50 ks").
    (r"$|\Delta G|$",          "abs_err",        lambda r: _fmt(r.get("abs_err"), 3)),
    (r"sd",                    "G_sd",           lambda r: _fmt(r.get("G_sd"), 3)),
    (r"HDI$_{94}$",            "G_lo",           lambda r: _hdi_width(r)),
    (r"$\widehat{R}$",         "max_r_hat",      lambda r: _fmt_rhat(r.get("max_r_hat"), 3)),
    (r"ESS/s",                 "ess_per_sec",    lambda r: _fmt(r.get("ess_per_sec"), 3)),
    (r"time (ks)",             "runtime_sec",    lambda r: _ks(r.get("runtime_sec"))),
    (r"batch",                 "batch",          lambda r: _fmt(r.get("batch"), 4)),
]


def _hdi_width(r):
    try:
        w = float(r.get("G_hi")) - float(r.get("G_lo"))
        return f"{w:.3g}" if np.isfinite(w) else "--"
    except Exception:
        return "--"


def _cov(r):
    r"""94% HDI coverage as a check/cross. It is 0/1 per cell, not a rate: with one run
    per cell there is nothing to average, so printing `0.0`/`1.0` invites reading it as a
    frequency."""
    try:
        v = float(r.get("coverage_94hdi"))
    except Exception:
        return "--"
    if not np.isfinite(v):
        return "--"
    return r"\checkmark" if v >= 0.5 else r"$\times$"


def _ks(x):
    """Runtime in kiloseconds -- see latex_master_table on why not seconds."""
    try:
        v = float(x)
    except Exception:
        return "--"
    return f"{v/1000.0:.3g}" if np.isfinite(v) else "--"


def _g_slug(G):
    """G* as a label-safe slug: 0.33 -> 033. A '.' in a \label is legal but a menace to
    grep for, and hyperref renders it inconsistently across backends."""
    return _fmt(G, 2).replace(".", "").replace("-", "m")


def latex_config_subtables(df, out_tex, Gs, label_stem="tab:cells", intro=None):
    r"""Per-configuration metric tables: ONE numbered table per $G^\star$, with the four
    {FC,FCD} x {cpu,gpu} configurations as parts (a)-(d).

    Backend and feature are the identity of each part rather than columns, which frees the
    width for the metrics. $G^\star$ is fixed per table, so each part is 6 sampler rows
    instead of 24. Grouping the four into one float means the text cites a single table
    number, and -- the bigger win -- the shared column conventions are stated ONCE in the
    parent caption instead of being repeated four times with four chances to drift apart.

    The G-dependence is not lost: `fig_recovery_vs_G` is faceted 2x2 over exactly this
    configuration grid and shows every coupling, so the figure carries the regime story
    and these tables carry the per-cell detail. Couplings not in the body get the identical
    treatment in the supplement -- hence `Gs` is a list, called once per destination file.
    """
    df = bench_subset(df)
    # framework in the key: see the row loop below -- JAX-on-CPU and PyMC-on-CPU are
    # distinct implementations and must not evict one another.
    sub = _latest_per_cell(df, ["sampler", "framework", "platform", "which_stat", "G_true"])
    out = list(intro) if intro else []

    for G in Gs:
        at_G = sub[np.isclose(sub["G_true"].astype(float), float(G))]
        if at_G.empty:
            continue
        blocks, caveats = [], set()
        for feat in ("FC", "FCD"):
            for plat in ("cpu", "gpu"):
                block = at_G[(at_G["which_stat"] == feat) & (at_G["platform"] == plat)]
                if block.empty:
                    continue
                rows = []
                for samp in sorted(block["sampler"].unique()):
                    star = ""
                    if (samp, plat) in _CAVEATS:
                        star = "$^{*}$"
                        caveats.add((samp, plat))
                    bs = block[block["sampler"] == samp]
                    # A CPU block can hold the SAME sampler under two frameworks -- the
                    # same-code JAX-on-CPU run and the PyMC one. Those are different
                    # implementations, so `.iloc[-1]` silently dropped one of them at
                    # whatever order the frame happened to be in; it hit exactly the three
                    # G*=0.2 SMC cells the same-code CPU/GPU comparison is argued from.
                    # Emit both, and name the framework only where both are present, so
                    # the other blocks are untouched.
                    fws = list(dict.fromkeys(bs["framework"].astype(str)))
                    for fw in fws:
                        r = bs[bs["framework"].astype(str) == fw].iloc[-1]
                        tag = ""
                        if len(fws) > 1:
                            tag = r" \textsubscript{JAX}" if not fw.startswith("pymc") \
                                  else r" \textsubscript{PyMC}"
                        rows.append(f"{_tt(samp)}{tag}{star} & "
                                    + " & ".join(f(r) for _, _, f in _METRICS) + r" \\")
                blocks.append((feat, plat, rows))
        if not blocks:
            continue

        out += [
            r"\begin{table}[p]\centering",
            rf"\caption{{Per-run metrics at $G^\star={_fmt(G,2)}$, one part per feature "
            rf"and backend, at the latest version of each run. "
            rf"$|\Delta G|=|\hat G-G^\star|$; sd and HDI$_{{94}}$ describe the posterior "
            rf"width; ESS/s is budget-normalized throughput; batch is the parallel width "
            rf"(chains or particles) the cell was run at. ``--'' = not run, or not defined "
            rf"for that sampler (single-population SMC has no $\widehat{{R}}$).}}",
            rf"\label{{{label_stem}:{_g_slug(G)}}}",
        ]
        for i, (feat, plat, rows) in enumerate(blocks):
            out += [
                r"\begin{subtable}{\linewidth}\centering",
                rf"\caption{{{feat} on {plat.upper()}}}",
                rf"\label{{{label_stem}:{_g_slug(G)}:{feat}:{plat}}}",
                r"\footnotesize\setlength{\tabcolsep}{4pt}",
                r"\begin{tabular}{l" + "r" * len(_METRICS) + "}",
                r"\toprule",
                r"Sampler & " + " & ".join(h for h, _, _ in _METRICS) + r" \\",
                r"\midrule",
            ] + rows + [r"\bottomrule", r"\end{tabular}", r"\end{subtable}"]
            if i < len(blocks) - 1:
                out.append(r"\vspace{7pt}")
        # One caveat note for the whole table rather than one per part: the flagged cells
        # recur across parts and repeating the footnote four times reads as four problems.
        if caveats:
            bits = "; ".join(f"{_tt(s)} ({p}): {_CAVEATS[(s, p)]}"
                             for (s, p) in sorted(caveats))
            out += [r"\vspace{4pt}",
                    r"{\footnotesize $^{*}$Not to be read as a settled number. "
                    + bits + r".}"]
        out += [r"\end{table}", ""]

    with open(out_tex, "w") as fh:
        fh.write("\n".join(out) + "\n")
    n = sum(1 for l in out if l.startswith(r"\begin{table}"))
    m = sum(1 for l in out if l.startswith(r"\begin{subtable}"))
    print(f"wrote {out_tex} ({n} tables x {m//max(n,1)} parts, "
          f"G*={[_fmt(g,2) for g in Gs]}, {len(_METRICS)} metrics)")


# smc_abc_demc is the DE-MC proposal VARIANT of smc_abc, not a seventh algorithm. It was
# absent from every figure because they iterate SAMPLER_ORDER, which lists six entries --
# while the tables include it, so figures and tables disagreed on which cells exist.
# Giving it a seventh hue would mean an unvalidated palette step; instead it shares the
# ABC hue (same entity) and is separated by marker (different kernel), which is the
# composite encoding a variant calls for.
# Marker encodes BACKEND+FRAMEWORK, matching fig:smc_scaling, so the same visual channel
# means the same thing in every figure. It previously encoded sampler variants here
# (backend being the facet), which left a reader decoding two different grammars three
# figures apart. Variants now get their own validated hue instead; and once JAX-on-CPU
# cells exist, a CPU panel genuinely contains two frameworks, so this channel earns its
# keep rather than being constant within a panel.
#   filled circle = JAX on GPU
#   open circle   = JAX on CPU  (same code, different device)
#   cross         = PyMC on CPU (different framework entirely)
def _backend_style(framework, platform):
    fw, plat = str(framework), str(platform)
    # Same marker vocabulary as Fig. 4 (plot_ess_scaling.py): lowercase "x" for PyMC and a
    # heavy open circle for same-code JAX-on-CPU. The two figures had drifted -- "X" here,
    # "x" there -- so a reader carrying the key across figures had to re-learn it.
    if fw.startswith("pymc"):
        return dict(marker="x", markerfacecolor=None, markeredgecolor=None,
                    markeredgewidth=1.8, ms=7)
    if plat == "cpu":
        return dict(marker="o", markerfacecolor="none", markeredgecolor=None,
                    markeredgewidth=2.0, ms=9)
    return dict(marker="o", markerfacecolor=None, markeredgecolor="white",
                markeredgewidth=1.2, ms=8)


def _backend_legend_handles():
    mk = lambda **kw: plt.Line2D([], [], linestyle="none", color="#666666", **kw)
    return [
        (mk(marker="o", ms=8, markeredgecolor="white", markeredgewidth=1.2),
         "JAX, GPU"),
        (mk(marker="o", ms=9, markerfacecolor="none", markeredgecolor="#666666",
            markeredgewidth=2.0), "JAX, CPU (same code)"),
        (mk(marker="x", ms=7, markeredgewidth=1.8),
         "PyMC, CPU (other framework)"),
    ]


# smc_abc_eps10 is the SAME sampler as smc_abc at a looser tolerance, so it takes the ABC
# hue rather than a seventh-plus palette step or the grey fallback (grey reads as "other",
# which it is not). The marker already separates it: every eps=10 cell is pymc, hence a
# cross, while the calibrated cells are JAX circles.
_HUE_ALIAS = {"smc_abc_eps10": "smc_abc"}


def _hue(samp, colors):
    return colors.get(_HUE_ALIAS.get(samp, samp), "#777777")


# Aliased hues let a variant share its parent's colour, which is right -- it IS the same
# algorithm -- but only if the legend then says which variant is which. Without this the
# uncalibrated epsilon=10 ABC and the calibrated epsilon=1 ABC were drawn in one blue with
# a single "smc_abc" key, so the flat prior-collapse line read as the algorithm failing
# rather than as one tolerance setting failing.
_SERIES_LABEL = {"smc_abc_eps10": r"smc_abc, $\epsilon$=10 (uncalibrated)",
                 "smc_abc": r"smc_abc, $\epsilon$=1"}

# Series drawn faded + dashed: shown for CONTRAST, not part of the claim. Fig. 4 uses the
# same code for the off-budget slice points. Module scope because Figs. 5, 6 and 7 must
# agree on which series that is.
FADED = {"smc_abc_eps10"}


def _series_label(samp, framework=""):
    return _SERIES_LABEL.get(samp, samp)


def _samplers_present(d):
    """SAMPLER_ORDER first (fixed hue order), then any variant actually in the data."""
    present = set(d["sampler"].dropna().unique())
    return [s for s in SAMPLER_ORDER if s in present] + \
           [s for s in sorted(present - set(SAMPLER_ORDER))]


def _shared_legend(fig, axes, extra=None, ncol=None, fontsize=None):
    """Legend built from EVERY panel, ordered by SAMPLER_ORDER.

    Taking handles from the first panel alone silently drops any series absent there --
    demcz is CPU-only and smc_abc_demc GPU-only, so both were plotted with a colour and
    no legend entry, which is identity-by-colour-alone with no key.
    """
    seen, handles, labels = {}, [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l and l not in seen:
                seen[l] = h
    # Order by SAMPLER_ORDER, matched through _series_label: the plotted labels are the
    # display names ("smc_abc, eps=1"), so comparing them to the raw sampler keys sent
    # every renamed series to the unordered tail.
    order = [_series_label(s) for s in SAMPLER_ORDER] + \
            sorted(set(seen) - {_series_label(s) for s in SAMPLER_ORDER})
    for lab in order:
        if lab not in seen:
            continue
        h = seen[lab]
        # Normalise the marker. These handles are lifted from the plot, so each entry
        # wore whichever backend it happened to be drawn with -- demcz showed a PyMC
        # cross while its neighbours showed discs, which reads as a backend claim. The
        # backend key alongside carries that channel; here only hue (and the faded/dashed
        # "for contrast" flag) is meaningful.
        handles.append(plt.Line2D([], [], color=h.get_color(), marker="o", ms=6,
                                  markeredgecolor="white",
                                  linestyle=h.get_linestyle() if h.get_linestyle() != "None"
                                            else "-",
                                  lw=1.8, alpha=h.get_alpha() or 1.0))
        labels.append(lab)
    if extra:
        for h, l in extra:
            handles.append(h); labels.append(l)
    # ncol matters more than it looks. The legend is laid out INSIDE the saved bounding
    # box, so a single wide row makes the image wide, and scaling that to a fixed column
    # width shrinks the panels. On the compact per-coupling copies the caller passes a
    # small ncol so the key wraps and the data keeps the width.
    fig.legend(handles, labels, loc="lower center",
               ncol=min(ncol or 7, len(labels)),
               fontsize=fontsize, frameon=False, bbox_to_anchor=(0.5, -0.02))


def _facet_axes(nrow=2, ncol=2, size=(5.2, 4.1)):
    fig, axes = plt.subplots(nrow, ncol, figsize=(size[0] * ncol, size[1] * nrow),
                             sharex=True, sharey=True)
    return fig, np.atleast_1d(axes).ravel()


def _cells_by_config(df):
    """One row per benchmark cell, on the SAME subset+pick as the tables and Fig. 5.

    Every figure that shows benchmark cells must route through here. Fig. 5 previously
    re-picked its own representative run and silently disagreed with the tables on 4 of
    90 cells; sharing this function is what makes agreement structural rather than lucky.

    `framework` belongs in the key: JAX-on-CPU and PyMC-on-CPU are different
    implementations, and without it one evicted the other in the three G*=0.2 SMC cells
    the same-code CPU/GPU comparison is argued from. Figs. 6 and 7 then grouped by
    framework for styling, but on data one framework had already been dropped from -- so
    the grouping could not put back what the pick had removed.
    """
    return _latest_per_cell(bench_subset(df),
                            ["sampler", "framework", "platform", "which_stat", "G_true"])


def fig_accuracy_vs_cost(df, out_png, G=None):
    r"""Accuracy against cost: the frontier the benchmark is actually about.

    The manuscript argues accuracy (Fig. recovery) and cost (Fig. ess scaling) in two
    separate figures that never share axes, so "GPU-batched SMC dominates" is something
    the reader has to assemble. Here each cell is one point, |dG| against wall-clock, so
    domination is visible as position: down-left is better on both.

    Log-log because both spread over orders of magnitude (GPU: 386x in error, 159x in
    runtime).

    Two modes:

      G=None  : every coupling drawn, so a sampler contributes one point per G*. Since
                runtime is set by the budget and not by the coupling, those points stack
                near-vertically and the height of the stack IS the sampler's accuracy
                variability across regimes at fixed cost. An on-figure note says so --
                that reading was previously left to the caption.
      G=<val> : a single coupling. Called once per G* so each per-coupling table has a
                figure showing exactly its cells.

    Aggregating the four couplings into a median with range bars was tried and reverted:
    it compressed the very spread this figure exists to show, and the error-bar treatment
    belongs in Fig. 4, where the batch sweep is the x-axis and G* is genuinely a nuisance
    dimension.
    """
    d = _cells_by_config(df).dropna(subset=["abs_err", "runtime_sec"])
    d = d[(d["abs_err"] > 0) & (d["runtime_sec"] > 0)]
    if d.empty:
        print("(accuracy-vs-cost figure skipped: no data)"); return
    # Frame fixed on the POOLED data, so every per-coupling copy shares one set of axes
    # with the main-text figure. Left to autoscale, each page would rescale to its own
    # cells and a reader paging through the supplement would compare positions that are
    # not comparable -- the whole point of this figure being positional.
    xlim = (float(d["runtime_sec"].min()) / 1e3 * 0.6,
            float(d["runtime_sec"].max()) / 1e3 * 1.7)
    ylim = (float(d["abs_err"].min()) * 0.5, float(d["abs_err"].max()) * 2.0)
    if G is not None:
        d = d[np.isclose(d["G_true"].astype(float), float(G))]
        if d.empty:
            print(f"(accuracy-vs-cost figure skipped: no data at G*={G})"); return
    n_G = d["G_true"].dropna().nunique()
    colors = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(SAMPLER_ORDER)}
    combos = [("FC", "gpu"), ("FCD", "gpu"), ("FC", "cpu"), ("FCD", "cpu")]
    combos = [c for c in combos
              if not d[(d["which_stat"] == c[0]) & (d["platform"] == c[1])].empty]
    # A smaller CANVAS for the per-coupling copies. Matplotlib font sizes are absolute
    # points, so shrinking the figure and printing it at the same width makes the text
    # relatively LARGER -- which is what these need, since the supplement prints two of
    # them side by side at half the text width.
    fig, axes = _facet_axes(2, 2, size=(4.1, 3.5) if G is not None else (5.2, 4.1))
    for ax, (stat, plat) in zip(axes, combos):
        sub = d[(d["which_stat"] == stat) & (d["platform"] == plat)]
        # Samplers whose cost is set by the same budget land on the SAME wall-clock: on
        # FC/GPU smc_abc, smc_lik and smc_abc_demc all sit at ~0.7 ks and three of them
        # vanished under the fourth. Nudge each by a fixed multiplicative step -- the axis
        # is log, so a multiplicative offset is the constant-width one. ~3% is far below
        # any cost difference being claimed (the panels span two orders of magnitude), and
        # the error bars still start at the true value. Stated in the caption.
        pres = _samplers_present(sub)
        for i, samp in enumerate(pres):
            g = sub[sub["sampler"] == samp]
            if g.empty:
                continue
            fac = 1.03 ** (i - (len(pres) - 1) / 2.0)
            c = _hue(samp, colors)
            faded = samp in FADED
            for (fw, _p), gg in g.groupby(["framework", "platform"], dropna=False):
                st = _backend_style(fw, plat)
                st = {k: (c if v is None else v) for k, v in st.items()}
                lab = _series_label(samp, fw)
                x = gg["runtime_sec"].astype(float) / 1e3 * fac
                y = gg["abs_err"].astype(float)
                ax.plot(x, y, linestyle="none", color=c,
                        alpha=0.45 if faded else 0.9,
                        zorder=2 if faded else 3, label=lab, **st)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_title(f"{stat}, {plat.upper()}", fontsize=10)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    # NO "better" arrow. It was meant to give the reading direction of a log-log scatter
    # where both axes are costs to minimise, but placed at fixed axes-fraction coordinates
    # it landed on top of whichever run happened to occupy that corner -- in the FC/GPU
    # panel, the single smc_abc cell at G*=0.33. An annotation pointing at one data point
    # reads as a claim about that point, not about the axes. The caption states the
    # direction instead, where it cannot collide with anything.
    for ax in axes[len(combos):]:
        ax.set_visible(False)
    for ax in axes[:len(combos)]:
        ax.set_xlabel("wall-clock per cell (ks)")
        ax.set_ylabel(r"$|\Delta G|$")
    for ax in axes[:len(combos)]:
        ax.label_outer()
    # Say what the point and the bar ARE, in the figure, so it survives being read apart
    # from its caption.
    # Only the single-coupling copies are stamped, and only so the file identifies itself
    # in the dated results folder. The pooled version carries no on-figure note: "one
    # point per coupling" is caption material, and printed inside the axes it competed
    # with the data for the one empty corner of the panel.
    note = rf"$G^\star={_fmt(float(G), 2)}$" if G is not None else ""
    if note:
        axes[0].text(0.02, 0.98, note, transform=axes[0].transAxes, fontsize=8.5,
                     color="#555555", va="top", ha="left")
    _shared_legend(fig, axes[:len(combos)], extra=_backend_legend_handles(),
                   ncol=3 if G is not None else None,
                   fontsize=8 if G is not None else None)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out_png} ({len(d)} cells"
          + (f", G*={_fmt(float(G), 2)}" if G is not None else f", {n_G} G* pooled") + ")")


# az.summary rounds sd to 3 decimals, so a reported 0.000 means sd < 5e-4, not zero.
# Such cells cannot be drawn on a log axis and must not be silently dropped -- they are
# the most alarming cells in the study. They are floored here and drawn hollow, with the
# caption stating the convention.
_SD_FLOOR = 5e-4


def fig_calibration(df, out_png, G=None):
    r"""Posterior width against actual error: is the reported uncertainty honest?

    |dG| and sd sit in separate table columns, so "confidently wrong" -- small sd at
    large error -- is invisible unless the reader divides one by the other. Plotting
    them against each other with the sd=|dG| diagonal makes it positional: on or above
    the line the posterior covers its own error; far below it the sampler is certain and
    wrong, which is strictly worse than being uncertain and wrong.

    G=<val> restricts to one coupling for the supplement. The AXIS LIMITS are computed
    from the pooled data either way, so every per-coupling copy shares one frame with the
    main-text figure -- otherwise each page would silently rescale and a reader paging
    through them would compare positions that are not comparable.
    """
    d = _cells_by_config(df).dropna(subset=["abs_err", "G_sd"]).copy()
    d = d[d["abs_err"] > 0]
    if d.empty:
        print("(calibration figure skipped: no data)"); return
    d["sd_plot"] = d["G_sd"].clip(lower=_SD_FLOOR)
    d["floored"] = d["G_sd"] < _SD_FLOOR
    lo = min(d["abs_err"].min(), d["sd_plot"].min()) * 0.5
    hi = max(d["abs_err"].max(), d["sd_plot"].max()) * 2.0
    if G is not None:
        d = d[np.isclose(d["G_true"].astype(float), float(G))]
        if d.empty:
            print(f"(calibration figure skipped: no data at G*={G})"); return
    colors = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(SAMPLER_ORDER)}
    combos = [("FC", "gpu"), ("FCD", "gpu"), ("FC", "cpu"), ("FCD", "cpu")]
    combos = [c for c in combos
              if not d[(d["which_stat"] == c[0]) & (d["platform"] == c[1])].empty]
    # Compact canvas for the per-coupling copies -- see fig_accuracy_vs_cost.
    fig, axes = _facet_axes(2, 2, size=(4.1, 3.5) if G is not None else (5.2, 4.1))
    for ax, (stat, plat) in zip(axes, combos):
        sub = d[(d["which_stat"] == stat) & (d["platform"] == plat)]
        ax.plot([lo, hi], [lo, hi], "-", color="#999999", lw=1.2, zorder=1)
        for samp in _samplers_present(d):
            g = sub[sub["sampler"] == samp]
            if g.empty:
                continue
            _c = _hue(samp, colors)
            for (fw, plat), gsub in g.groupby(["framework", "platform"], dropna=False):
                st0 = _backend_style(fw, plat)
                for floored in (False, True):
                    gg = gsub[gsub["floored"] == floored]
                    if gg.empty:
                        continue
                    st = {k: (_c if v is None else v) for k, v in st0.items()}
                    if floored:      # sd below reporting precision: hollow + heavier ring
                        st["markerfacecolor"] = "none"
                        st["markeredgecolor"] = _c
                        st["markeredgewidth"] = 1.8
                    ax.plot(gg["abs_err"], gg["sd_plot"], linestyle="none", color=_c,
                            alpha=0.45 if samp in FADED else 0.9,
                            zorder=2 if samp in FADED else 3,
                            label=_series_label(samp, fw) if not floored else None, **st)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{stat}, {plat.upper()}", fontsize=10)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].annotate("posterior covers its error", xy=(0.05, 0.90),
                     xycoords="axes fraction", fontsize=8, color="#555555")
    axes[0].annotate("confidently wrong", xy=(0.42, 0.06),
                     xycoords="axes fraction", fontsize=8, color="#555555")
    for ax in axes[len(combos):]:
        ax.set_visible(False)
    for ax in axes[:len(combos)]:
        ax.set_xlabel(r"$|\Delta G|$"); ax.set_ylabel("posterior sd")
        ax.label_outer()
    # the hollow marker is a second encoding and needs its own key, not just a caption
    proxy = plt.Line2D([], [], marker="o", ms=8, linestyle="none", color="#777777",
                       markerfacecolor="none", markeredgecolor="#777777",
                       markeredgewidth=1.8)
    _shared_legend(fig, axes[:len(combos)],
                   extra=_backend_legend_handles()
                         + [(proxy, "sd below reporting precision")],
                   ncol=3 if G is not None else None,
                   fontsize=8 if G is not None else None)
    if G is not None:
        axes[0].text(0.02, 0.98, rf"$G^\star={_fmt(float(G), 2)}$",
                     transform=axes[0].transAxes, fontsize=8.5, color="#555555",
                     va="top", ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)
    n_f = int(d["floored"].sum())
    print(f"wrote {out_png} ({len(d)} cells, {n_f} with sd below {_SD_FLOOR} drawn hollow"
          + (f", G*={_fmt(float(G), 2)}" if G is not None else "") + ")")


def _sbc_files(results_dirs):
    out = []
    for d in results_dirs:
        out += glob.glob(os.path.join(d, "**", "sbc_ranks_*.npz"), recursive=True)
    return sorted(set(out))


def fig_sbc_recovery(results_dirs, out_png):
    r"""Per-replicate recovery over the SBC prior draws: $\hat G$ against $G^\star$.

    The rank ECDF (fig_sbc) compresses each replicate to a single number, which is what
    makes the calibration test sharp but also hides what the posteriors actually did. This
    shows the raw material: one point per replicate at its own prior-drawn ground truth,
    with the central 94% of the posterior as a vertical bar.

    It is a different picture from the benchmark recovery figure. There, four hand-picked
    $G^\star$ are each run once; here 100 ground truths are DRAWN FROM THE PRIOR, so the
    density of points along the axis is the prior itself and the coverage of the bars over
    the identity line is what the SBC ranks formalise.
    """
    files = _sbc_files(results_dirs)
    if not files:
        print("(SBC recovery figure skipped: no sbc_ranks_*.npz found)"); return
    colors = {s_: PALETTE[i % len(PALETTE)] for i, s_ in enumerate(SAMPLER_ORDER)}

    panels = []
    for f in files:
        base = os.path.basename(f)[len("sbc_ranks_"):-len(".npz")]
        samp = base.rsplit("_", 1)[0]; stat = base.rsplit("_", 1)[-1]
        d = os.path.dirname(f)
        rows = []
        for g in sorted(glob.glob(os.path.join(d, "draws_*.npz"))):
            if f"sampler_{samp}_" not in os.path.basename(g):
                continue
            if f"which_stat_{stat}_" not in os.path.basename(g):
                continue
            z = np.load(g)
            v = np.asarray(z["G"], float).ravel()
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            rows.append((float(z["G_true"]), float(v.mean()),
                         float(np.percentile(v, 3)), float(np.percentile(v, 97))))
        if rows:
            panels.append((samp, stat, np.array(rows)))
    if not panels:
        print("(SBC recovery figure skipped: no matching draws_*.npz)"); return

    # Wrap to two rows beyond two panels: four side by side squeezes each below the width
    # where the error bars are readable.
    ncol = 2 if len(panels) > 2 else len(panels)
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.6 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    allv = np.concatenate([p_[2][:, :1].ravel() for p_ in panels]
                          + [p_[2][:, 1:].ravel() for p_ in panels])
    lim = (0.0, float(np.nanpercentile(allv, 99.5)) * 1.05)
    for ax, (samp, stat, a) in zip(axes, panels):
        c = _hue(samp, colors)
        ax.plot(lim, lim, "-", color="#999999", lw=1.2, zorder=1)
        # central 94% of the posterior; drawn first and thin so the means stay readable
        ax.vlines(a[:, 0], a[:, 2], a[:, 3], color=c, lw=0.8, alpha=0.35, zorder=2)
        ax.plot(a[:, 0], a[:, 1], "o", ms=4.5, color=c, alpha=0.95,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3)
        cov = float(np.mean((a[:, 0] >= a[:, 2]) & (a[:, 0] <= a[:, 3])) * 100.0)
        ax.set_title(f"{samp} ({stat})", fontsize=10)
        ax.set_xlabel(r"prior-drawn truth $G^\star$")
        ax.set_xlim(*lim); ax.set_ylim(*lim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, ls=":", alpha=0.4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.annotate(f"{len(a)} replicates\n{cov:.0f}% of bars cover the truth",
                    xy=(0.04, 0.88), xycoords="axes fraction", fontsize=8.5,
                    color="#444444", va="top")
    axes[0].set_ylabel(r"posterior mean $\hat G$")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out_png} ({', '.join(f'{s_}:{len(a)}' for s_, _, a in panels)})")


def fig_sbc(results_dirs, out_png, out_tex=None, conf=0.95):
    r"""Simulation-based calibration: are the posterior ranks uniform?

    SBC draws G* from the prior, fits the posterior to data simulated at that G*, and
    records the rank of G* within the posterior draws. Under correct calibration those
    ranks are Uniform(0,1) -- the one check that tests the whole pipeline (model, sampler,
    tolerance) rather than any part of it.

    Plotted as the ECDF DIFFERENCE rather than a rank histogram: with 100 replicates a
    histogram's shape depends on an arbitrary bin count, while the ECDF uses every
    replicate and its deviation is directly comparable to a band. The band is the
    Kolmogorov-Smirnov critical value, so it is the same statistic as the KS test the
    driver already reports -- a curve leaving the band is exactly a rejection at that
    level, and the two cannot disagree.

    Shape carries meaning: a systematic tilt is bias, a U (ends high) is an overconfident
    posterior, an inverted U is an over-dispersed one.
    """
    files = _sbc_files(results_dirs)
    if not files:
        print("(SBC figure skipped: no sbc_ranks_*.npz found)"); return
    try:
        from scipy import stats as _st
    except Exception:
        _st = None

    colors = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(SAMPLER_ORDER)}
    # 1.358 is the 95% Kolmogorov quantile; 1.628 the 99%.
    kq = {0.95: 1.358, 0.99: 1.628}.get(conf, 1.358)

    # Colour is the SAMPLER, as everywhere else, so the feature needs its own channel or
    # smc_lik/FC and smc_lik/FCD render identically. Solid = FCD (the feature the paper
    # argues from), dashed = FC.
    _LS = {"FCD": "-", "FC": "--"}
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    rows, Ls = [], []
    for f in files:
        z = np.load(f)
        r = np.asarray(z["ranks"], float); n = np.asarray(z["sizes"], float)
        u = np.sort(r / np.maximum(n, 1.0))
        L = u.size; Ls.append(L)
        base = os.path.basename(f)[len("sbc_ranks_"):-len(".npz")]
        samp = base.rsplit("_", 1)[0]; stat = base.rsplit("_", 1)[-1]
        ecdf = np.arange(1, L + 1) / L
        ax.step(u, ecdf - u, where="post", lw=2, linestyle=_LS.get(stat, "-"),
                color=_hue(samp, colors), label=f"{samp} ({stat})", zorder=3)
        d = p_ = float("nan")
        if _st is not None:
            ks = _st.kstest(u, "uniform"); d, p_ = float(ks.statistic), float(ks.pvalue)
        rows.append({"sampler": samp, "which_stat": stat, "L": L, "ks_D": d, "ks_p": p_})

    L = max(Ls)
    band = kq / np.sqrt(L)
    ax.axhspan(-band, band, color="#999999", alpha=0.18, zorder=1,
               label=f"{int(conf*100)}% simultaneous band")
    ax.axhline(0.0, color="#999999", lw=1.0, zorder=2)
    ax.set_xlabel("normalised rank of $G^\\star$ in the posterior")
    ax.set_ylabel("ECDF $-$ uniform")
    ax.set_xlim(0, 1)
    ax.grid(True, ls=":", alpha=0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out_png} ({len(rows)} samplers, L={L})")

    if out_tex:
        lines = [
            r"\begin{table}[t]\centering",
            rf"\caption{{Simulation-based calibration. {L} replicates per sampler: a "
            rf"ground truth is drawn from the prior, data are simulated at it, and the "
            rf"rank of that truth among the posterior draws is recorded. Correct "
            rf"calibration makes the normalised ranks Uniform$(0,1)$; $D$ is the "
            rf"Kolmogorov--Smirnov distance from uniformity and $p$ its $p$-value, so "
            rf"large $p$ means no detectable miscalibration.}}",
            r"\label{tab:sbc}\small",
            r"\begin{tabular}{llrrr}",
            r"\toprule",
            r"Sampler & Feat. & replicates & KS $D$ & $p$ \\",
            r"\midrule",
        ]
        for r_ in rows:
            lines.append(f"{_tt(r_['sampler'])} & {r_['which_stat']} & {r_['L']} & "
                         f"{_fmt(r_['ks_D'], 3)} & {_fmt(r_['ks_p'], 3)} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        with open(out_tex, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"wrote {out_tex}")
    for r_ in rows:
        print(f"   [sbc] {r_['sampler']}/{r_['which_stat']}: L={r_['L']} "
              f"KS D={r_['ks_D']:.4f} p={r_['ks_p']:.4f}")


def fig_sbc_2d(results_dirs, out_png, out_tex=None, conf=0.95):
    r"""2-D SBC: rank ECDFs for G and for eta, side by side.

    Kept as two panels rather than one pooled statistic because that is where the result
    lives: the joint posterior is calibrated in G and not in eta, and pooling the ranks
    would average exactly that away.
    """
    files = [f for f in _sbc_files(results_dirs) if f.endswith("_2d.npz")]
    if not files:
        print("(2-D SBC figure skipped: no *_2d.npz found)"); return
    try:
        from scipy import stats as _st
    except Exception:
        _st = None
    colors = {s_: PALETTE[i % len(PALETTE)] for i, s_ in enumerate(SAMPLER_ORDER)}
    _LS = {"FCD": "-", "FC": "--"}
    kq = {0.95: 1.358, 0.99: 1.628}.get(conf, 1.358)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), sharey=True)
    rows, L = [], 0
    for ax, (key, rk, sz, ttl) in zip(axes, [
            ("G", "ranks", "sizes", r"$G$"),
            ("eta_mag", "eta_ranks", "eta_sizes", r"$|\eta|$")]):
        for f in sorted(files):
            z = np.load(f)
            if rk not in z:
                continue
            base = os.path.basename(f)[len("sbc_ranks_"):-len("_2d.npz")]
            samp, stat = base.rsplit("_", 1)
            u = np.sort(np.asarray(z[rk], float)
                        / np.maximum(np.asarray(z[sz], float), 1.0))
            n = u.size; L = max(L, n)
            ax.step(u, np.arange(1, n + 1) / n - u, where="post", lw=2,
                    linestyle=_LS.get(stat, "-"), color=_hue(samp, colors),
                    label=f"{samp} ({stat})", zorder=3)
            if _st is not None:
                ks = _st.kstest(u, "uniform")
                rows.append({"param": key, "sampler": samp, "which_stat": stat,
                             "L": n, "ks_D": float(ks.statistic),
                             "ks_p": float(ks.pvalue)})
        ax.set_title(ttl, fontsize=11)
        ax.set_xlabel("normalised rank of the truth")
        ax.set_xlim(0, 1); ax.grid(True, ls=":", alpha=0.4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    band = kq / np.sqrt(max(L, 1))
    for ax in axes:
        ax.axhspan(-band, band, color="#999999", alpha=0.18, zorder=1)
        ax.axhline(0.0, color="#999999", lw=1.0, zorder=2)
    axes[0].set_ylabel("ECDF $-$ uniform")
    axes[1].legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out_png} ({len(rows)} series, L={L}, band=+/-{band:.3f})")

    if out_tex and rows:
        lines = [r"\begin{table}[t]\centering",
                 rf"\caption{{Two-dimensional simulation-based calibration. $L={L}$ "
                 rf"replicates per sampler and feature, with BOTH $G^\star$ and "
                 rf"$|\eta^\star|$ drawn from the priors the inference fits. Ranks are "
                 rf"recorded per parameter: the joint posterior is calibrated in $G$ "
                 rf"throughout, while $|\eta|$ is rejected in three of the four "
                 rf"combinations.}}",
                 r"\label{tab:sbc2d}\small", r"\begin{tabular}{llrrr}", r"\toprule",
                 r"Parameter & Sampler (feature) & $L$ & KS $D$ & $p$ \\", r"\midrule"]
        for key, ttl in (("G", r"$G$"), ("eta_mag", r"$|\eta|$")):
            for r_ in [x for x in rows if x["param"] == key]:
                flag = "" if r_["ks_p"] > 0.05 else r"$^{\dagger}$"
                lines.append(f"{ttl} & {_tt(r_['sampler'])} ({r_['which_stat']}){flag} & "
                             f"{r_['L']} & {_fmt(r_['ks_D'],3)} & {_fmt(r_['ks_p'],3)} \\\\")
        lines += [r"\bottomrule", r"\\[2pt]",
                  r"\multicolumn{5}{l}{\footnotesize $^{\dagger}$rejected at the 5\% "
                  r"level. Eight tests at $\alpha=0.05$ would give $\approx0.4$ false "
                  r"rejections by chance; all three fall on the same parameter.}\\",
                  r"\end{tabular}", r"\end{table}"]
        with open(out_tex, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"wrote {out_tex}")
    for r_ in rows:
        print(f"   [sbc2d] {r_['param']:8s} {r_['sampler']}/{r_['which_stat']}: "
              f"D={r_['ks_D']:.4f} p={r_['ks_p']:.4f}")


def fig_joint_posterior(results_dirs, out_png, n_show=3):
    r"""Joint $(G, |\eta|)$ posteriors for a few 2-D replicates.

    The marginal rank tests say WHETHER each coordinate is calibrated; this says why. If
    the two parameters trade off, the joint posterior is a ridge and the eta marginal is
    a projection of it -- which is the mechanism a reader will ask about the moment they
    see eta rejected and G not.
    """
    files = []
    for d in results_dirs:
        files += glob.glob(os.path.join(d, "**", "draws_*_eta*.npz"), recursive=True)
    files = sorted(set(files))
    if not files:
        print("(joint posterior figure skipped: no 2-D draws found)"); return
    picked = []
    for f in files:
        if "sampler_smc_lik_" not in os.path.basename(f):
            continue
        if "which_stat_FCD_" not in os.path.basename(f):
            continue
        z = np.load(f)
        if "eta_mag" not in z:
            continue
        picked.append((float(z["G_true"]), f))
    if not picked:
        print("(joint posterior figure skipped: no smc_lik/FCD 2-D draws)"); return
    picked.sort()
    idx = np.linspace(0, len(picked) - 1, min(n_show, len(picked))).astype(int)
    sel = [picked[i] for i in idx]

    fig, axes = plt.subplots(1, len(sel), figsize=(4.6 * len(sel), 4.4))
    axes = np.atleast_1d(axes)
    for ax, (gt, f) in zip(axes, sel):
        z = np.load(f)
        G = np.asarray(z["G"], float).ravel()
        E = np.asarray(z["eta_mag"], float).ravel()
        m = np.isfinite(G) & np.isfinite(E)
        G, E = G[m], E[m]
        et = float(np.abs(np.asarray(z["theta_true"], float).ravel()[1]))
        r = float(np.corrcoef(G, E)[0, 1]) if G.size > 2 else float("nan")
        ax.plot(G, E, ".", ms=2.0, alpha=0.25, color=PALETTE[1], zorder=2)
        ax.axvline(gt, color="#555555", lw=1.1, ls="--", zorder=3)
        ax.axhline(et, color="#555555", lw=1.1, ls="--", zorder=3)
        ax.plot([gt], [et], "*", ms=13, color="#111111", zorder=4)
        ax.set_title(rf"$G^\star={gt:.3f}$, $|\eta^\star|={et:.2f}$" "\n"
                     rf"posterior corr $={r:+.2f}$", fontsize=9.5)
        ax.set_xlabel(r"$G$")
        ax.grid(True, ls=":", alpha=0.4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel(r"$|\eta|$")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out_png} ({len(sel)} replicates)")


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
    # Same subset and same per-cell pick as the tables. Both are mandatory, and for the
    # same reasons the tables need them:
    #   bench_subset  -- without it the 88-node and long-t_end runs, which share G*=0.2
    #                    with the benchmark, join these lines while being excluded from
    #                    the tables, so the two disagree the moment such a run lands.
    #   _latest_per_cell -- sorting on `batch` alone leaves same-batch re-runs to glob
    #                    order, which is how the figure came to plot the SUPERSEDED
    #                    50/100 slice cells while the table showed the budget-matched
    #                    1000/1000 re-run the text argues from (4 of 90 cells disagreed).
    # Route through the shared selector rather than re-picking here -- that duplication is
    # exactly how this figure drifted from the tables before. `framework` is part of that
    # key: without it JAX-on-CPU and PyMC-on-CPU are the same cell and one silently evicts
    # the other, which made smc_abc/FC/cpu/G*=0.2 resolve to the PyMC run (0.169) while the
    # other three G came from JAX, all drawn with PyMC's marker.
    d = _cells_by_config(df).dropna(subset=["G_true", ycol])
    if d.empty:
        print("(recovery figure skipped: no data)"); return

    colors = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(SAMPLER_ORDER)}
    # NO (feature, backend) dash encoding. It was redundant -- every panel is titled with
    # its own feature and backend -- and worse, it collided with Fig. 4, where a dashed,
    # faded series means "a configuration shown for contrast that is NOT the one the paper
    # argues from". Dashing is now reserved for exactly that meaning in both figures, and
    # here it marks the uncalibrated epsilon=10 ABC (module-level FADED).

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
        # identity: perfect recovery. Distance from it IS the error, to scale.
        ax.plot(lim, lim, "-", color="#999999", lw=1.2, zorder=1)
        # One line per (algorithm, framework): on CPU the same algorithm can appear under
        # both JAX and PyMC, and those are different implementations, not two points on
        # one curve. Keying the series on the sampler alone joined them.
        series = {}
        for samp in _samplers_present(d):
            g0 = d[(d["sampler"] == samp) & (d["which_stat"] == stat)
                   & (d["platform"] == plat)]
            if g0.empty:
                continue
            for fw, g in g0.groupby(g0["framework"].astype(str)):
                # `d` is already one row per cell (see _latest_per_cell above), so this
                # only orders the line; re-picking here is what previously diverged from
                # the table.
                series[(samp, fw)] = g.sort_values("G_true")

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
        for i, (samp, fw) in enumerate(present):
            g = series[(samp, fw)]
            dx = (i - (len(present) - 1) / 2.0) * span
            _c = _hue(samp, colors)
            st = _backend_style(fw, plat)
            st = {k: (_c if v is None else v) for k, v in st.items()}
            st["ms"] = max(5.5, st.get("ms", 6) - 2)
            faded = samp in FADED
            ax.plot(g["G_true"].astype(float) + dx, g[ycol].astype(float),
                    "--" if faded else "-",
                    color=_c, lw=1.6 if faded else 1.8,
                    alpha=0.45 if faded else 0.95,
                    zorder=2 if faded else 3,
                    label=_series_label(samp, fw), **st)
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

    # Two keys, as in Fig. 4: hue = algorithm, marker = backend. The marker key is not
    # optional here even though the panels are already split by platform, because a CPU
    # panel carries BOTH the same-code JAX run and the PyMC one and they mean different
    # things. The faded dashed entry is listed with the algorithms, since it is a variant
    # of one, not a backend.
    from matplotlib.lines import Line2D
    alg = [Line2D([], [], color=colors[s], lw=2, marker="o", ms=5,
                  markeredgecolor="white", label=_series_label(s))
           for s in _samplers_present(d) if s not in FADED]
    alg += [Line2D([], [], color=_hue(s, colors), lw=1.6, ls="--", alpha=0.45,
                   marker="o", ms=5, label=_series_label(s))
            for s in _samplers_present(d) if s in FADED]
    # tight_layout first, then place both legends against the finished geometry: called
    # the other way round the algorithm key lands on top of the "true G*" x-label.
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    leg = fig.legend(handles=alg, frameon=False, fontsize=9,
                     ncol=min(4, len(alg)), loc="lower center",
                     bbox_to_anchor=(0.5, -0.055), title="algorithm",
                     alignment="left")
    leg.get_title().set_fontsize(9)
    bh = _backend_legend_handles()
    # Upper left of the last panel: the recovery curves all run bottom-left to top-right,
    # so that corner is the one region guaranteed to be empty in every panel.
    axes[len(combos) - 1].legend(handles=[h for h, _ in bh],
                                 labels=[l for _, l in bh],
                                 frameon=False, fontsize=8, loc="upper left",
                                 title="backend", alignment="left",
                                 handletextpad=0.6, borderpad=0.2)
    # No second tight_layout: it would undo the rect reserved for the algorithm key.
    fig.savefig(out_png, dpi=300, bbox_inches="tight",
                bbox_extra_artists=[leg]); plt.close(fig)
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
    ap.add_argument("--table_G", type=float, default=0.2,
                    help="coupling whose runtime/ESS-per-second fill the cost columns "
                         "of the master table; accuracy is shown at every G* regardless")
    ap.add_argument("--table_label", default="tab:accuracy",
                    help="LaTeX label for the benchmark table; set to match the "
                         "\\ref already used in main.tex")
    ap.add_argument("--legacy_tables", action="store_true",
                    help="also emit the superseded benchmark_table.tex/full_grid_table.tex "
                         "pair; main.tex inputs neither since they merged into the master "
                         "table, so this exists only to reproduce an older draft")
    args = ap.parse_args()

    tables = os.path.join(args.out, "tables"); os.makedirs(tables, exist_ok=True)
    figs = os.path.join(args.out, "figures"); os.makedirs(figs, exist_ok=True)

    df = load_master(args.results)
    master = os.path.join(args.out, "master_results.csv")
    df.to_csv(master, index=False)
    print(f"wrote {master} ({len(df)} cells)")

    # master_table (one wide table, all G* + cost) is SUPERSEDED by the per-configuration
    # subtables below and is no longer input by main.tex; kept behind --legacy_tables.
    if args.legacy_tables:
        latex_master_table(df, os.path.join(tables, "master_table.tex"),
                           cost_G=args.table_G, label=args.table_label)
    # Per-config metric tables. The BODY carries one coupling (--table_G); the supplement
    # repeats the identical four tables for every other coupling, so the main text stays
    # readable while the full record is still on the page somewhere.
    all_Gs = sorted(bench_subset(df)["G_true"].dropna().astype(float).unique())
    supp_Gs = [g for g in all_Gs if not np.isclose(g, args.table_G)]
    latex_config_subtables(df, os.path.join(tables, "config_subtables_main.tex"),
                           Gs=[args.table_G])
    # ONE FILE PER COUPLING. The supplement interleaves table and figures per G* -- table
    # page, then a page carrying that coupling's Figs. 4, 6 and 7 -- so main.tex has to be
    # able to place them individually. A single file holding all three couplings' tables
    # could only be dropped in as one block.
    for _g in supp_Gs:
        latex_config_subtables(
            df, os.path.join(tables, f"config_subtables_G{_g_slug(_g)}.tex"), Gs=[_g])
    # Kept for backward compatibility: main.tex no longer inputs this.
    latex_config_subtables(
        df, os.path.join(tables, "config_subtables_supp.tex"), Gs=supp_Gs)
    if args.legacy_tables:
        latex_benchmark_table(df, os.path.join(tables, "benchmark_table.tex"),
                              G=args.table_G, label=args.table_label)
        latex_full_grid_table(df, os.path.join(tables, "full_grid_table.tex"))
    latex_settings_table(args.config, os.path.join(tables, "settings_table.tex"))
    fig_recovery_vs_G(df, os.path.join(figs, "recovery_vs_G.png"))
    fig_accuracy_vs_cost(df, os.path.join(figs, "accuracy_vs_cost.png"))
    # One per coupling for the supplement, so every supplementary table has a figure that
    # shows the same cells. The main-text figure pools them with range bars.
    fig_calibration(df, os.path.join(figs, "calibration_sd_vs_err.png"))
    # One copy of each per coupling: the supplement gives every G* a page carrying its own
    # cost/accuracy and calibration views next to its table.
    for _g in all_Gs:
        fig_accuracy_vs_cost(df, os.path.join(
            figs, f"accuracy_vs_cost_G{_g_slug(_g)}.png"), G=_g)
        fig_calibration(df, os.path.join(
            figs, f"calibration_sd_vs_err_G{_g_slug(_g)}.png"), G=_g)
    fig_sbc_recovery(args.results, os.path.join(figs, "sbc_recovery.png"))
    fig_sbc(args.results, os.path.join(figs, "sbc_ranks.png"),
            out_tex=os.path.join(tables, "sbc_table.tex"))
    fig_sbc_2d(args.results, os.path.join(figs, "sbc_ranks_2d.png"),
               out_tex=os.path.join(tables, "sbc_2d_table.tex"))
    fig_joint_posterior(args.results, os.path.join(figs, "joint_posterior.png"))
    fig_throughput(df, os.path.join(figs, "throughput_ess.png"))
    print("done.")


if __name__ == "__main__":
    main()
