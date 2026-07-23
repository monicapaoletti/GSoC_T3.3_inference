"""Shared sampler-benchmark metrics, used by both the PyMC (mpr_jax_pymc.py) and
the JAX/NumPyro+BlackJAX (mpr_jax_numpyro.py) inference scripts so every sampler
is scored the same way. Metrics follow Baldy et al. 'DCM in Probabilistic
Programming Languages': convergence (split-R̂, ESS bulk/tail, relative ESS),
efficiency (wall-time, ESS/sec), accuracy (RMSE, posterior z-score, shrinkage),
identifiability (parameter correlation) and sampling health (divergences, etc.).
"""
import numpy as np
import arviz as az


def benchmark_metrics(trace, theta_true, my_var_names, prior_sd, runtime, rmse, meta):
    """Comprehensive sampler-benchmark metrics. Parameter-agnostic: loops over
    my_var_names, so it already supports (G, eta, ...) once more params are added.
    `trace` is any arviz InferenceData (from PyMC, NumPyro, or hand-built from
    BlackJAX). Returns (run_row: dict for a one-line summary, param_rows: list[dict])."""
    post = trace.posterior
    n_chains = int(post.sizes["chain"])
    n_draws = int(post.sizes["draw"])
    total_draws = n_chains * n_draws

    try:
        summ = az.summary(trace, var_names=list(my_var_names), hdi_prob=0.94)
    except Exception:
        summ = az.summary(trace, hdi_prob=0.94)

    def _get(v, col):
        try:
            return float(summ.loc[v, col])
        except Exception:
            return float("nan")

    # --- sampling health (present for NUTS/HMC; mostly absent for the others) ---
    n_div = accept = tree_depth = mean_lp = float("nan")
    if "sample_stats" in trace.groups():
        ss = trace.sample_stats
        if "diverging" in ss:
            n_div = float(np.asarray(ss["diverging"]).sum())
        for a in ("acceptance_rate", "mean_tree_accept", "accept_prob"):
            if a in ss:
                accept = float(np.asarray(ss[a]).mean()); break
        for t in ("tree_depth", "num_steps", "n_steps"):
            if t in ss:
                tree_depth = float(np.asarray(ss[t]).mean()); break
        if "lp" in ss:
            mean_lp = float(np.asarray(ss["lp"]).mean())

    # --- per-parameter metrics ---
    param_rows = []
    for i, v in enumerate(my_var_names):
        draws = np.asarray(post[v]).ravel()
        mean = float(np.mean(draws)); sd = float(np.std(draws, ddof=1))
        true = float(np.ravel(theta_true)[i])
        hist, edges = np.histogram(draws, bins=50)
        mapv = float((edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2.0)
        ess_bulk = _get(v, "ess_bulk"); ess_tail = _get(v, "ess_tail")
        hdi_lo = _get(v, "hdi_3%"); hdi_hi = _get(v, "hdi_97%")
        psd = prior_sd.get(v, float("nan")) if isinstance(prior_sd, dict) else float(prior_sd)
        param_rows.append({
            **meta,
            "param": v, "true": true,
            "post_mean": mean, "post_sd": sd, "post_map": mapv,
            "abs_err_mean": abs(mean - true), "abs_err_map": abs(mapv - true),
            "hdi_3%": hdi_lo, "hdi_97%": hdi_hi, "hdi_width": hdi_hi - hdi_lo,
            "in_94hdi": bool(hdi_lo <= true <= hdi_hi),
            "z_score": (mean - true) / sd if sd > 0 else float("nan"),
            "shrinkage": 1.0 - (sd ** 2) / (psd ** 2) if psd and psd > 0 else float("nan"),
            "r_hat": _get(v, "r_hat"), "ess_bulk": ess_bulk, "ess_tail": ess_tail,
            "rel_ess_bulk": ess_bulk / total_draws if total_draws else float("nan"),
            "ess_bulk_per_sec": ess_bulk / runtime if runtime and runtime > 0 else float("nan"),
            "mcse_mean": _get(v, "mcse_mean"),
        })

    # --- parameter correlation (identifiability); trivial while only G ---
    if len(my_var_names) > 1:
        M = np.column_stack([np.asarray(post[v]).ravel() for v in my_var_names])
        od = np.corrcoef(M, rowvar=False)[np.triu_indices(len(my_var_names), 1)]
        corr_min, corr_max = float(np.min(od)), float(np.max(od))
    else:
        corr_min = corr_max = float("nan")

    pr = param_rows
    _nanmax = lambda k: float(np.nanmax([r[k] for r in pr]))
    _nanmin = lambda k: float(np.nanmin([r[k] for r in pr]))
    run_row = {
        **meta,
        "runtime_sec": runtime,
        "n_chains": n_chains, "n_draws": n_draws, "total_draws": total_draws,
        "n_params": len(my_var_names),
        # convergence / efficiency (worst-case across params)
        "max_r_hat": _nanmax("r_hat"),
        "min_ess_bulk": _nanmin("ess_bulk"), "min_ess_tail": _nanmin("ess_tail"),
        "min_rel_ess_bulk": _nanmin("rel_ess_bulk"),
        "min_ess_bulk_per_sec": _nanmin("ess_bulk_per_sec"),
        # accuracy / calibration
        "rmse_param_mean": rmse["param_mean"], "rmse_param_map": rmse["param_map"],
        "rmse_fit_mean": rmse["fit_mean"], "rmse_fit_map": rmse["fit_map"],
        "max_abs_z": float(np.nanmax([abs(r["z_score"]) for r in pr])) if pr else float("nan"),
        "mean_shrinkage": float(np.nanmean([r["shrinkage"] for r in pr])),
        "coverage_94hdi": float(np.mean([r["in_94hdi"] for r in pr])),
        # identifiability + health
        "corr_min": corr_min, "corr_max": corr_max,
        "n_divergences": n_div, "mean_accept": accept,
        "mean_tree_depth": tree_depth, "mean_lp": mean_lp,
    }
    return run_row, param_rows
