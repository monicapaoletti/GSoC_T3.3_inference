#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runnable inference script for your FC model using Numpyro + JAX.
Now automatically appends parameter tags to all saved outputs.
"""

import sys
import os
import time
import warnings
from copy import deepcopy
import argparse

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") 
import networkx as nx
import seaborn as sns
import pandas as pd
import time

import arviz as az
import numpyro as npr
from numpyro import plate
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_value#, Pathfinder
import blackjax
from numpyro.infer.util import initialize_model


# local modules
import utils
from FCD_jax import *
import mpr_jax
mpr_jax = __import__("mpr_jax")
import json
from benchmark_utils import benchmark_metrics
from smc_jax import run_tempered_smc, run_abc_smc
from mcmc_jax import run_parallel_rwmh, run_demc, run_slice

# ------------------ Argument parser ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run MPR JAX NumPyro inference")

    # Model/simulation parameters
    parser.add_argument("--G", type=float, default=0.33)

    # Statistics
    parser.add_argument("--which_stat", type=str, default="FCD")

    # scale the next three parameters together 
    parser.add_argument("--t_end", type=int, default=3000)
    parser.add_argument("--tr", type=int, default=5)
    parser.add_argument("--cut", type=int, default=10)

    parser.add_argument("--wwidth", type=int, default=30)
    parser.add_argument("--maxWindows", type=int, default=200)
    parser.add_argument("--olap", type=float, default=0.5)

    # Inference settings
    parser.add_argument("--obs_err", type=float, default=1.0,
                        help="Gaussian likelihood sigma on the (FC/FCD) features. Default 1.0 "
                             "matches vbjax's dist.Normal(mu, 1); kept identical across scripts.")
    parser.add_argument("--n_prior", type=int, default=100)
    parser.add_argument("--n_warmup", type=int, default=20)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--n_chains", type=int, default=1)
    parser.add_argument("--chain_method", type=str, default="vectorized",
                    choices=["parallel", "vectorized", "sequential"],
                    help="NumPyro NUTS chain execution: 'vectorized' vmaps chains on-device "
                         "(fills the GPU -> more chains ~free); 'parallel' spawns host "
                         "processes (CPU-style); 'sequential' one after another.")
    parser.add_argument("--scale", type=int, default=1, help="prior scale")
    parser.add_argument("--sampler", type=str, default="pathfinder",
                    choices=["nuts", "pathfinder", "blackjax", "smc_lik", "smc_abc",
                             "rwmh", "demc", "slice"],
                    help="nuts=NumPyro NUTS, blackjax=BlackJAX NUTS, pathfinder=BlackJAX "
                         "Pathfinder VI, smc_lik=JAX likelihood-tempered SMC, smc_abc=JAX "
                         "ABC SMC, rwmh=JAX parallel-chain RW-Metropolis, demc=JAX "
                         "DE-Metropolis, slice=JAX slice sampling (all GPU chain-vmapped).")
    # --- batched-SMC controls (smc_lik / smc_abc); n_particles is the batched axis ---
    parser.add_argument("--n_particles", type=int, default=1000,
                    help="SMC particle-cloud size (the on-device vmap batch).")
    parser.add_argument("--n_stages", type=int, default=50,
                    help="SMC tempering stages (lambda 0->1, or eps schedule length).")
    parser.add_argument("--n_mcmc", type=int, default=5,
                    help="random-walk MH moves per SMC stage.")
    parser.add_argument("--rw_step", type=float, default=0.1,
                    help="log-space random-walk proposal scale for the SMC MH move.")
    parser.add_argument("--abc_eps0", type=float, default=None,
                    help="smc_abc initial (large) epsilon; default = median prior distance.")
    parser.add_argument("--abc_eps_target", type=float, default=None,
                    help="smc_abc final epsilon; default = obs_err * sqrt(n_features).")

    # Misc
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=int(time.time()))
    # SC matrix type
    parser.add_argument("--SC_type", type=str, default="data",
                    help="Type of structural connectivity: 'sim' for simulated, 'data' for real data")
    parser.add_argument("--SC_size", type=int, default=88,
                        help="Number of nodes if using simulated SC") #change to 6 or 10 for small networks

    # --- gradient-based knobs (see mpr_jax.py) ---
    parser.add_argument("--grad_horizon", type=int, default=0,
                        help="Truncated-BPTT horizon in neural steps (0=full BPTT). ~200 tames chaotic gradient explosion.")
    parser.add_argument("--clip_mode", type=str, default="hard", choices=["hard", "soft", "none"],
                        help="Derivative bounding in f_mpr: hard=jnp.clip, soft=tanh (smooth, gradient-friendly), none=unbounded.")
    parser.add_argument("--fast_bold", action="store_true",
                        help="Integrate BOLD once per rv_decimate block (approx, ~1.4x faster).")
    parser.add_argument("--fcd_stride1", action=argparse.BooleanOptionalAction, default=True,
                        help="Use vbi-style FCD windowing (shift=1 sample) instead of the "
                             "overlap-fraction shift; pairs with the hardcoded triu offset k=30. "
                             "Pass --no-fcd_stride1 to revert to the prior overlap windowing.")
    parser.add_argument("--fc_eps", type=float, default=0.0,
                        help="Stabilise FC/FCD correlation denominator (e.g. 1e-6) to keep gradients finite.")
    # --- likelihood feature scaling (mirror of mpr_jax_pymc.py) ---
    parser.add_argument("--fisher_z", action=argparse.BooleanOptionalAction, default=True,
                        help="arctanh (Fisher z) on correlation features: variance-stabilizes and "
                             "unbounds them so a fixed-sigma Gaussian is well-specified (no extra sims).")
    parser.add_argument("--standardize_features", action=argparse.BooleanOptionalAction, default=True,
                        help="Z-score each feature by its scatter across noise seeds so obs_err is a "
                             "unit-scale sigma and FC/FCD are comparable (applied to model + observed).")
    parser.add_argument("--n_ref_seeds", type=int, default=8,
                        help="Noise seeds for the per-feature scatter estimate (--standardize_features).")
    parser.add_argument("--noisy_obs", action=argparse.BooleanOptionalAction, default=True,
                        help="Add N(0, obs_err) to the synthetic observed data so it is a genuine "
                             "draw from the model (needed for honest calibration).")
    parser.add_argument("--keep_negative_fc", action=argparse.BooleanOptionalAction, default=True,
                        help="Keep negative (anti-)correlations instead of the default ReLU.")
    parser.add_argument("--grad_method", type=str, default="fd", choices=["fd", "autodiff"],
                        help="Gradient for NUTS/BlackJAX: fd=central finite differences via custom_vjp "
                             "(correct across the bifurcation), autodiff (exact but unreliable in chaos).")
    parser.add_argument("--fd_h", type=float, default=1e-2, help="Finite-difference step for grad_method=fd.")

    return parser.parse_args()


# ------------------ Timer decorator ------------------
def timer(func):
    def wrapper_timer(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.2f} seconds")
        return result
    return wrapper_timer


# ------------------ Wrapper ------------------
def _fisher_z(v, eps=1e-4):
    """Fisher z-transform arctanh(r): variance-stabilizes correlation features and
    maps [-1,1]->R so a fixed-sigma Gaussian likelihood is well-specified. Mirrors
    mpr_jax_pymc._fisher_z. Applied inside the forward so the FD gradient sees it."""
    return jnp.arctanh(jnp.clip(v, -1.0 + eps, 1.0 - eps))

@timer
def wrapper_fc(G, par, cut, tr, starts, nn, grad_horizon=0, fast_bold=False, eps=0.0,
               fisher_z=False, keep_negative=False):
    par = deepcopy(par)
    par["G"] = G
    sde = mpr_jax.MPR_sde.create(par)
    data = sde.run({}, record_rv=False, fast_bold=fast_bold, grad_horizon=grad_horizon)
    bold_d = data["bold_d"]

    FC_full = get_fc(bold_d[int(cut):].T, eps=eps, keep_negative=keep_negative)   # <<< ensure int(cut)
    #print(FC_full)
    tri_idx = jnp.triu_indices(FC_full.shape[0], k=1)
    v = FC_full[tri_idx]
    return _fisher_z(v) if fisher_z else v

@timer
def wrapper_fcd(G, par, cut, tr, starts, nn, grad_horizon=0, fast_bold=False, eps=0.0,
                fisher_z=False, keep_negative=False):
    par = deepcopy(par)
    par["G"] = G
    sde = mpr_jax.MPR_sde.create(par)
    data = sde.run({}, record_rv=False, fast_bold=fast_bold, grad_horizon=grad_horizon)
    bold_d = data["bold_d"]

    bold_d_sub = bold_d[cut::tr].T

    # non-jitted extract (we're already inside numpyro's traced/jitted model, and
    # this lets eps be a concrete float for the stabilised correlation)
    FCD_full = extract_FCD_jax(bold_d_sub, starts, nn, wwidth=30, olap=0.94, eps=eps, keep_negative=keep_negative)
    #print(FCD_full)
    tri_idx = jnp.triu_indices(FCD_full.shape[0], k=30)
    v = FCD_full[tri_idx]
    return _fisher_z(v) if fisher_z else v


# ------------------ per-feature scatter (for --standardize_features) ------------------
def feature_reference_scatter(par, which_stat, cut, tr, starts, nn,
                              fast_bold=False, eps=0.0, n_ref=8, base_seed=10_000,
                              fisher_z=False, keep_negative=False):
    """Per-feature mean/std of the FC/FCD statistic over n_ref noise seeds. Mirrors
    mpr_jax_pymc.feature_reference_scatter; extraction matches wrapper_fc/fcd so the
    standardized features stay congruent with model + observed."""
    wrap = wrapper_fc if which_stat == "FC" else wrapper_fcd
    reps = []
    for i in range(n_ref):
        p = dict(par); p["seed"] = base_seed + i
        v = wrap(G=p["G"], par=p, cut=cut, tr=tr, starts=starts, nn=nn,
                 grad_horizon=0, fast_bold=fast_bold, eps=eps, fisher_z=fisher_z,
                 keep_negative=keep_negative)
        reps.append(np.asarray(v, dtype=np.float32))
    reps = np.stack(reps, 0)
    mu = reps.mean(0).astype(np.float32)
    sd = np.maximum(reps.std(0, ddof=1), 1e-6).astype(np.float32)
    return mu, sd


# ------------------ Forward model (shared by NumPyro NUTS + BlackJAX) ------------------
def make_forward_fn(par, which_stat, cut, tr, starts, nn, grad_horizon=0,
                    fast_bold=False, eps=0.0, grad_method="fd", fd_h=1e-2,
                    fisher_z=False, keep_negative=False):
    """Return forward_fn(G) -> flat FC/FCD statistic, matching wrapper_fc/fcd.

    grad_method="autodiff": plain JAX forward (autodiff gradient — unreliable in
    the chaotic regime). grad_method="fd": wraps the forward in jax.custom_vjp so
    that jax.grad returns the CENTRAL FINITE-DIFFERENCE gradient (valid because the
    noise is frozen). This lets NumPyro/BlackJAX NUTS run on-device with the
    reliable FD gradient instead of the exploding autodiff one."""
    base = dict(par)

    def raw(G):
        p = dict(base); p["G"] = G
        sde = mpr_jax.MPR_sde.create(p)
        bold = sde.run({}, record_rv=False, fast_bold=fast_bold, grad_horizon=grad_horizon)["bold_d"]
        if which_stat == "FC":
            F = get_fc(bold[int(cut):].T, eps=eps, keep_negative=keep_negative)
            v = F[jnp.triu_indices(F.shape[0], k=1)]
        else:
            sub = bold[cut::tr].T
            F = extract_FCD_jax(sub, starts, nn, wwidth=30, olap=0.94, eps=eps, keep_negative=keep_negative)
            v = F[jnp.triu_indices(F.shape[0], k=30)]
        return _fisher_z(v) if fisher_z else v

    if grad_method == "autodiff":
        return raw

    @jax.custom_vjp
    def fwd(G):
        return raw(G)

    def fwd_fwd(G):
        return raw(G), G                       # save G for the backward pass

    def fwd_bwd(G, ct):
        # Central FD (frozen noise -> valid). The two +/-h simulations are run in
        # ONE vmapped call, so on GPU they execute in parallel (and XLA fuses them
        # on CPU) instead of two sequential sims per leapfrog step.
        both = jax.vmap(raw)(jnp.stack([G + fd_h, G - fd_h]))  # (2, m)
        jacv = (both[0] - both[1]) / (2.0 * fd_h)              # d(stat)/dG, (m,)
        return (jnp.vdot(ct, jacv),)                           # scalar cotangent wrt G

    fwd.defvjp(fwd_fwd, fwd_bwd)
    return fwd


# ------------------ Helpers ------------------
def calcula_map(chains_):
    params_map = []
    for i in range(int(chains_.shape[0])):
        y = np.asarray(chains_[i])
        y = y[np.isfinite(y)]                       # a diverged sampler (e.g. pathfinder)
        if y.size == 0:                             # can produce all-NaN chains; record
            params_map.append(float("nan"))         # NaN instead of crashing np.histogram
            continue
        hist, bin_edges = np.histogram(y, bins=50)
        x_value_at_peak = (bin_edges[np.argmax(hist)] + bin_edges[np.argmax(hist) + 1]) / 2
        params_map.append(x_value_at_peak)
    return params_map


def plot_trace_chains(mcmc, theta_true, var_names, savepath=None):
    #az_obj = az.from_numpyro(mcmc)
    out = az.plot_trace(
        #az_obj,
        mcmc,
        var_names=var_names,
        compact=True,
        kind="trace",
        backend_kwargs={"layout": "constrained", "figsize": (6, 3 * len(var_names))},
    )
    if isinstance(out, tuple):
        fig, axes = out
    else:
        axes = np.atleast_2d(out)
        fig = axes[0, 0].figure

    axes = np.atleast_2d(axes)
    for i, true_val in enumerate(theta_true):
        for ax in [axes[i, 0]]:
            ax.axvline(x=true_val, color="red", linestyle="--")
        for ax in [axes[i, 1]]:
            ax.axhline(y=true_val, color="red", linestyle="--")

    fig.suptitle("Posterior samples", fontsize=16)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        print(f"Trace plot saved to: {savepath}")
    plt.close(fig)


def plot_posterior_pooled(var_names, theta_true, prior_predictions, chains_pooled, title, savepath=None):
    params_map_pooled = calcula_map(chains_pooled)
    n_params = len(var_names)
    fig, ax = plt.subplots(nrows=n_params, ncols=1, figsize=(6, 4 * n_params))
    if n_params == 1:
        ax = [ax]
    for i, prm in enumerate(var_names):
        a = ax[i]
        a.set_xlabel(prm)
        a.axvline(theta_true[i], color='r', linestyle='--', label='true')
        a.axvline(params_map_pooled[i], color='lightblue', linestyle='--', lw=2, label='MAP')
        if prior_predictions is not None and prm in prior_predictions:
            sns.kdeplot(prior_predictions[prm], ax=a, color='lime', alpha=0.5, lw=2, label='prior', fill=True)
        sns.kdeplot(chains_pooled[i, :], ax=a, color='darkblue', alpha=0.2, lw=2, label='posterior', fill=True)
        a.legend()
    fig.suptitle(title, fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Pooled posterior plot saved to: {savepath}")
    plt.close(fig)
    return fig, ax


# ------------------ Model ------------------
def model_fc(data, prior_specs):
    FC_obs = data["FC_obs"]
    n_obs = FC_obs.shape[0]
    G = npr.sample("G", dist.HalfNormal(scale=prior_specs['scale_G']))#dist.Beta(2.0, 3.0)
    FC_hat_flat = data["forward_fn"](G)   # config-baked forward (custom_vjp FD or autodiff)
    obs_err = data["obs_err"]
    with plate("fc_elements", n_obs):
        npr.sample("FC_obs", dist.Normal(FC_hat_flat, obs_err), obs=FC_obs)
    npr.deterministic("FC_hat", FC_hat_flat)


def model_fcd(data, prior_specs):
    FC_obs = data["FC_obs"]
    n_obs = FC_obs.shape[0]
    G = npr.sample("G", dist.HalfNormal(scale=prior_specs['scale_G']))#dist.Beta(2.0, 3.0)
    FC_hat_flat = data["forward_fn"](G)   # config-baked forward (custom_vjp FD or autodiff)
    obs_err = data["obs_err"]
    with plate("fc_elements", n_obs):
        npr.sample("FC_obs", dist.Normal(FC_hat_flat, obs_err), obs=FC_obs)
    npr.deterministic("FC_hat", FC_hat_flat)


# ------------------ Main ------------------
def main():
    args = parse_args()

    # --- unpack args ---
    G, t_end, tr, cut = args.G, args.t_end, args.tr, args.cut
    wwidth, maxWindows, olap = args.wwidth, args.maxWindows, args.olap
    obs_err, n_prior, n_warmup, n_samples, n_chains = (
        args.obs_err, args.n_prior, args.n_warmup, args.n_samples, args.n_chains
    )
    seed = args.seed
    SC_type = args.SC_type.lower()
    SC_size = args.SC_size
    scale = args.scale
    sampler = args.sampler
    which_stat = args.which_stat.upper()
    resultspath = args.save_dir or utils.results_folder()

    print("Running inference with parameters:")
    print(f"G = {G}, t_end = {t_end}, tr = {tr}, cut = {cut}, wwidth = {wwidth}, "
          f"maxWindows = {maxWindows}, olap = {olap}")
    print(f"obs_err = {obs_err}, n_prior = {n_prior}, n_warmup = {n_warmup}, "
          f"n_samples = {n_samples}, n_chains = {n_chains}")

    # backend (gpu/cpu) + chain_method go into the tag so parallel sweep cells
    # (GPU-vectorized vs CPU-parallel at the same n_chains) don't overwrite each other.
    backend = jax.devices()[0].platform

    # --- parameter tag for filenames ---  # <<< added
    tag = f"G{G}_cut{cut}_tr{tr}_seed{seed}_tend{t_end}_ns{n_samples}_nc{n_chains}_SC_{SC_size}_sampler_{sampler}_which_stat_{which_stat}_{backend}_cm{args.chain_method}_np{args.n_particles}"
    print(f"\nSaving all outputs with tag: {tag}\n")

    # --- file paths ---  # <<< added
    fname_prior_pred = os.path.join(resultspath, f"prior_predictive_G_{tag}.png")
    fname_trace = os.path.join(resultspath, f"trace_G_{tag}.png")
    fname_posteriors = os.path.join(resultspath, f"pooled_posteriors_{tag}.png")
    fname_logjoint = os.path.join(resultspath, f"log_joint_density_{tag}.png")
    fname_postpred_summary = os.path.join(resultspath, f"posterior_predictive_summary_{tag}.png")
    fname_summary_csv = os.path.join(resultspath, f"summary_{tag}.csv")
    fname_netcdf = os.path.join(resultspath, f"inference_results_{tag}.nc")
    fname_benchmark_csv = os.path.join(resultspath, f"benchmark_{tag}.csv")
    fname_benchmark_params_csv = os.path.join(resultspath, f"benchmark_params_{tag}.csv")
    fname_benchmark_json = os.path.join(resultspath, f"benchmark_{tag}.json")

    # --- setup params ---
    if SC_type == "sim":
        SC = nx.to_numpy_array(nx.complete_graph(SC_size))
    elif SC_type == "data":
        datapath = utils.DATA_ROOT
        weights = np.loadtxt(os.path.join(datapath, "weights.txt"))
        weights = weights[:SC_size,:SC_size]
        nn = len(weights)
        SC = jnp.array(weights) / jnp.max(weights)
    else:
        raise ValueError(f"Invalid SC_type '{SC_type}'. Must be 'sim' or 'data'.")
    params = {
        "G": G, "weights": SC, "t_end": t_end,
        "dt": 0.01, "eta": jnp.array([-4.6]), "rv_decimate": 10,
        "noise_amp": 0.037, "tr": 300.0, "seed": seed,  # model BOLD TR; 300 (=ParMPR default,
        # matches pymc) -> ~90 BOLD frames. tr=1.0 gave 30000 frames -> FCD OOM & incomparable.
        "clip_mode": args.clip_mode,
    }

    # --- setup model features  and data --
    if which_stat == "FC":
        wrapper = wrapper_fc
        model = model_fc
    elif which_stat == "FCD":
        wrapper = wrapper_fcd
        model = model_fcd
    else:
        raise ValueError(f"Invalid Functional Connectivity type '{which_stat}'. Must be 'FC' or 'FCD'.")

    # FCD window starts must be computed from the ACTUAL reference-BOLD length after
    # cut+tr subsampling. The model decimates BOLD internally (t_end=30000 -> 90 frames),
    # so (t_end-cut)//tr massively overcounts windows -> a ~30000x30000 FCD matrix -> OOM.
    # Mirror mpr_jax_pymc, which measures the real BOLD length.
    if which_stat == "FCD":
        _ref_bold = mpr_jax.MPR_sde.create(params).run(
            {}, record_rv=False, fast_bold=args.fast_bold)["bold_d"]
        T = int(_ref_bold[cut::tr].shape[0])
    else:
        T = (t_end - cut) // tr
    shift, starts = precompute_shift_and_starts(T, wwidth=30, olap=0.94, stride1=args.fcd_stride1)

    # observed data uses fast_bold/eps/fisher_z for consistency with the model;
    # grad_horizon is irrelevant here (no gradient) so we leave it at 0.
    FC_obs_flat = wrapper(G=G, par=params, cut=cut, tr=tr, starts=starts, nn=nn,
                          grad_horizon=0, fast_bold=args.fast_bold, eps=args.fc_eps,
                          fisher_z=args.fisher_z, keep_negative=args.keep_negative_fc)
    # Single config-baked forward used by the model (custom_vjp FD gradient by
    # default so NUTS/BlackJAX get the reliable finite-difference gradient).
    forward_fn = make_forward_fn(params, which_stat, cut, tr, starts, nn,
                                 grad_horizon=args.grad_horizon, fast_bold=args.fast_bold,
                                 eps=args.fc_eps, grad_method=args.grad_method, fd_h=args.fd_h,
                                 fisher_z=args.fisher_z, keep_negative=args.keep_negative_fc)

    # --- optional per-feature standardization (mirror of pymc): affine (x-mu)/sd on
    # both the model forward and the observed, so a scalar obs_err is a unit-scale
    # sigma. Identity when off -> unchanged behaviour. ---
    if args.standardize_features:
        feat_mu, feat_sd = feature_reference_scatter(
            params, which_stat, cut, tr, starts, nn, fast_bold=args.fast_bold,
            eps=args.fc_eps, n_ref=args.n_ref_seeds, fisher_z=args.fisher_z,
            keep_negative=args.keep_negative_fc)
        print(f"Standardizing {which_stat} features over {args.n_ref_seeds} noise seeds: "
              f"dim={feat_sd.shape[0]}, median feature sd={np.median(feat_sd):.4g}")
        _mu_j, _sd_j = jnp.asarray(feat_mu), jnp.asarray(feat_sd)
        _fwd0 = forward_fn
        forward_fn = lambda G, _f=_fwd0: (_f(G) - _mu_j) / _sd_j
        FC_obs_flat = (jnp.asarray(FC_obs_flat) - _mu_j) / _sd_j

    # --- optional observation noise: make the synthetic observed a real draw from
    # the model (N(0, obs_err)); needed for honest calibration. ---
    if args.noisy_obs:
        _rng_obs = np.random.default_rng(seed)
        FC_obs_flat = jnp.asarray(FC_obs_flat) + jnp.asarray(
            _rng_obs.normal(0.0, obs_err, size=np.asarray(FC_obs_flat).shape), dtype=jnp.float32)
        print(f"Added observation noise N(0, {obs_err}) to observed "
              f"({np.asarray(FC_obs_flat).shape[0]} features).")
    print(FC_obs_flat)
    data = {"FC_obs": FC_obs_flat, "params": params, "obs_err": obs_err, "cut": cut, "tr":tr, "starts":starts, "nn":nn,
            "grad_horizon": args.grad_horizon, "fast_bold": args.fast_bold, "eps": args.fc_eps,
            "forward_fn": forward_fn}
    prior_specs = {"scale_G": scale}

    os.makedirs(resultspath, exist_ok=True)

    # --- prior predictive ---
    rng_key = jax.random.PRNGKey(seed)
    
    prior_predictive = Predictive(model, num_samples=n_prior)
    prior_predictions = prior_predictive(rng_key, data, prior_specs)

    plt.hist(np.asarray(prior_predictions["G"]), bins=20, color="#295785", alpha=0.8)
    plt.axvline(params["G"], color="r", linestyle="--", label="True G")
    plt.title("Prior Predictive Check for G")
    plt.xlabel("G"); plt.legend(frameon=False); plt.tight_layout()
    plt.savefig(fname_prior_pred, dpi=300); plt.close()

    # --- inference ---
    theta_true = np.array([params["G"]])
    runtime = float("nan")

    rng_key, rng_key_run = jax.random.split(rng_key)

    if sampler == "nuts":
        print("Running NUTS inference...")
        tails_5th = jnp.percentile(prior_predictions["G"], 5, axis=0)
        init_to_low_prob = init_to_value(values={"G": tails_5th})
        kernel = NUTS(model, init_strategy=init_to_low_prob)
        mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples,
                    num_chains=n_chains, chain_method=args.chain_method)
        start_time = time.time()
        mcmc.run(rng_key_run, data, prior_specs, extra_fields=("potential_energy", "num_steps", "diverging"))
        runtime = time.time() - start_time
        posterior_samples = az.from_numpyro(mcmc)

        # --- potential energy ---
        lp = -mcmc.get_extra_fields()["potential_energy"]
        plt.figure()
        plt.plot(lp, label="Potential energy")
        plt.xlabel("Iteration"); plt.ylabel("Log joint density")
        plt.title("Potential energy over sampling")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(fname_logjoint, dpi=300)
        plt.close()

        # --- summary CSV ---
        az_summary = az.summary(mcmc, var_names=["G"]).reset_index().rename(columns={"index": "parameter"})
        az_summary["runtime_sec"] = runtime
        az_summary.to_csv(fname_summary_csv, index=False)
        print(f"Summary saved to {fname_summary_csv}")


    elif sampler == "blackjax":
        print("Running NUTS inference via BlackJAX...")
        # Unconstrained potential from the NumPyro model (handles the prior
        # transform); its gradient uses the model's forward_fn -> custom_vjp FD.
        rng_key, rng_key_init = jax.random.split(rng_key_run)
        param_info, potential_fn, postprocess_fn, *_ = initialize_model(
            rng_key_init, model, model_args=(data, prior_specs), dynamic_args=True)
        logdensity_fn = lambda z: -potential_fn(data, prior_specs)(z)
        initial_position = param_info.z

        # Window adaptation (step size + mass matrix) on one chain.
        rng_key, rng_key_warmup = jax.random.split(rng_key)
        warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn)
        (warm_state, parameters), _ = warmup.run(rng_key_warmup, initial_position, num_steps=n_warmup)
        kernel = blackjax.nuts(logdensity_fn, **parameters)

        def inference_loop(rng_key, init_state, n):
            def one(state, key):
                state, info = kernel.step(key, state)
                return state, (state.position, info)
            keys = jax.random.split(rng_key, n)
            _, (positions, infos) = jax.lax.scan(one, init_state, keys)
            return positions, infos

        # n_chains vectorized on-device (vmap), all from the adapted state, distinct RNG.
        rng_key, rng_key_chains = jax.random.split(rng_key)
        chain_keys = jax.random.split(rng_key_chains, n_chains)
        init_state = kernel.init(initial_position)
        start_time = time.time()
        positions, infos = jax.vmap(lambda k: inference_loop(k, init_state, n_samples))(chain_keys)
        jax.block_until_ready(positions)
        runtime = time.time() - start_time

        # Unconstrained z -> constrained params, per (chain, draw); keep just "G".
        samples = jax.vmap(jax.vmap(lambda z: postprocess_fn(data, prior_specs)(z)))(positions)
        posterior = {"G": np.asarray(samples["G"])}   # (chain, draw)
        sample_stats = {
            "diverging": np.asarray(infos.is_divergent),
            "acceptance_rate": np.asarray(infos.acceptance_rate),
            "num_steps": np.asarray(infos.num_integration_steps),
            "lp": -np.asarray(infos.energy),
        }
        posterior_samples = az.from_dict(posterior=posterior, sample_stats=sample_stats)
        mcmc = posterior_samples

        az_summary = az.summary(posterior_samples, var_names=["G"]).reset_index().rename(columns={"index": "parameter"})
        az_summary["runtime_sec"] = runtime
        az_summary.to_csv(fname_summary_csv, index=False)
        print(f"Summary saved to {fname_summary_csv}")


    elif sampler == "pathfinder":
        print("Running Pathfinder inference via BlackJAX...")

        # Initialize the model in unconstrained space
        rng_key, rng_key_init = jax.random.split(rng_key_run)
        param_info, potential_fn, postprocess_fn, *_ = initialize_model(
            rng_key_init, model, model_args=(data, prior_specs), dynamic_args=True
        )
        initial_position = param_info.z  # unconstrained initial position
        print('initial_position',initial_position)
        # Define log density function for Pathfinder
        logdensity_fn = lambda z: -potential_fn(data, prior_specs)(z)  # use original args

        # Run Pathfinder approximation
        rng_key, rng_key_pf = jax.random.split(rng_key)
        pf_state, diagnostics = blackjax.vi.pathfinder.approximate(
            rng_key=rng_key_pf, 
            logdensity_fn=logdensity_fn, 
            initial_position=initial_position, 
            num_samples=200,
            ftol=1e-05,#1e-4, #
            maxiter=30,#200, #
            maxcor=10, #100, 
            gtol=1e-08,#1e-06 #
            #num_steps=100
        )
        print(pf_state)
        print('diagnostics',diagnostics)

        #import pdb; pdb.set_trace() #debug mode for tracing the code and inspect variables manually

        # Sample from the approximate posterior
        rng_key, rng_key_samp = jax.random.split(rng_key)
        samples_unconstrained, _ = blackjax.vi.pathfinder.sample(
            rng_key=rng_key_samp, 
            state=pf_state, 
            num_samples=1000
        )
        print(samples_unconstrained)

        # Transform to constrained (original) space
        samples = jax.vmap(
            lambda z: postprocess_fn(data, prior_specs)(z)
            )(samples_unconstrained)
        print(samples)

        # Convert to ArviZ InferenceData
        posterior_samples = az.from_dict(posterior={k: np.asarray(v) for k, v in samples.items()})
        mcmc = posterior_samples
        print(posterior_samples)


    elif sampler in ("smc_lik", "smc_abc"):
        which = "likelihood-tempered" if sampler == "smc_lik" else "ABC epsilon-tempered"
        print(f"Running {which} SMC ({args.n_particles} particles vmapped on-device, "
              f"{args.n_stages} stages x {args.n_mcmc} MH moves)...")
        fwd = data["forward_fn"]                 # config-baked forward (vmaps over G)
        obs = jnp.asarray(FC_obs_flat)
        scale_G = prior_specs["scale_G"]
        logprior_fn = lambda G: dist.HalfNormal(scale_G).log_prob(G)

        rng_key, k_init, k_smc = jax.random.split(rng_key_run, 3)
        init_particles = dist.HalfNormal(scale_G).sample(k_init, (args.n_particles,))

        if sampler == "smc_lik":
            # exact Normal-likelihood posterior (same target as NUTS/BlackJAX)
            loglik_fn = lambda G: jnp.sum(dist.Normal(fwd(G), obs_err).log_prob(obs))
            start_time = time.time()
            parts, info = run_tempered_smc(k_smc, init_particles, logprior_fn, loglik_fn,
                                           n_stages=args.n_stages, n_mcmc=args.n_mcmc,
                                           rw_step=args.rw_step)
            jax.block_until_ready(parts)
            runtime = time.time() - start_time
        else:  # smc_abc  (likelihood-free)
            distance_fn = lambda G: jnp.sqrt(jnp.sum((fwd(G) - obs) ** 2))
            n_feat = int(obs.shape[0])
            # target epsilon = the PER-FEATURE noise sd = obs_err (NOT obs_err*sqrt(n_feat)).
            # The Gaussian ABC kernel -0.5*(dist/eps)^2 with dist=||sim-obs|| EQUALS the
            # Gaussian log-likelihood -0.5*sum((sim-obs)/obs_err)^2 exactly when eps=obs_err;
            # annealing only to obs_err*sqrt(n_feat) (~sqrt(n_feat)x too loose) leaves the
            # kernel flat over G -> accept~0.94, uninformative posterior. So eps_target=obs_err
            # calibrates ABC to the likelihood (it then recovers G like smc_lik).
            eps_target = args.abc_eps_target if args.abc_eps_target is not None else float(obs_err)
            if args.abc_eps0 is not None:
                eps0 = args.abc_eps0
            else:
                d0 = np.asarray(jax.vmap(distance_fn)(init_particles))
                eps0 = float(np.percentile(d0, 90))          # start loose
            eps0 = max(eps0, eps_target * 3.0)               # guarantee a real decreasing schedule
            eps_schedule = jnp.geomspace(eps0, eps_target, args.n_stages + 1)
            print(f"  ABC epsilon schedule: {eps0:.3g} -> {eps_target:.3g}")
            start_time = time.time()
            parts, info = run_abc_smc(k_smc, init_particles, logprior_fn, distance_fn,
                                      eps_schedule, n_mcmc=args.n_mcmc, rw_step=args.rw_step)
            jax.block_until_ready(parts)
            runtime = time.time() - start_time

        parts_np = np.asarray(parts).reshape(1, -1)          # (chain=1, draw=N particles)
        posterior_samples = az.from_dict(posterior={"G": parts_np})
        mcmc = posterior_samples
        print(f"SMC done in {runtime:.1f}s | posterior G mean={parts_np.mean():.4f} "
              f"sd={parts_np.std():.4f} | mean accept={float(np.mean(np.asarray(info['accept']))):.2f} "
              f"| final ESS={float(np.asarray(info['ess'])[-1]):.1f}")

        az_summary = az.summary(posterior_samples, var_names=["G"]).reset_index().rename(columns={"index": "parameter"})
        az_summary["runtime_sec"] = runtime
        az_summary["n_particles"] = args.n_particles
        az_summary.to_csv(fname_summary_csv, index=False)
        print(f"Summary saved to {fname_summary_csv}")


    elif sampler in ("rwmh", "demc", "slice"):
        # GPU chain-vmapped gradient-free samplers (mcmc_jax). The batched axis is
        # n_chains: N independent chains run at once, so every forward eval is a vmap
        # over chains -> the GPU-batching win, with no gradient (no NUTS pathology).
        names = {"rwmh": "parallel RW-Metropolis", "demc": "DE-Metropolis", "slice": "slice"}
        print(f"Running {names[sampler]} ({n_chains} chains vmapped on-device, "
              f"{n_warmup} tune + {n_samples} draws)...")
        fwd = data["forward_fn"]
        obs = jnp.asarray(FC_obs_flat)
        scale_G = prior_specs["scale_G"]
        logprior_G = lambda G: dist.HalfNormal(scale_G).log_prob(G)
        loglik_G = lambda G: jnp.sum(dist.Normal(fwd(G), obs_err).log_prob(obs))

        rng_key, k_init, k_run = jax.random.split(rng_key_run, 3)
        init_G = dist.HalfNormal(scale_G).sample(k_init, (n_chains,))
        sampler_fn = {"rwmh": run_parallel_rwmh, "demc": run_demc, "slice": run_slice}[sampler]

        start_time = time.time()
        G_draws, info = sampler_fn(k_run, init_G, logprior_G, loglik_G, n_warmup, n_samples)
        jax.block_until_ready(G_draws)
        runtime = time.time() - start_time

        posterior = {"G": np.asarray(G_draws)}          # (chain=n_chains, draw=n_samples)
        posterior_samples = az.from_dict(posterior=posterior)
        mcmc = posterior_samples
        print(f"{sampler} done in {runtime:.1f}s | posterior G mean={float(np.mean(G_draws)):.4f} "
              f"sd={float(np.std(G_draws)):.4f} | accept={float(np.asarray(info['accept'])):.2f}")

        az_summary = az.summary(posterior_samples, var_names=["G"]).reset_index().rename(columns={"index": "parameter"})
        az_summary["runtime_sec"] = runtime
        az_summary.to_csv(fname_summary_csv, index=False)
        print(f"Summary saved to {fname_summary_csv}")



    # --- diagnostics and plots ---
    summary = az.summary(mcmc, var_names=["G"])
    print(summary)
    plot_trace_chains(mcmc, theta_true, ["G"], savepath=fname_trace)
    
    #az_obj = az.from_numpyro(mcmc)
    #print(mcmc['G'])
    #print(az_obj)
    #az.to_netcdf(posterior_samples, fname_netcdf)

    chains_pooled = posterior_samples.posterior["G"].values.reshape(1, -1)
    try:  # plotting is non-essential; never let it crash the run before benchmarks save
        plot_posterior_pooled(["G"], theta_true, prior_predictions, chains_pooled, "Pooled Posteriors", savepath=fname_posteriors)
    except Exception as _e:
        print(f"WARNING: plot_posterior_pooled failed ({type(_e).__name__}: {_e}); continuing to benchmark save.")

    # --- comprehensive benchmark metrics (same schema/file as mpr_jax_pymc.py) ---
    try:
        fwd_eval = make_forward_fn(params, which_stat, cut, tr, starts, nn,
                                   grad_horizon=0, fast_bold=args.fast_bold,
                                   eps=args.fc_eps, grad_method="autodiff")
        obs_np = np.asarray(FC_obs_flat)
        Gmean = float(np.mean(posterior_samples.posterior["G"].values))
        Gmap = float(calcula_map(chains_pooled)[0])
        xs_mean = np.asarray(fwd_eval(jnp.float32(Gmean)))
        xs_map = np.asarray(fwd_eval(jnp.float32(Gmap)))
        rmse = {
            "param_mean": abs(Gmean - float(theta_true[0])),
            "param_map": abs(Gmap - float(theta_true[0])),
            "fit_mean": float(np.sqrt(np.mean((xs_mean - obs_np) ** 2))),
            "fit_map": float(np.sqrt(np.mean((xs_map - obs_np) ** 2))),
        }
        prior_sd = {"G": float(np.std(np.asarray(prior_predictions["G"]).ravel(), ddof=1))}
        meta = {
            "sampler": sampler, "framework": "numpyro/blackjax", "which_stat": which_stat,
            "SC_type": SC_type, "SC_size": SC_size, "G_true": float(params["G"]),
            "t_end": t_end, "cut": cut, "obs_err": obs_err, "seed": seed,
            "n_warmup": n_warmup, "grad_method": args.grad_method, "clip_mode": args.clip_mode,
            "platform": backend, "chain_method": args.chain_method,
            "n_particles": args.n_particles,
            "fcd_stride1": bool(args.fcd_stride1), "fisher_z": bool(args.fisher_z),
            "standardize_features": bool(args.standardize_features), "noisy_obs": bool(args.noisy_obs),
            "keep_negative_fc": bool(args.keep_negative_fc),
        }
        run_row, param_rows = benchmark_metrics(posterior_samples, theta_true, ["G"],
                                                prior_sd, runtime, rmse, meta)
        pd.DataFrame([run_row]).to_csv(fname_benchmark_csv, index=False)
        pd.DataFrame(param_rows).to_csv(fname_benchmark_params_csv, index=False)
        with open(fname_benchmark_json, "w") as fh:
            json.dump({"run": run_row, "params": param_rows}, fh, indent=2)
        print(f"Benchmark metrics saved to: {fname_benchmark_csv}")
        print(f"  runtime={runtime:.1f}s  max_r_hat={run_row['max_r_hat']:.3f}  "
              f"min_ess_bulk={run_row['min_ess_bulk']:.1f}  "
              f"rmse_param_mean={run_row['rmse_param_mean']:.4f}")
    except Exception as e:
        print(f"WARNING: benchmark_metrics failed: {e}")

    print(f"\nAll results and plots saved in:\n{resultspath}\n")


if __name__ == "__main__":
    main()
