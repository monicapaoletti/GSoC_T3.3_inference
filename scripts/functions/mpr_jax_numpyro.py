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
    parser.add_argument("--obs_err", type=float, default=0.1)
    parser.add_argument("--n_prior", type=int, default=100)
    parser.add_argument("--n_warmup", type=int, default=20)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--n_chains", type=int, default=1)
    parser.add_argument("--scale", type=int, default=1, help="prior scale")
    parser.add_argument("--sampler", type=str, default="pathfinder",
                    choices=["nuts", "pathfinder"], help="Choose inference method")

    # Misc
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=int(time.time()))
    # SC matrix type
    parser.add_argument("--SC_type", type=str, default="data",
                    help="Type of structural connectivity: 'sim' for simulated, 'data' for real data")
    parser.add_argument("--SC_size", type=int, default=88,
                        help="Number of nodes if using simulated SC") #change to 6 or 10 for small networks

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
@timer
def wrapper_fc(G, par, cut, tr, starts, nn):
    par = deepcopy(par)
    par["G"] = G
    sde = mpr_jax.MPR_sde.create(par)
    data = sde.run({})
    bold_d = data["bold_d"]

    FC_full = get_fc(bold_d[int(cut):].T)   # <<< ensure int(cut)
    #print(FC_full)
    tri_idx = jnp.triu_indices(FC_full.shape[0], k=1) 
    return FC_full[tri_idx]

@timer
def wrapper_fcd(G, par, cut, tr, starts, nn):
    par = deepcopy(par)
    par["G"] = G
    sde = mpr_jax.MPR_sde.create(par)
    data = sde.run({})
    bold_d = data["bold_d"]

    bold_d_sub = bold_d[cut::tr].T

    FCD_full = extract_FCD_jax_jitted(bold_d_sub, starts, nn, wwidth=30, olap=0.94) #FC_full = get_fc(bold_d[int(cut):].T)   # <<< ensure int(cut)
    #print(FCD_full)
    tri_idx = jnp.triu_indices(FCD_full.shape[0], k=30)
    return FCD_full[tri_idx]


# ------------------ Helpers ------------------
def calcula_map(chains_):
    params_map = []
    for i in range(int(chains_.shape[0])):
        y = chains_[i]
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
    params_sim = dict(data["params"])
    params_sim["G"] = G
    FC_hat_flat = wrapper_fc(G, params_sim, cut=data["cut"], tr=data["tr"], starts=data["starts"], nn=data["nn"])
    obs_err = data["obs_err"]
    with plate("fc_elements", n_obs):
        npr.sample("FC_obs", dist.Normal(FC_hat_flat, obs_err), obs=FC_obs)
    npr.deterministic("FC_hat", FC_hat_flat)


def model_fcd(data, prior_specs):
    FC_obs = data["FC_obs"]
    n_obs = FC_obs.shape[0]
    G = npr.sample("G", dist.HalfNormal(scale=prior_specs['scale_G']))#dist.Beta(2.0, 3.0)
    params_sim = dict(data["params"])
    params_sim["G"] = G
    FC_hat_flat = wrapper_fcd(G, params_sim, cut=data["cut"], tr=data["tr"], starts=data["starts"], nn=data["nn"])
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

    # --- parameter tag for filenames ---  # <<< added
    tag = f"G{G}_cut{cut}_tr{tr}_seed{seed}_tend{t_end}_ns{n_samples}_nc{n_chains}_SC_{SC_size}_sampler_{sampler}_which_stat_{which_stat}"
    print(f"\nSaving all outputs with tag: {tag}\n")

    # --- file paths ---  # <<< added
    fname_prior_pred = os.path.join(resultspath, f"prior_predictive_G_{tag}.png")
    fname_trace = os.path.join(resultspath, f"trace_G_{tag}.png")
    fname_posteriors = os.path.join(resultspath, f"pooled_posteriors_{tag}.png")
    fname_logjoint = os.path.join(resultspath, f"log_joint_density_{tag}.png")
    fname_postpred_summary = os.path.join(resultspath, f"posterior_predictive_summary_{tag}.png")
    fname_summary_csv = os.path.join(resultspath, f"summary_{tag}.csv")
    fname_netcdf = os.path.join(resultspath, f"inference_results_{tag}.nc")

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
        "noise_amp": 0.037, "tr": 1.0, "seed": seed
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

    T = (t_end-cut)//tr
    shift, starts = precompute_shift_and_starts(T, wwidth=30, olap=0.94)

    FC_obs_flat = wrapper(G=G, par=params, cut=cut, tr=tr, starts=starts, nn=nn)
    print(FC_obs_flat)
    data = {"FC_obs": FC_obs_flat, "params": params, "obs_err": obs_err, "cut": cut, "tr":tr, "starts":starts, "nn":nn}
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

    rng_key, rng_key_run = jax.random.split(rng_key)

    if sampler == "nuts":
        print("Running NUTS inference...")
        tails_5th = jnp.percentile(prior_predictions["G"], 5, axis=0)
        init_to_low_prob = init_to_value(values={"G": tails_5th})
        kernel = NUTS(model, init_strategy=init_to_low_prob)
        mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples,
                    num_chains=n_chains, chain_method="parallel")
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



    # --- diagnostics and plots ---
    summary = az.summary(mcmc, var_names=["G"])
    print(summary)
    plot_trace_chains(mcmc, theta_true, ["G"], savepath=fname_trace)
    
    #az_obj = az.from_numpyro(mcmc)
    #print(mcmc['G'])
    #print(az_obj)
    #az.to_netcdf(posterior_samples, fname_netcdf)

    chains_pooled = posterior_samples.posterior["G"].values.reshape(1, -1)
    plot_posterior_pooled(["G"], theta_true, prior_predictions, chains_pooled, "Pooled Posteriors", savepath=fname_posteriors)


    print(f"\nAll results and plots saved in:\n{resultspath}\n")


if __name__ == "__main__":
    main()
