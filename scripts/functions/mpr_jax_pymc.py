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

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import arviz as az
import errno

import pymc as pm
import pytensor
import pytensor.tensor as pt

from pytensor.compile.ops import as_op
from scipy.optimize import least_squares

# local modules
import utils
from FCD_jax import *    
import mpr_jax
mpr_jax = __import__("mpr_jax")
#import config
#from mpr_pytensor_model import pytensor_forward_model_matrix

# ------------------ Argument parser ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run MPR JAX NumPyro inference")

    # Model/simulation parameters
    parser.add_argument("--G", type=float, default=0.33)

    # Statistics
    parser.add_argument("--which_stat", type=str, default="FCD")

    # scale the next three parameters together 
    parser.add_argument("--t_end", type=int, default=30000) #300000
    parser.add_argument("--tr", type=int, default=50) #500
    parser.add_argument("--cut", type=int, default=1000) #10000

    parser.add_argument("--wwidth", type=int, default=30)
    parser.add_argument("--maxWindows", type=int, default=200)
    parser.add_argument("--olap", type=float, default=0.94)

    # Inference settings
    parser.add_argument("--obs_err", type=float, default=0.01)
    parser.add_argument("--n_prior", type=int, default=10000)
    parser.add_argument("--n_warmup", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--n_chains", type=int, default=4)

    parser.add_argument("--sampler", type=str, default="smcabc",
                    choices=["slice", "metropolis", "demetropolisz", "demetropolis", "smclik", "smcabc"], help="Choose inference method")
    parser.add_argument("--epsilon", type=float, default=10, help="epsilon parameter for smc abc algorithm")
    parser.add_argument("--threshold", type=float, default=0.4, help="threshold parameter for smc algorithm")
    parser.add_argument("--correlation_threshold", type=float, default=0.05, help="correlation_threshold parameter for smc algorithm")

    # Misc
    parser.add_argument("--seed", type=int, default=int(time.time()))
    # SC matrix type
    parser.add_argument("--SC_type", type=str, default="data",
                    help="Type of structural connectivity: 'sim' for simulated, 'data' for real data")
    parser.add_argument("--SC_size", type=int, default=6,
                        help="Number of nodes in SC") #change to 6 or 10 for small networks

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
    
    tri_idx = jnp.triu_indices(FC_full.shape[0], k=1) 
    return FC_full[tri_idx]

#@timer
def wrapper_fcd(G, weights, t_end, cut, tr, starts, nn):
    #par = deepcopy(par)
    par={}
    par["G"] = G
    par['weights'] = weights
    par['t_end'] = t_end
    sde = mpr_jax.MPR_sde.create(par)
    data = sde.run({})
    bold_d = data["bold_d"]

    bold_d_sub = bold_d[cut::tr].T
    #print(starts,nn)
    FCD_full = extract_FCD_jax_jitted(bold_d_sub, starts, nn, wwidth=30, olap=0.94) 
    
    tri_idx = jnp.triu_indices(FCD_full.shape[0], k=30) 
    #print(len(bold_d),len(bold_d_sub.T),np.shape(FCD_full),len(FCD_full[tri_idx]))
    return FCD_full[tri_idx]


# decorator with input and output types a Pytensor double float tensors


@as_op(itypes=[pt.dvector, pt.dmatrix, pt.dscalar, pt.dscalar, pt.dscalar, pt.dvector, pt.dscalar], otypes=[pt.dvector])
def pytensor_forward_model_matrix(G, weights, t_end, cut, tr, starts, nn):
    # inputs inside op are numpy arrays / scalars
    G_val = float(np.atleast_1d(G))
    weights_np = np.asarray(weights, dtype=np.float64)
    t_end_val = int(float(t_end))
    cut_val = int(float(cut))
    tr_val = int(float(tr))
    starts_np = np.asarray(starts, dtype=np.int64)   # if you need ints inside wrapper
    nn_val = int(float(nn))

    # call your wrapper which expects python/numpy/jax types:
    return np.asarray(
        wrapper_fcd(G_val, weights_np, t_end_val, cut_val, tr_val, starts_np, nn_val),
        dtype=np.float64
    ).flatten()

# ------------------ Helpers ------------------
def tails_percentile(my_var_names, prior_predictions, thr):

    tails_xth_percentile = {}
    for key, value in prior_predictions.items():
        if key in my_var_names:
            sorted_values = np.sort(value)[0, :] if value.shape[0] == 1 else np.sort(value)
            top_xth_percentile = sorted_values[int(0.05 * len(sorted_values))]
            tails_xth_percentile[key] = np.array(top_xth_percentile)
    return tails_xth_percentile

def plot_trace(trace, my_var_names, theta_true, sampler, fname=None):
    axes = az.plot_trace(trace,var_names=my_var_names,compact=True, kind="trace",
        backend_kwargs={"figsize": (10, 5), "layout": "constrained"},)

    for ax, true_val in zip(axes[:, 0], theta_true):
        ax.axvline(x=true_val, color='red', linestyle='--')
    for ax, true_val in zip(axes[:, 1], theta_true):
        ax.axhline(y=true_val, color='red', linestyle='--')
        
    plt.suptitle(f"Trace Plot {sampler}")
    plt.tight_layout()

    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Trace plot saved to: {fname}")
    else:
        plt.show()

    return axes   

def rmse_fit_mean(wrapper, trace, theta_true, my_var_names, weights, t_end, cut, tr, starts, nn, FC_obs_flat):

    theta_mean=np.mean(az.extract(trace).to_dataframe()[my_var_names], axis=0).to_numpy()
    theta_err=np.sqrt(np.mean(theta_mean-theta_true)**2)
    xs_= wrapper(G=theta_mean, weights=weights, t_end=t_end, cut=cut, tr=tr, starts=starts, nn=nn)
    xs_err=np.sqrt(np.mean((xs_-FC_obs_flat)**2))
    
    return (theta_err), np.mean(xs_err)

def calcula_map(chains_):

    params_map = []
    for i in range(int(chains_.shape[0])):
        y = chains_[i]
        hist, bin_edges = np.histogram(y, bins=50)
        x_value_at_peak = (bin_edges[np.argmax(hist)] + bin_edges[np.argmax(hist) + 1]) / 2
        params_map.append(x_value_at_peak)
    return params_map

def rmse_fit_map(wrapper, trace, theta_true, my_var_names, weights, t_end, cut, tr, starts, nn, FC_obs_flat):

    chains_pooled = trace.posterior["G"].values.reshape(1, -1)
    theta_map=calcula_map(chains_pooled)
    theta_map =jnp.array(theta_map)
    theta_err=np.sqrt(np.mean(theta_map-theta_true)**2)
    xs_= wrapper(G=theta_map, weights=weights, t_end=t_end, cut=cut, tr=tr, starts=starts, nn=nn)
    xs_err=np.sqrt(np.mean((xs_-FC_obs_flat)**2))
    
    return (theta_err), np.mean(xs_err)

def tails_percentile(my_var_names, prior_predictions, thr):

    tails_xth_percentile = {}
    for key, value in prior_predictions.items():
        if key in my_var_names:
            sorted_values = np.sort(value)[0, :] if value.shape[0] == 1 else np.sort(value)
            top_xth_percentile = sorted_values[int(0.05 * len(sorted_values))]
            tails_xth_percentile[key] = np.array(top_xth_percentile)
    return tails_xth_percentile


# ------------------ Main ------------------
def main():
    args = parse_args()

    # --- unpack args ---
    G, t_end, tr, cut = args.G, args.t_end, args.tr, args.cut
    wwidth, maxWindows, olap = args.wwidth, args.maxWindows, args.olap
    obs_err, n_prior, n_warmup, n_samples, n_chains, epsilon, threshold, correlation_threshold = (
        args.obs_err, args.n_prior, args.n_warmup, args.n_samples, args.n_chains, args.epsilon, args.threshold, args.correlation_threshold
    )
    seed = args.seed
    SC_type = args.SC_type.lower()
    SC_size = args.SC_size
    sampler = args.sampler
    which_stat = args.which_stat.upper()
    resultspath = utils.results_folder()

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
    
    T = (t_end-cut)//tr
    shift, starts = precompute_shift_and_starts(T, wwidth=wwidth, olap=olap)

    params = {
        "G": G, "weights": SC, "t_end": t_end, "tr": 1.0, "seed": seed
    }

    print("Running inference with parameters:")
    print(f"G = {G}, t_end = {t_end}, tr = {tr}, cut = {cut}, wwidth = {wwidth}, "
          f"maxWindows = {maxWindows}, olap = {olap}")
    print(f"obs_err = {obs_err}, n_prior = {n_prior}, n_warmup = {n_warmup}, "
          f"n_samples = {n_samples}, n_chains = {n_chains}")

    # --- parameter tag for filenames ---  
    tag = f"G{G}_cut{cut}_tr{tr}_seed{seed}_tend{t_end}_ns{n_samples}_nc{n_chains}_SC_{SC_size}_sampler_{sampler}_which_stat_{which_stat}"
    print(f"\nSaving all outputs with tag: {tag}\n")

    # --- file paths ---  
    fname_prior_pred = os.path.join(resultspath, f"prior_predictive_G_{tag}.png")
    fname_trace = os.path.join(resultspath, f"trace_G_{tag}.png")
    fname_summary_csv = os.path.join(resultspath, f"summary_{tag}.csv")
    fname_summary_RMSE_csv = os.path.join(resultspath, f"summary_RMSE_{tag}.csv")
    fname_netcdf = os.path.join(resultspath, f"inference_results_{tag}.nc")

    # --- setup model features  and data --
    if which_stat == "FC":
        wrapper = wrapper_fc
        #model = model_fc
    elif which_stat == "FCD":
        wrapper = wrapper_fcd
        #model = model_fcd
    else:
        raise ValueError(f"Invalid Functional Connectivity type '{which_stat}'. Must be 'FC' or 'FCD'.")


    # --- Generate observed data ---
    print("Using wrapper:", wrapper.__name__)
    #print(G,params,cut,tr,starts,nn)
    FC_obs_flat = wrapper(G, weights, t_end, cut, tr, starts, nn)#(G=G, par=params, cut=cut, tr=tr, starts=starts, nn=nn)
    #print('FC_obs_flat_shape',np.shape(FC_obs_flat))
    data = {"FC_obs": FC_obs_flat, "params": params, "obs_err": obs_err, "cut": cut, "tr":tr, "starts":starts, "nn":nn}
    prior_specs = {"mu_G" : 0.5, "sigma": 0.7, "lower": 0}
    my_var_names = ['G']

    with pm.Model() as prior_model:
        G = pm.TruncatedNormal("G", mu=prior_specs["mu_G"], sigma=prior_specs["sigma"], lower=prior_specs["lower"])

    with prior_model:
        prior_predict= pm.sample_prior_predictive(n_prior)

    az.plot_posterior(prior_predict, var_names=["G"], group="prior", kind="hist", hdi_prob=0.94, bins=50)
    plt.tight_layout()
    plt.savefig(fname_prior_pred, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Prior predictive plot saved to: {fname_prior_pred}")

    tails_5th_percentile=tails_percentile(my_var_names, prior_predict.prior, 0.05) 

    os.makedirs(resultspath, exist_ok=True)

    with pm.Model() as model:
        # Priors
        G = pm.TruncatedNormal("G", mu=prior_specs["mu_G"], lower=prior_specs["lower"])
        
        params_samples = [G]
        
        # Ode solution function
        #FC_hat_flat = pytensor_forward_model_matrix(pm.math.stack(params_samples))
        FC_hat_flat = pytensor_forward_model_matrix(pm.math.stack(params_samples), 
                                                    pt.as_tensor_variable(np.array(weights)),   
                                                    pt.as_tensor_variable(np.array(float(t_end))),
                                                    pt.as_tensor_variable(np.array(float(cut))),
                                                    pt.as_tensor_variable(np.array(float(tr))),
                                                    pt.as_tensor_variable(np.asarray(starts, dtype=np.float64)),    
                                                    pt.as_tensor_variable(np.array(float(nn)))
                                                    )
                                                        
        # Likelihood
        pm.Normal("FC_obs", mu=FC_hat_flat, sigma=obs_err, observed=FC_obs_flat)

    vars_list = list(model.values_to_rvs.keys())[:-1]

    # --- inference ---
    theta_true = np.array([params["G"]])

    rng_key = jax.random.PRNGKey(seed)

    rng_key, rng_key_run = jax.random.split(rng_key)

    if sampler == "slice":

        sampler = "Slice Sampler"
        start_time = time.time()
        with model:
            trace_slice = pm.sample(step=[pm.Slice(vars_list)], tune=n_warmup, draws=n_samples, chains=n_chains, 
                                    initvals={var_name: tails_5th_percentile[var_name] for var_name in my_var_names}
                                ) 
        crudetime_slice=time.time() - start_time
        print("---running took: %s seconds ---" % crudetime_slice)
        trace = trace_slice

    elif sampler == "metropolis":

        sampler = "Metropolis"
        start_time = time.time()
        with model:
            trace_M = pm.sample(step=[pm.Metropolis()], tune=n_warmup, draws=n_samples, chains=n_chains,
                            initvals={var_name: tails_5th_percentile[var_name] for var_name in my_var_names}
                            )
        crudetime_M=time.time() - start_time
        print("---running took: %s seconds ---" % crudetime_M)
        trace = trace_M

    elif sampler == "demetropolisz":

        sampler = "DE MetropolisZ"
        start_time = time.time()
        with model:
            trace_DEMZ = pm.sample(step=[pm.DEMetropolisZ()], tune=n_warmup, draws=n_samples, chains=n_chains,
                                    initvals={var_name: tails_5th_percentile[var_name] for var_name in my_var_names}
                                ) 
        crudetime_DEMZ=time.time() - start_time
        print("---running took: %s seconds ---" % crudetime_DEMZ)
        trace = trace_DEMZ

    elif sampler == "demetropolis":

        sampler = "DEMetropolis"
        start_time = time.time()
        with model:
            trace_DEM = pm.sample(step=[pm.DEMetropolis()],  draws=n_samples, chains=n_chains,
                                initvals={var_name: tails_5th_percentile[var_name] for var_name in my_var_names}
                                )
        crudetime_DEM=time.time() - start_time
        print("---running took: %s seconds ---" % crudetime_DEM)
        trace = trace_DEM

    elif sampler == "smclik":

        sampler = "SMC with Likelihood"
        start_time = time.time()
        with model:
            trace_SMC_like = pm.sample_smc(draws=n_samples,         
                                        kernel=pm.smc.IMH,  
                                        threshold=threshold,  
                                        correlation_threshold=correlation_threshold,  
                                        progressbar=False, 
                                        chains=n_chains,
                                        cores=n_chains
                                        )
        crudetime_SMC_like=time.time() - start_time
        print("---running took: %s seconds ---" % crudetime_SMC_like)
        trace = trace_SMC_like

    elif sampler == "smcabc":

        def simulator_forward_model(rng,  G, size=None):
            theta = G 
            #mu = wrapper(G=G.item(), par=params, cut=cut, tr=tr, starts=starts, nn=nn).reshape(-1, 1)
            mu = wrapper(G=G.item(), weights=weights, t_end=t_end, cut=cut, tr=tr, starts=starts, nn=nn).reshape(-1, 1)
            return rng.normal(mu, obs_err)
        
        with pm.Model() as model:
            # Priors
            G = pm.TruncatedNormal("G", mu=prior_specs["mu_G"], lower=prior_specs["lower"])
            
            params_samples=[G]
            
            pm.Simulator(
                "FC_obs",
                simulator_forward_model,
                params=params_samples,
                epsilon=epsilon,
                observed=FC_obs_flat,)
            
        sampler = f"SMC_epsilon = {epsilon}"

        start_time = time.time()

        with model:
            # Run the SMC sampler
            trace_SMC_e = pm.sample_smc(draws=n_samples,         
                                        kernel=pm.smc.IMH,  
                                        threshold=threshold,  #default 0.5
                                        correlation_threshold=correlation_threshold,  #defaul 0.01
                                        progressbar=False, 
                                        chains=n_chains,
                                        cores=n_chains)
        crudetime_SMC_e1=time.time() - start_time

        print("---running took: %s seconds ---" % crudetime_SMC_e1)
        trace = trace_SMC_e

    else: 
        ValueError(f"Invalid sampler: {sampler}")

    #print(trace.sample_stats.beta)

    az.to_netcdf(trace.posterior, fname_netcdf)
    print(f"Trace saved to: {fname_netcdf}")

    summary_df = az.summary(trace)
    summary_df.to_csv(fname_summary_csv)
    print(summary_df)
    print(f"Summary saved to: {fname_summary_csv}")

    plot_trace(trace, my_var_names, theta_true, sampler, fname=fname_trace)

    rmse_paramsmean, rmse_fitmean  = rmse_fit_mean(wrapper, trace, theta_true, my_var_names, weights, t_end, cut, tr, starts, nn, FC_obs_flat)
    print ('RMSE to true parameters', rmse_paramsmean),
    print ('RMSE to true observation', rmse_fitmean)

    rmse_paramsmap, rmse_fitmap  = rmse_fit_map(wrapper, trace, theta_true, my_var_names, weights, t_end, cut, tr, starts, nn, FC_obs_flat)
    print ('RMSE to true parameters', rmse_paramsmap),
    print ('RMSE to true observation', rmse_fitmap)

    summary_data = {
        "RMSE_param_mean": [rmse_paramsmean],
        "RMSE_fit_mean": [rmse_fitmean],
        "RMSE_param_map": [rmse_paramsmap],
        "RMSE_fit_map": [rmse_fitmap],
    }

    summary_df_RMSE = pd.DataFrame(summary_data)
    summary_df_RMSE.to_csv(fname_summary_RMSE_csv, index=False)
    print(f"RMSE summary saved to: {fname_summary_RMSE_csv}")

if __name__ == "__main__":

    main()
