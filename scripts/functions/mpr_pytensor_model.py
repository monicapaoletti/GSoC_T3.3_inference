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
from scipy.integrate import odeint
from scipy.optimize import least_squares

# ------------------ Argument parser ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run MPR JAX NumPyro inference")

    # Model/simulation parameters
    parser.add_argument("--G", type=float, default=0.33)

    # Statistics
    parser.add_argument("--which_stat", type=str, default="FCD")

    # scale the next three parameters together 
    parser.add_argument("--t_end", type=int, default=30000)
    parser.add_argument("--tr", type=int, default=50)
    parser.add_argument("--cut", type=int, default=1000)

    parser.add_argument("--wwidth", type=int, default=30)
    parser.add_argument("--maxWindows", type=int, default=200)
    parser.add_argument("--olap", type=float, default=0.5)

    # Inference settings
    parser.add_argument("--obs_err", type=float, default=0.01)
    parser.add_argument("--n_prior", type=int, default=10000)
    parser.add_argument("--n_warmup", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--n_chains", type=int, default=4)

    parser.add_argument("--sampler", type=str, default="smcabc",
                    choices=["slice", "metropolis", "demetropolisz", "demetropolis", "smclik", "smcabc"], help="Choose inference method")
    parser.add_argument("--epsilon", type=float, default=1, help="epsilon parameter for smc abc algorithm")

    # Misc
    parser.add_argument("--seed", type=int, default=int(time.time()))
    # SC matrix type
    parser.add_argument("--SC_type", type=str, default="data",
                    help="Type of structural connectivity: 'sim' for simulated, 'data' for real data")
    parser.add_argument("--SC_size", type=int, default=6,
                        help="Number of nodes in SC") #change to 6 or 10 for small networks

    return parser.parse_args()

args = parse_args()

# local modules
import utils
from FCD_jax import *    
import mpr_jax
mpr_jax = __import__("mpr_jax")

# --- setup params ---
if args.SC_type == "sim":
    SC = nx.to_numpy_array(nx.complete_graph(args.SC_size))
elif args.SC_type == "data":
    datapath = utils.DATA_ROOT
    weights = np.loadtxt(os.path.join(datapath, "weights.txt"))
    weights = weights[:args.SC_size, :args.SC_size]
    NN = len(weights)
    SC = jnp.array(weights) / jnp.max(weights)

T = (args.t_end - args.cut)//args.tr
shift, STARTS = precompute_shift_and_starts(T, wwidth=30, olap=0.94)

PARAMS = {
    "G": args.G,
    "weights": SC,
    "t_end": args.t_end,
    "dt": 0.01,
    "eta": jnp.array([-4.6]),
    "rv_decimate": 10,
    "noise_amp": 0.037,
    "tr": 1.0,
    "seed": args.seed,
}
CUT = args.cut
TR_ = args.tr
OBS_ERR = args.obs_err


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

@timer
def wrapper_fcd(G, par, cut, tr, starts, nn):
    par = deepcopy(par)
    par["G"] = G
    sde = mpr_jax.MPR_sde.create(par)
    data = sde.run({})
    bold_d = data["bold_d"]

    bold_d_sub = bold_d[cut::tr].T

    FCD_full = extract_FCD_jax_jitted(bold_d_sub, starts, nn, wwidth=30, olap=0.94) 
    
    tri_idx = jnp.triu_indices(FCD_full.shape[0], k=30) 
    return FCD_full[tri_idx]


# decorator with input and output types a Pytensor double float tensors
@as_op(itypes=[pt.dvector], otypes=[pt.dvector])
def pytensor_forward_model_matrix(G):

    if PARAMS is None:
        raise RuntimeError("Globals not set: set PARAMS and STARTS before building model.")
    result =  wrapper_fcd(G=float(np.atleast_1d(G).item()), 
                          par=PARAMS, 
                          cut=CUT, 
                          tr=TR_, 
                          starts=STARTS, 
                          nn=NN)
    
    return np.asarray(result, dtype=np.float64).flatten()

