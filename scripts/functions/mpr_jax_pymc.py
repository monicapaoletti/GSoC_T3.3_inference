#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference script for FC + FCD statistics using PyMC + JAX.
"""

import sys
import os
import time
import json
import warnings
from copy import deepcopy
import argparse

import numpy as np
# GPU memory: disable XLA's default ~75% preallocation so repeated JAX calls (and
# any multi-process use) don't grab the whole device. No-op on CPU. Must be set
# before importing jax; override by exporting XLA_PYTHON_CLIENT_PREALLOCATE yourself.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
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
pytensor.config.floatX = "float32"

from pytensor.compile.ops import as_op
from pytensor.graph.op import Op
from pytensor.graph.basic import Apply
from pytensor.link.jax.dispatch import jax_funcify   # for nuts_sampler="blackjax"
from scipy.optimize import least_squares

# local modules
import utils
import FCD_jax  
import mpr_jax
mpr_jax = __import__("mpr_jax")
#import mpr_jax_old as mpr_jax

# ------------------ Argument parser ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run MPR JAX NumPyro inference")

    # Model/simulation parameters
    parser.add_argument("--G", type=float, default=0.33)

    # Statistics
    parser.add_argument("--which_stat", type=str, default="FCD")

    # scale the next three parameters together 
    parser.add_argument("--t_end", type=int, default=30000) #300000
    parser.add_argument("--tr", type=int, default=300) #500
    parser.add_argument("--cut", type=int, default=30) #10000

    parser.add_argument("--wwidth", type=int, default=30)
    parser.add_argument("--maxWindows", type=int, default=200)
    parser.add_argument("--olap", type=float, default=0.94)

    # Inference settings
    parser.add_argument("--obs_err", type=float, default=1.0,
                        help="Gaussian likelihood sigma on the (FC/FCD) features. Default 1.0 "
                             "matches vbjax's dist.Normal(mu, 1); kept identical across scripts.")
    parser.add_argument("--standardize_features", action=argparse.BooleanOptionalAction, default=True,
                        help="Z-score each FC/FCD feature by its scatter across noise seeds so a "
                             "unit obs_err is meaningful and FC vs FCD sit on the same scale "
                             "(applied congruently to observed, likelihood, FD-grad and SMC).")
    parser.add_argument("--n_ref_seeds", type=int, default=8,
                        help="Number of noise seeds used to estimate per-feature scatter for "
                             "--standardize_features (extra full sims run once at setup).")
    parser.add_argument("--noisy_obs", action=argparse.BooleanOptionalAction, default=True,
                        help="Add N(0, obs_err) observation noise to the synthetic observed data so "
                             "it is a genuine draw from the model (needed for honest calibration).")
    parser.add_argument("--fisher_z", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply arctanh (Fisher z) to correlation features before the Gaussian "
                             "likelihood: variance-stabilizes and unbounds them so fixed-sigma is "
                             "well-specified (no extra sims needed). Applied inside the forward.")
    parser.add_argument("--keep_negative_fc", action=argparse.BooleanOptionalAction, default=True,
                        help="Keep negative (anti-)correlations instead of the default ReLU that "
                             "zeroes them: more information, no zero-variance-feature spike, smoother.")
    parser.add_argument("--n_prior", type=int, default=10000)
    parser.add_argument("--mu_G", type=float, default=0.3)
    parser.add_argument("--sigma_G", type=float, default=0.5)
    parser.add_argument("--lower_G", type=float, default=0.0)

    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--n_chains", type=int, default=4)
    parser.add_argument("--sample_cores", type=int, default=0,
                        help="Parallel sampling processes. 0=auto (=n_chains on CPU, 1 on GPU). "
                             "Set 1 to run chains sequentially in-process (much lower peak RAM; "
                             "avoids XLA 'Cannot allocate memory' with a big t_end).")

    parser.add_argument("--sampler", type=str, default="smcabc",
                    choices=["slice", "metropolis", "demetropolisz", "demetropolis", "smclik", "smcabc", "nuts", "blackjax", "numpyro"], help="Choose inference method")
    parser.add_argument("--epsilon", type=float, default=10, help="epsilon parameter for smc abc algorithm")
    parser.add_argument("--threshold", type=float, default=0.5, help="threshold parameter for smc algorithm")
    parser.add_argument("--correlation_threshold", type=float, default=0.05, help="correlation_threshold parameter for smc algorithm")

    # Misc
    parser.add_argument("--seed", type=int, default=int(time.time()))
    # SC matrix type
    parser.add_argument("--SC_type", type=str, default="data",
                    help="Type of structural connectivity: 'sim' for simulated, 'data' for real data")
    parser.add_argument("--SC_size", type=int, default=6,
                        help="Number of nodes in SC")

    # --- forward-model speed/stability knobs (mpr_jax) ---
    parser.add_argument("--clip_mode", type=str, default="hard", choices=["hard", "soft", "none"],
                        help="Derivative bounding: hard=jnp.clip, soft=tanh (stable across the transition), none=unbounded.")
    parser.add_argument("--fast_bold", action="store_true",
                        help="Integrate BOLD once per rv_decimate block (approx, ~1.4x faster).")
    parser.add_argument("--fc_eps", type=float, default=0.0,
                        help="Stabilise FC/FCD correlation denominator (e.g. 1e-6) for NaN-safety.")
    parser.add_argument("--fcd_stride1", action=argparse.BooleanOptionalAction, default=True,
                        help="Use vbi-style FCD windowing (shift=1 sample) instead of the "
                             "overlap-fraction shift; pairs with the hardcoded triu offset k=30. "
                             "Pass --no-fcd_stride1 to revert to the prior overlap windowing.")
    # --- gradient-based NUTS options (only used when --sampler nuts) ---
    parser.add_argument("--grad_method", type=str, default="fd", choices=["fd", "autodiff"],
                        help="Gradient for NUTS: fd=finite differences (correct in chaos), autodiff (unreliable past bifurcation).")
    parser.add_argument("--fd_h", type=float, default=1e-2, help="Finite-difference step for grad_method=fd.")
    parser.add_argument("--grad_horizon", type=int, default=0,
                        help="Truncated-BPTT horizon (only affects grad_method=autodiff).")

    return parser.parse_args()


# ------------------ Timer decorator ------------------
def timer(func):
    def wrapper_timer(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.2f} seconds")
        return result
    return wrapper_timer


# ------------------ Forward model ------------------
# There is a SINGLE forward model for every sampler, built in main() as a set of
# config-explicit closures (see make_forward_fns) wrapped in ForwardGradOp. The
# statistic (FC/FCD) and the mpr_jax knobs (clip_mode/fast_bold/eps/grad_horizon)
# are captured by value, so (a) the whole script is congruent with --which_stat
# and (b) the config survives cloudpickle to spawned multiprocessing workers.
# ForwardGradOp exposes a gradient wrt the scalar G, computed either by FINITE
# DIFFERENCES (grad_method="fd", correct for the chaotic regime) or by AUTODIFF
# (grad_method="autodiff", exact but unreliable past the bifurcation); gradient-
# free samplers simply never call it.

def _fisher_z(v, eps=1e-4):
    """Fisher z-transform arctanh(r) for correlation features. Variance-stabilizes
    (a correlation's own variance depends on its value; arctanh makes it ~constant)
    and maps [-1,1] -> R so a fixed-sigma Gaussian likelihood is well-specified.
    Clip keeps arctanh(+/-1) finite. Applied INSIDE the forward so the FD gradient
    and SMC paths all see the transformed feature."""
    return jnp.arctanh(jnp.clip(v, -1.0 + eps, 1.0 - eps))


def _forward_stat_jax(G, weights, t_end, cut, which_stat, starts, nn,
                      clip_mode="hard", fast_bold=False, eps=0.0, grad_horizon=0,
                      fisher_z=False, keep_negative=False):
    """JAX forward: simulate at coupling G and return the flat FC/FCD statistic.

    Config (clip_mode/fast_bold/eps/grad_horizon/fisher_z/keep_negative) is passed
    EXPLICITLY rather than read from a module global, so the closures that capture it
    (built in main) are serialized by value to spawned multiprocessing workers.
    """
    par = {"G": G, "weights": weights, "t_end": t_end, "clip_mode": clip_mode}
    sde = mpr_jax.MPR_sde.create(par)
    bold = sde.run({}, record_rv=False, fast_bold=fast_bold,
                   grad_horizon=grad_horizon)["bold_d"]
    if which_stat == "FC":
        F = FCD_jax.get_fc(bold[int(cut):].T, eps=eps, keep_negative=keep_negative)
        v = F[jnp.triu_indices(F.shape[0], k=1)]
    else:
        F = FCD_jax.extract_FCD_jax(bold[int(cut):].T, starts, nn, 30, 0.94, eps=eps,
                                    keep_negative=keep_negative)
        v = F[jnp.triu_indices(F.shape[0], k=30)]
    return _fisher_z(v) if fisher_z else v


class _JacOp(Op):
    """Returns d(stat)/dG as a vector (used inside ForwardGradOp.grad)."""
    def __init__(self, jac_fn):
        self.jac_fn = jac_fn
    def make_node(self, G):
        G = pt.as_tensor_variable(G)
        return Apply(self, [G], [pt.fvector()])
    def perform(self, node, inputs, outputs):
        outputs[0][0] = np.asarray(self.jac_fn(float(inputs[0])), dtype=np.float32).ravel()
        _maybe_clear_jax_cache()


# The JAX forward accumulates ~20-35 MB of cache per call (measured: jax.clear_caches()
# fully reclaims it, gc.collect() only partly, so it is a JAX cache leak not Python).
# Unchecked it OOMs long chains (~150 draws/chain at t_end=40000). Clearing every call
# would force a recompile each time; instead clear PERIODICALLY -> peak bounded to
# ~N*per-call, amortized recompile cost negligible on a long run. FORWARD_CLEAR_EVERY=0
# disables. Counter is process-local (spawned chain workers each get their own).
_FWD_CLEAR_EVERY = int(os.environ.get("FORWARD_CLEAR_EVERY", "40"))
_fwd_call_count = [0]

def _maybe_clear_jax_cache():
    if _FWD_CLEAR_EVERY <= 0:
        return
    _fwd_call_count[0] += 1
    if _fwd_call_count[0] % _FWD_CLEAR_EVERY == 0:
        jax.clear_caches()


class ForwardGradOp(Op):
    """Differentiable pytensor Op wrapping the JAX forward model (scalar G -> stat
    vector). Two gradient paths, both FD: native PyMC samplers use .grad (numpy FD
    via _JacOp); the JAX backend (nuts_sampler="blackjax"/"numpyro") uses jax_fn,
    a jax.custom_vjp forward whose backward is the same FD gradient (see jax_funcify
    registration below). The .grad here also satisfies PyMC's NUTS-eligibility check."""
    def __init__(self, forward_fn, jac_fn, jax_fn=None):
        self.forward_fn = forward_fn
        self._jacop = _JacOp(jac_fn)
        self.jax_fn = jax_fn              # JAX custom_vjp forward for the JAX backend
    def make_node(self, G):
        G = pt.as_tensor_variable(G)
        return Apply(self, [G], [pt.fvector()])
    def perform(self, node, inputs, outputs):
        # forward_fn is the make_forward_fns closure (or its standardize wrapper),
        # which already calls _maybe_clear_jax_cache -> no extra clear needed here.
        outputs[0][0] = np.asarray(self.forward_fn(float(inputs[0])), dtype=np.float32).ravel()
    def grad(self, inputs, output_grads):
        G = inputs[0]
        gvec = output_grads[0]
        jac = self._jacop(G)              # d(stat)/dG, shape (m,)
        return [pt.dot(gvec, jac)]         # scalar gradient wrt G


@jax_funcify.register(ForwardGradOp)
def _forwardgradop_jax_funcify(op, **kwargs):
    """Convert ForwardGradOp to its JAX forward for pm.sample(nuts_sampler=...).
    Returns the custom_vjp forward, so jax.grad (used by BlackJAX/NumPyro NUTS)
    sees the finite-difference gradient."""
    fn = op.jax_fn
    if fn is None:
        raise NotImplementedError("ForwardGradOp has no jax_fn; build it via make_forward_fns.")
    def forward(G):
        return fn(G)
    return forward


def make_forward_fns(weights, t_end, cut, which_stat, starts, nn,
                     clip_mode="hard", fast_bold=False, eps=0.0, grad_horizon=0,
                     grad_method="fd", fd_h=1e-2, fisher_z=False, keep_negative=False):
    """Build (forward, jac) closures for the JAX forward model with ALL config
    captured BY VALUE. Because they close over plain values (not module globals),
    cloudpickle serializes them correctly to spawned multiprocessing workers, so
    every chain computes the same statistic with the same knobs. grad_method in
    {"fd","autodiff"}; forward(g) and jac(g) both take a scalar float G."""
    w = np.asarray(weights, dtype=np.float32)
    s = None if starts is None else np.asarray(starts)

    def forward(g):
        out = np.asarray(
            _forward_stat_jax(jnp.float32(g), w, t_end, cut, which_stat, s, nn,
                              clip_mode, fast_bold, eps, grad_horizon, fisher_z, keep_negative),
            dtype=np.float32,
        ).ravel()
        # Bound the ~20-50MB/call JAX-cache leak on EVERY forward path (the Op, but
        # also pm.Simulator/smcabc, smclik and rmse call this closure directly and
        # would otherwise bypass the clear -> OOM). See _maybe_clear_jax_cache.
        _maybe_clear_jax_cache()
        return out

    if grad_method == "fd":
        def jac(g):
            fp = _forward_stat_jax(jnp.float32(g + fd_h), w, t_end, cut, which_stat, s, nn,
                                   clip_mode, fast_bold, eps, grad_horizon, fisher_z, keep_negative)
            fm = _forward_stat_jax(jnp.float32(g - fd_h), w, t_end, cut, which_stat, s, nn,
                                   clip_mode, fast_bold, eps, grad_horizon, fisher_z, keep_negative)
            return np.asarray((fp - fm) / (2.0 * fd_h), dtype=np.float32).ravel()
    elif grad_method == "autodiff":
        _jfn = jax.jacobian(lambda g: _forward_stat_jax(
            g, w, t_end, cut, which_stat, s, nn, clip_mode, fast_bold, eps, grad_horizon, fisher_z, keep_negative))
        def jac(g):
            return np.asarray(_jfn(jnp.float32(g)), dtype=np.float32).ravel()
    else:
        raise ValueError(f"grad_method must be 'fd' or 'autodiff', got {grad_method}")

    # JAX forward for the pytensor JAX backend (nuts_sampler="blackjax"/"numpyro").
    # custom_vjp injects the SAME central-FD gradient (the two +/-h sims run in one
    # vmapped call) so on-device NUTS uses the reliable FD gradient too.
    def _raw_jax(G):
        return _forward_stat_jax(G, w, t_end, cut, which_stat, s, nn,
                                 clip_mode, fast_bold, eps, grad_horizon, fisher_z, keep_negative)
    if grad_method == "autodiff":
        jax_forward = _raw_jax
    else:
        @jax.custom_vjp
        def jax_forward(G):
            return _raw_jax(G)
        def _jf_fwd(G):
            return _raw_jax(G), G
        def _jf_bwd(G, ct):
            both = jax.vmap(_raw_jax)(jnp.stack([G + fd_h, G - fd_h]))
            jacv = (both[0] - both[1]) / (2.0 * fd_h)
            return (jnp.vdot(ct, jacv),)
        jax_forward.defvjp(_jf_fwd, _jf_bwd)

    return forward, jac, jax_forward


def feature_reference_scatter(G, weights, t_end, cut, which_stat, starts, nn,
                              clip_mode="hard", fast_bold=False, eps=0.0,
                              n_ref=8, base_seed=10_000, fisher_z=False,
                              keep_negative=False):
    """Per-feature mean/std of the FC/FCD statistic across `n_ref` noise seeds.

    Used by --standardize_features: dividing each feature by its noise-scatter puts
    all features on unit scale (so obs_err=1 is meaningful) and places FC and FCD on
    the same scale. Runs n_ref extra full sims once at setup. Extraction mirrors
    _forward_stat_jax exactly so standardized features stay congruent with the model."""
    reps = []
    for i in range(n_ref):
        par = {"G": G, "weights": weights, "t_end": t_end,
               "clip_mode": clip_mode, "seed": base_seed + i}
        bold = mpr_jax.MPR_sde.create(par).run(
            {}, record_rv=False, fast_bold=fast_bold, grad_horizon=0)["bold_d"]
        b = bold[int(cut):].T
        if which_stat == "FC":
            F = FCD_jax.get_fc(b, eps=eps, keep_negative=keep_negative)
            v = F[jnp.triu_indices(F.shape[0], k=1)]
        else:
            F = FCD_jax.extract_FCD_jax(b, starts, nn, 30, 0.94, eps=eps,
                                        keep_negative=keep_negative)
            v = F[jnp.triu_indices(F.shape[0], k=30)]
        if fisher_z:
            v = _fisher_z(v)          # scatter estimated on the transformed feature
        reps.append(np.asarray(v, dtype=np.float32))
    reps = np.stack(reps, 0)
    mu = reps.mean(0).astype(np.float32)
    sd = np.maximum(reps.std(0, ddof=1), 1e-6).astype(np.float32)  # floor: never /~0
    return mu, sd


def make_forward_grad_op(weights, t_end, cut, which_stat, starts, nn,
                         clip_mode="hard", fast_bold=False, eps=0.0, grad_horizon=0,
                         grad_method="fd", fd_h=1e-2):
    """Build a differentiable ForwardGradOp with config baked into its closures."""
    forward, jac, jax_forward = make_forward_fns(weights, t_end, cut, which_stat, starts, nn,
                                                 clip_mode, fast_bold, eps, grad_horizon,
                                                 grad_method, fd_h)
    return ForwardGradOp(forward, jac, jax_forward)


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

def rmse_fit_mean(forward_fn, trace, theta_true, my_var_names, FC_obs_flat):

    theta_mean=np.mean(az.extract(trace).to_dataframe()[my_var_names], axis=0).to_numpy()
    theta_err=np.sqrt(np.mean(theta_mean-theta_true)**2)
    xs_= forward_fn(float(np.asarray(theta_mean).ravel()[0]))
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

def rmse_fit_map(forward_fn, trace, theta_true, FC_obs_flat):

    chains_pooled = trace.posterior["G"].values.reshape(1, -1)
    theta_map=calcula_map(chains_pooled)
    theta_map =np.asarray(theta_map)
    theta_err=np.sqrt(np.mean(theta_map-theta_true)**2)
    xs_= forward_fn(float(theta_map.ravel()[0]))
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


# ------------------ Benchmark metrics ------------------
# Shared with mpr_jax_numpyro.py so every sampler is scored identically.
from benchmark_utils import benchmark_metrics


# ------------------ Main ------------------
def main():
    args = parse_args()
    # NB: 'spawn' is set at import time (top of module) so each chain worker is a
    # fresh process — on CPU this avoids fork+JAX segfaults and lets cloudpickle
    # ship the forward closures (with baked-in config) to workers by value.

    # --- unpack args ---
    G, t_end, tr, cut = args.G, args.t_end, args.tr, args.cut

    wwidth, maxWindows, olap = args.wwidth, args.maxWindows, args.olap

    obs_err, n_prior, mu_G, sigma_G, lower_G, n_warmup, n_samples, n_chains, epsilon, threshold, correlation_threshold = (
        args.obs_err, args.n_prior, args.mu_G, args.sigma_G, args.lower_G, args.n_warmup, args.n_samples, args.n_chains, 
        args.epsilon, args.threshold, args.correlation_threshold
    )
    seed = args.seed
    SC_type = args.SC_type.lower()
    SC_size = args.SC_size
    sampler = args.sampler
    which_stat = args.which_stat.upper()
    resultspath = utils.results_folder()

    # --- device-aware parallelism ---
    # On CPU: one chain per core via spawned processes (fast, isolated JAX each).
    # On GPU: a single process — N workers would each grab a CUDA context / device
    # memory and contend or OOM. So force cores=1 and rely on in-process sampling;
    # true GPU speedup would come from on-device batching (vmap), not processes.
    backend = jax.default_backend()  # "cpu" or "gpu"
    if args.sample_cores and args.sample_cores > 0:
        sample_cores = args.sample_cores   # explicit override (e.g. 1 to cap peak RAM)
    else:
        sample_cores = 1 if backend == "gpu" else n_chains
    print(f"JAX backend: {backend} | chains={n_chains}, sampling cores={sample_cores}")

    # --- setup params ---
    if SC_type == "sim":
        SC = nx.to_numpy_array(nx.complete_graph(SC_size)).astype(np.float32)
    elif SC_type == "data":
        datapath = utils.DATA_ROOT
        SC = np.loadtxt(os.path.join(datapath, "weights.txt")).astype(np.float32)
        SC = SC[:SC_size, :SC_size]
    else:
        raise ValueError(f"Invalid SC_type '{SC_type}'. Must be 'sim' or 'data'.")

    # Defined the same way for both SC types so every code path (observed data,
    # gradient-free op, smcabc, nuts) uses the same normalized coupling matrix.
    nn = len(SC)
    weights = jnp.array(SC) / jnp.max(SC)
    
    params = {
        "G": G, "weights": weights, "t_end": t_end, "seed": seed
    }

    print("Running inference with parameters:")
    print(f"G = {G}, t_end = {t_end}, tr = {tr}, cut = {cut}, wwidth = {wwidth}, "
          f"maxWindows = {maxWindows}, olap = {olap}")
    print(f"obs_err = {obs_err}, n_prior = {n_prior}, n_warmup = {n_warmup}, "
          f"n_samples = {n_samples}, n_chains = {n_chains}")

    # --- parameter tag for filenames ---  
    tag = f"G{G}_cut{cut}_tr{tr}_seed{seed}_tend{t_end}_ns{n_samples}_nc{n_chains}_SC_{SC_size}_sampler_{sampler}_which_stat_{which_stat}_epsilon_{epsilon}"
    print(f"\nSaving all outputs with tag: {tag}\n")

    # --- file paths ---  
    fname_prior_pred = os.path.join(resultspath, f"prior_predictive_G_{tag}.png")
    fname_trace = os.path.join(resultspath, f"trace_G_{tag}.png")
    fname_summary_csv = os.path.join(resultspath, f"summary_{tag}.csv")
    fname_summary_RMSE_csv = os.path.join(resultspath, f"summary_RMSE_{tag}.csv")
    fname_netcdf = os.path.join(resultspath, f"inference_results_{tag}.nc")
    fname_benchmark_csv = os.path.join(resultspath, f"benchmark_{tag}.csv")
    fname_benchmark_params_csv = os.path.join(resultspath, f"benchmark_params_{tag}.csv")
    fname_benchmark_json = os.path.join(resultspath, f"benchmark_{tag}.json")

    # --- build the single config-explicit forward model (used by EVERY path) ---
    if which_stat not in ("FC", "FCD"):
        raise ValueError(f"Invalid statistic '{which_stat}'. Must be 'FC' or 'FCD'.")

    nnodes = int(np.asarray(weights).shape[0])
    # Precompute FCD sliding-window starts once, from a reference-simulation BOLD
    # length (after cut), so observed data and every sampler use identical windows.
    if which_stat == "FCD":
        ref_bold = mpr_jax.MPR_sde.create(
            {"G": float(params["G"]), "weights": weights, "t_end": t_end,
             "clip_mode": args.clip_mode}
        ).run({}, record_rv=False, fast_bold=args.fast_bold,
              grad_horizon=args.grad_horizon)["bold_d"]
        T = int(ref_bold[int(cut):].shape[0])
        _, starts = FCD_jax.precompute_shift_and_starts(T, 30, 0.94, stride1=args.fcd_stride1)
        starts = np.asarray(starts)
    else:
        starts = None

    # forward_np(G)->flat statistic and jac_np(G)->d(stat)/dG. Config is captured
    # by value inside these closures, so a single fwd_op works for gradient-free
    # samplers, smcabc and nuts, and survives cloudpickle to spawned workers.
    forward_np, jac_np, jax_fwd = make_forward_fns(
        weights, t_end, cut, which_stat, starts, nnodes,
        clip_mode=args.clip_mode, fast_bold=args.fast_bold, eps=args.fc_eps,
        grad_horizon=args.grad_horizon, grad_method=args.grad_method, fd_h=args.fd_h,
        fisher_z=args.fisher_z, keep_negative=args.keep_negative_fc)

    # --- optional per-feature standardization -----------------------------------
    # Wrap the forward as an AFFINE transform  x -> (x - mu)/sd  applied to EVERY
    # consumer of the statistic (observed, likelihood, FD-grad, SMC), so a single
    # scalar obs_err is a unit-scale sigma and FC/FCD are comparable. Identity
    # (mu=0, sd=1) when --standardize_features is off -> bit-exact prior behaviour.
    feat_mu = np.float32(0.0)
    feat_sd = np.float32(1.0)
    if args.standardize_features:
        feat_mu, feat_sd = feature_reference_scatter(
            float(params["G"]), weights, t_end, cut, which_stat, starts, nnodes,
            clip_mode=args.clip_mode, fast_bold=args.fast_bold, eps=args.fc_eps,
            n_ref=args.n_ref_seeds, fisher_z=args.fisher_z,
            keep_negative=args.keep_negative_fc)
        print(f"Standardizing {which_stat} features over {args.n_ref_seeds} noise "
              f"seeds: dim={feat_sd.shape[0]}, median feature sd={np.median(feat_sd):.4g}")
        _mu_j, _sd_j = jnp.asarray(feat_mu), jnp.asarray(feat_sd)
        _f0, _j0, _x0 = forward_np, jac_np, jax_fwd
        forward_np = lambda g, _f=_f0: ((np.asarray(_f(g)) - feat_mu) / feat_sd).astype(np.float32)
        jac_np     = lambda g, _j=_j0: (np.asarray(_j(g)) / feat_sd).astype(np.float32)
        jax_fwd    = lambda G, _x=_x0: (_x(G) - _mu_j) / _sd_j   # affine ∘ custom_vjp: FD grad /sd

    fwd_op = ForwardGradOp(forward_np, jac_np, jax_fwd)

    # --- Generate observed data (same forward as the model -> shapes/semantics match) ---
    FC_obs_flat = np.asarray(forward_np(float(params["G"])), dtype=np.float32)
    if args.noisy_obs:
        # Make the synthetic observation a genuine draw from the likelihood: add
        # N(0, obs_err) (in standardized space obs_err=1 == unit per-feature noise).
        _rng_obs = np.random.default_rng(seed)
        FC_obs_flat = (FC_obs_flat
                       + _rng_obs.normal(0.0, obs_err, size=FC_obs_flat.shape)).astype(np.float32)
        print(f"Added observation noise N(0, {obs_err}) to observed "
              f"({FC_obs_flat.shape[0]} features).")
    prior_specs = {"mu_G" : mu_G, "sigma_G": sigma_G, "lower_G": lower_G}
    my_var_names = ['G']

    with pm.Model() as prior_model:
        G = pm.TruncatedNormal("G", mu=prior_specs["mu_G"], sigma=prior_specs["sigma_G"], lower=prior_specs["lower_G"])

    with prior_model:
        prior_predict= pm.sample_prior_predictive(n_prior)

    az.plot_posterior(prior_predict, var_names=["G"], group="prior", kind="hist", hdi_prob=0.94, bins=50)
    plt.tight_layout()
    plt.savefig(fname_prior_pred, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Prior predictive plot saved to: {fname_prior_pred}")

    tails_5th_percentile=tails_percentile(my_var_names, prior_predict.prior, 0.05)

    # prior SD per parameter (for the shrinkage metric = 1 - post_var/prior_var)
    prior_sd = {v: float(np.std(np.asarray(prior_predict.prior[v]).ravel(), ddof=1))
                for v in my_var_names}

    os.makedirs(resultspath, exist_ok=True)

    with pm.Model() as model:

        G = pm.TruncatedNormal("G", mu=prior_specs["mu_G"], sigma=prior_specs["sigma_G"], lower=prior_specs["lower_G"])

        # Single shared forward Op (config baked in). Gradient-free samplers just
        # never call its .grad; nuts uses the same op below.
        FC_hat_flat = fwd_op(G)

        # Likelihood
        pm.Normal("FC_obs", mu=FC_hat_flat, sigma=obs_err, observed=FC_obs_flat)

    with model:
        # Debug the model: check logp at initial point
        print("Debugging model for NaNs or -inf in logp...")
        model.debug()

    

    #vars_list = list(model.values_to_rvs.keys())[:-1]
    vars_list = [model["G"]]

    # --- inference ---
    theta_true = np.array([params["G"]])

    rng_key = jax.random.PRNGKey(seed)

    rng_key, rng_key_run = jax.random.split(rng_key)

    
    # ---- SAFE forward-model test (NumPy only, correct evaluation) ----

    #test_val = FC_hat_flat.eval({
    #    G: np.float32(0.33),
    #    # IMPORTANT: must match the symbolic tensor you used in the model:
    #    pt.fmatrix(np.array(SC, dtype=np.float32)): np.array(SC, dtype=np.float32),
    #    pt.fscalar(np.float32(t_end)): np.float32(t_end),
    #    pt.fscalar(np.float32(cut)): np.float32(cut),
    #})

    #print("Test forward model shape:", test_val.shape)
    #print("Any NaNs?", np.isnan(test_val).any())


    if sampler == "slice":

        sampler = "Slice Sampler"
        start_time = time.time()
        with model:
            trace_slice = pm.sample(step=[pm.Slice(vars_list)], tune=n_warmup, draws=n_samples, chains=n_chains, cores=sample_cores, 
                                    initvals={var_name: tails_5th_percentile[var_name] for var_name in my_var_names}
                                ) 
        crudetime_slice=time.time() - start_time
        print("---running took: %s seconds ---" % crudetime_slice)
        trace = trace_slice

    elif sampler == "metropolis":

        sampler = "Metropolis"
        start_time = time.time()
        with model:
            trace_M = pm.sample(step=[pm.Metropolis()], tune=n_warmup, draws=n_samples, chains=n_chains, cores=sample_cores,
                            initvals={var_name: tails_5th_percentile[var_name] for var_name in my_var_names}
                            )
        crudetime_M=time.time() - start_time
        print("---running took: %s seconds ---" % crudetime_M)
        trace = trace_M

    elif sampler == "demetropolisz":

        sampler = "DE MetropolisZ"
        start_time = time.time()
        with model:
            trace_DEMZ = pm.sample(step=[pm.DEMetropolisZ()], tune=n_warmup, draws=n_samples, chains=n_chains, cores=sample_cores,
                                    initvals={var_name: tails_5th_percentile[var_name] for var_name in my_var_names}
                                ) 
        crudetime_DEMZ=time.time() - start_time
        print("---running took: %s seconds ---" % crudetime_DEMZ)
        trace = trace_DEMZ

    elif sampler == "demetropolis":

        sampler = "DEMetropolis"
        start_time = time.time()
        # WARNING: DEMetropolis is unreliable with this JAX/XLA forward model.
        # It is a POPULATION sampler (coupled chains): with cores>1 PyMC's own
        # PopulationStepper spawns subprocesses that crash on the JAX op, and even
        # with cores=1 (set below to avoid that path) the full pipeline hits an
        # XLA LLVM "Cannot allocate memory" during compilation and segfaults
        # (reproducible; works only in a minimal script). Prefer DEMetropolisZ for
        # differential-evolution sampling: its chains are independent and DO run
        # one-per-core under spawn. cores=1 is kept as the least-broken option.
        with model:
            trace_DEM = pm.sample(step=[pm.DEMetropolis()],  draws=n_samples, chains=n_chains, cores=1,
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
                                        cores=sample_cores
                                        )
        crudetime_SMC_like=time.time() - start_time
        print("---running took: %s seconds ---" % crudetime_SMC_like)
        trace = trace_SMC_like

    elif sampler == "smcabc":

        def simulator_forward_model(rng,  G, size=None):
            mu = forward_np(float(np.asarray(G).ravel()[0])).reshape(-1, 1)
            return rng.normal(mu, obs_err)
        
        with pm.Model() as model:
            # Priors
            G = pm.TruncatedNormal("G", mu=prior_specs["mu_G"], sigma=prior_specs["sigma_G"], lower=prior_specs["lower_G"])
            
            params_samples=[G]
            
            pm.Simulator(
                "FC_obs",
                simulator_forward_model,
                params=params_samples,
                epsilon=epsilon,
                observed=FC_obs_flat,)
            
        sampler = f"SMC, $\epsilon$ = {epsilon}"

        start_time = time.time()

        with model:
            # Run the SMC sampler
            trace_SMC_e = pm.sample_smc(draws=n_samples,         
                                        kernel=pm.smc.IMH,  
                                        threshold=threshold,  #default 0.5
                                        correlation_threshold=correlation_threshold,  #defaul 0.01
                                        progressbar=False, 
                                        chains=n_chains,
                                        cores=sample_cores)
        crudetime_SMC_e1=time.time() - start_time

        print("---running took: %s seconds ---" % crudetime_SMC_e1)
        trace = trace_SMC_e

    elif sampler == "nuts":

        # Gradient-based NUTS using the same shared differentiable fwd_op built
        # above. grad_method="fd" (default) uses central finite differences
        # (correct across the bifurcation); "autodiff" uses jax.jacobian (exact
        # but unreliable past the transition). The observed FC_obs_flat came from
        # the same forward_np, so shapes/semantics already match.
        with pm.Model() as model_nuts:
            G = pm.TruncatedNormal("G", mu=prior_specs["mu_G"],
                                   sigma=prior_specs["sigma_G"],
                                   lower=prior_specs["lower_G"])
            mu = fwd_op(G)
            pm.Normal("FC_obs", mu=mu, sigma=obs_err, observed=FC_obs_flat)

        sampler = f"NUTS ({args.grad_method})"
        start_time = time.time()
        with model_nuts:
            trace_nuts = pm.sample(
                step=[pm.NUTS([model_nuts["G"]])],
                tune=n_warmup, draws=n_samples, chains=n_chains, cores=sample_cores,
                initvals={var_name: tails_5th_percentile[var_name] for var_name in my_var_names},
            )
        crudetime_nuts = time.time() - start_time
        print("---running took: %s seconds ---" % crudetime_nuts)
        trace = trace_nuts

    elif sampler in ("numpyro", "blackjax"):

        # On-device NUTS via PyMC's JAX backend (nuts_sampler = "numpyro" or
        # "blackjax"). The default `model` (with fwd_op) compiles to JAX through
        # the registered jax_funcify, and the custom_vjp gives NUTS the FD
        # gradient. Chains are vmapped on-device ("vectorized"), so this is the
        # GPU-ready path within the PyMC benchmark. NumPyro is the more mature/
        # convenient backend, so it is the recommended default of the two.
        nuts_sampler = sampler
        sampler = f"{'NumPyro' if nuts_sampler == 'numpyro' else 'BlackJAX'} NUTS ({args.grad_method})"
        start_time = time.time()
        with model:
            trace_bj = pm.sample(
                nuts_sampler=nuts_sampler,
                tune=n_warmup, draws=n_samples, chains=n_chains,
                nuts_sampler_kwargs={"chain_method": "vectorized"},
                progressbar=False, random_seed=seed,
            )
        print("---running took: %s seconds ---" % (time.time() - start_time))
        trace = trace_bj

    else:
        raise ValueError(f"Invalid sampler: {sampler}")

    # Wall-clock sampling time of whichever branch ran: each branch sets
    # start_time right before its pm.sample/pm.sample_smc call, so this measures
    # the sampling itself (excluding model build) consistently across samplers.
    runtime = time.time() - start_time

    #print(trace.sample_stats.beta)

    # Save the FULL InferenceData (posterior + sample_stats etc.) for benchmarking.
    # SMC samplers (smclik/smcabc) store a 'beta' tempering schedule in sample_stats
    # as an object array with mixed int/float, which az.to_netcdf cannot serialize.
    # Coerce any object-dtype sample_stats var to float, and make the save non-fatal
    # so a serialization hiccup never discards a completed (expensive) run.
    if hasattr(trace, "sample_stats"):
        for _v in list(trace.sample_stats.data_vars):
            if trace.sample_stats[_v].dtype == object:
                try:
                    trace.sample_stats[_v] = trace.sample_stats[_v].astype("float64")
                except (ValueError, TypeError):
                    # Ragged/nested object stat (e.g. SMC 'beta' tempering schedule
                    # stored as an array of lists) is not netCDF-serializable and
                    # can't be coerced element-wise. Drop it so the rest still saves.
                    trace.sample_stats = trace.sample_stats.drop_vars(_v)
    try:
        az.to_netcdf(trace, fname_netcdf)
        print(f"Trace saved to: {fname_netcdf}")
    except Exception as _e:
        print(f"WARNING: full az.to_netcdf failed ({type(_e).__name__}: {_e}); "
              f"saving posterior group only.")
        az.to_netcdf(trace.posterior, fname_netcdf)

    summary_df = az.summary(trace)
    summary_df.to_csv(fname_summary_csv)
    print(summary_df)
    print(f"Summary saved to: {fname_summary_csv}")

    plot_trace(trace, my_var_names, theta_true, sampler, fname=fname_trace)

    rmse_paramsmean, rmse_fitmean  = rmse_fit_mean(forward_np, trace, theta_true, my_var_names, FC_obs_flat)
    print ('RMSE to true parameters', rmse_paramsmean),
    print ('RMSE to true observation', rmse_fitmean)

    rmse_paramsmap, rmse_fitmap  = rmse_fit_map(forward_np, trace, theta_true, FC_obs_flat)
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

    # --- comprehensive benchmark metrics (one row per run; concat across runs
    #     to compare samplers × statistic on convergence/efficiency/accuracy) ---
    meta = {
        "sampler": sampler, "which_stat": which_stat, "SC_type": SC_type,
        "SC_size": SC_size, "G_true": float(params["G"]), "t_end": t_end,
        "cut": cut, "obs_err": obs_err, "seed": seed,
        "n_warmup": n_warmup, "grad_method": args.grad_method,
        "clip_mode": args.clip_mode, "epsilon": epsilon,
        "fcd_stride1": bool(args.fcd_stride1),
        "standardize_features": bool(args.standardize_features),
        "noisy_obs": bool(args.noisy_obs),
        "fisher_z": bool(args.fisher_z),
        "keep_negative_fc": bool(args.keep_negative_fc),
    }
    rmse = {"param_mean": float(rmse_paramsmean), "param_map": float(rmse_paramsmap),
            "fit_mean": float(rmse_fitmean), "fit_map": float(rmse_fitmap)}
    try:
        run_row, param_rows = benchmark_metrics(
            trace, theta_true, my_var_names, prior_sd, runtime, rmse, meta)
        pd.DataFrame([run_row]).to_csv(fname_benchmark_csv, index=False)
        pd.DataFrame(param_rows).to_csv(fname_benchmark_params_csv, index=False)
        with open(fname_benchmark_json, "w") as fh:
            json.dump({"run": run_row, "params": param_rows}, fh, indent=2)
        print(f"Benchmark metrics saved to: {fname_benchmark_csv}")
        print(f"  runtime={runtime:.1f}s  max_r_hat={run_row['max_r_hat']:.3f}  "
              f"min_ess_bulk={run_row['min_ess_bulk']:.1f}  "
              f"ess/sec={run_row['min_ess_bulk_per_sec']:.2f}  "
              f"rmse_param_mean={run_row['rmse_param_mean']:.4f}")
    except Exception as e:
        print(f"WARNING: benchmark_metrics failed: {e}")

if __name__ == "__main__":

    main()
