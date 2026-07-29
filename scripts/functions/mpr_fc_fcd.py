"""
simulate the mpr model, calculate FC and FCD, store the output and, optionally, plot it (e.g. to reproduce fig S1 of the paper)
"""
import argparse
import site
import sys
import os
import logging
import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from copy import deepcopy
from importlib import reload
from jax import profiler
import matplotlib.ticker as ticker

site_user_base = site.getusersitepackages()
if site_user_base not in sys.path:
    sys.path.append(site_user_base)

import mpr_jax
import utils
import FCD_jax 

#from jax import config
#config.update("jax_log_compiles", True)



logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')


parser = argparse.ArgumentParser(description="Run simulation and save output")
parser.add_argument('--g_val', type=float, required=True, help='Coupling constant for simulation')
parser.add_argument('--t_end', type=float, required=True, help='Time for simulation in ms')
parser.add_argument('--do_simulation', type=lambda x: (str(x).lower() == 'true'), default=True, help='Run the simulation or skip it (default: True)')
parser.add_argument('--do_fc_bold', type=lambda x: (str(x).lower() == 'true'), default=True, help='Calculate and save FC for bold (default: True)')
parser.add_argument('--do_fcd_bold', type=lambda x: (str(x).lower() == 'true'), default=True, help='Calculate and save FCD for bold (default: True)')
parser.add_argument('--do_fc_r', type=lambda x: (str(x).lower() == 'true'), default=False, help='Calculate and save FC for r (default: False)')
parser.add_argument('--do_plot', type=lambda x: (str(x).lower() == 'true'), default=False, help='If True, run plotting only and skip simulation and FC/FCD calculations (default: False)')
parser.add_argument('--cut', type=int, default=50, help='Drop the first `cut` BOLD frames (initial transient + Balloon-Windkessel startup) before FC/FCD (default: 50)')


mpr_jax = reload(mpr_jax)

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.2f} seconds")
        return result
    return wrapper

def create_sde(par):
    return mpr_jax.MPR_sde.create(par)

#@jax.jit
#@timer
def simulate(sde, g_val, nn, seed):
    # Generate a JAX PRNG key from the scalar seed
    key = jax.random.PRNGKey(seed)
    
    # run with functional PRNG key
    data = sde.run({"G": g_val, "seed": seed})  # modify run to accept seed
    rv_t = data["rv_t"]
    rv_d = data["rv_d"]
    r = rv_d[:, :nn]
    v = rv_d[:, nn:]
    bold_d = data["bold_d"]
    bold_t = data["bold_t"]
    return (rv_t, r, v, bold_t, bold_d)

@timer
def batched_simulate(sde, g_vals, nn, seed):
    # Generate N different seeds for N runs
    seeds = jnp.arange(seed, seed + len(g_vals))
    
    def single_sim(inputs):
        g, s = inputs
        return simulate(sde, g, nn, s)
    
    # vmap over (g_vals, seeds)
    results = jax.vmap(single_sim)((g_vals, seeds))
    return results




def plot_fc_fcd(rv_t, r, bold_t, bold_d, FC, FCD, g_val, results_path, sim_filename):
    step = 10

    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1], figure=fig)

    # r plot (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(rv_t[::step], r[::step, :], lw=0.1)
    ax1.set_ylabel("r")
    ax1.set_title(f"Firing Rates, G = {g_val}")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1000:.1f}'))

    # BOLD plot (bottom-left)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(bold_t, bold_d, lw=0.1)
    ax2.set_ylabel("BOLD")
    ax2.set_xlabel("Time (s)")
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1000:.1f}'))

    # FC plot (top-right, square)
    ax3 = fig.add_subplot(gs[0, 1])
    im1 = ax3.imshow(FC, vmin=-0.5, vmax=1, cmap='hot', aspect='equal')
    ax3.set_title("FC")
    plt.colorbar(im1, ax=ax3, fraction=0.046, pad=0.04)

    # FCD plot (bottom-right, square)
    ax4 = fig.add_subplot(gs[1, 1])
    im2 = ax4.imshow(FCD, vmin=0, vmax=1, cmap='hot', aspect='equal')
    ax4.set_title("FCD")
    plt.colorbar(im2, ax=ax4, fraction=0.046, pad=0.04)

    plt.tight_layout()
    png_filename = sim_filename.replace('.npz', '.png')
    plt.savefig(os.path.join(f'{png_filename}'),dpi=300)
    plt.show()



if __name__ == "__main__":

    args = parser.parse_args()
    g_val = args.g_val
    t_end_val = args.t_end

    results_path = utils.results_folder()

    # --- Load weights ---
    weights = np.loadtxt(utils.DATA_ROOT + "/weights.txt")
    weights = jnp.array(weights)

    # --- Base parameters ---
    params = {
        "G": 0.33,  # will be overwritten
        "t_end": t_end_val,  # will be overwritten
        "weights": weights / jnp.max(weights),
        "dt": 0.01,
        "eta": jnp.array([-4.6]),
        "rv_decimate": 10,
        "noise_amp": 0.037,
        "tr": 300.0,
        "seed": 42
    }

    if args.do_plot:

        logging.info("do_plot is True, disabling simulation and FC/FCD calculations.")
        args.do_simulation = False
        args.do_fc_bold = False
        args.do_fcd_bold = False
        args.do_fc_r = False
        
        # Load simulation data
        sim_filename = os.path.join(results_path, f"sim_G_{float(g_val):.3f}_tend_{t_end_val}.npz")
        logging.info(f"Loading simulation data from {sim_filename}")
        try:
            sim_data = np.load(sim_filename)
            rv_t = sim_data['rv_t']
            r = sim_data['r']
            bold_t = sim_data['bold_t']
            bold_d = sim_data['bold_d']
            logging.info("Simulation data loaded successfully.")
        except FileNotFoundError:
            logging.error(f"Simulation file {sim_filename} not found.")
            sys.exit(1)

        # Load FC data (use bold FC as FC variable)
        fc_filename = os.path.join(results_path, f"FC_G_{float(g_val):.3f}_tend_{t_end_val}.npz")
        logging.info(f"Loading FC data from {fc_filename}")
        try:
            fc_data = np.load(fc_filename)
            FC = fc_data['FC']
            logging.info("FC data loaded successfully.")
        except FileNotFoundError:
            logging.error(f"FC file {fc_filename} not found.")
            sys.exit(1)

        # Load FCD data
        fcd_filename = os.path.join(results_path, f"FCD_G_{float(g_val):.3f}_tend_{t_end_val}.npz")
        logging.info(f"Loading FCD data from {fcd_filename}")
        try:
            fcd_data = np.load(fcd_filename)
            FCD = fcd_data['FCD']
            logging.info("FCD data loaded successfully.")
        except FileNotFoundError:
            logging.error(f"FCD file {fcd_filename} not found.")
            sys.exit(1)

        # Call the plotting function
        logging.info("Calling plot_fc_fcd function")
        plot_fc_fcd(rv_t, r, bold_t, bold_d, FC, FCD, g_val, results_path, sim_filename)
        

    if args.do_simulation:
        sde = create_sde(params)

        # to simulate for multiple than one G value is convenient to keep use the loop 
        g_vals = jnp.array([0.0,0.5,0.7])  
        #g_vals = jnp.array([g_val])
        #g_vals = jnp.arange(10)/11

        key = sde.key  # initial key
        seed = sde.P.seed
        nn = int(sde.P.nn)
        results = batched_simulate(sde, g_vals, nn, seed)
        rv_t_all, r_all, v_all, bold_t_all, bold_d_all = results
        logging.info(f't_end = {sde.key}')
        logging.info(f"Simulation done!")

        for i, g in enumerate(g_vals):
            sim_filename = os.path.join(
                results_path, f"sim_G_{float(g):.3f}_tend_{t_end_val}.npz"
            )
            np.savez(
                sim_filename,
                rv_t=np.array(rv_t_all[i]),
                r=np.array(r_all[i]),
                v=np.array(v_all[i]),
                bold_t=np.array(bold_t_all[i]),
                bold_d=np.array(bold_d_all[i]),
            )
            logging.info(f"Saved simulation for g={g:.3f} to {sim_filename}")

    else:
        sim_filename = os.path.join(results_path, f"sim_G_{float(g_val):.3f}_tend_{t_end_val}.npz")
        logging.info(f"Loading simulation data from {sim_filename}")

        try:
            data = np.load(sim_filename)
            rv_t = data['rv_t']
            r = data['r']
            v = data['v']
            bold_t = data['bold_t']
            bold_d = data['bold_d']
            logging.info("Simulation data loaded successfully.")
        except FileNotFoundError:
            logging.error(f"File {sim_filename} not found. Please run simulation first or check the path.")
            sys.exit(1)

    # Drop the initial transient ONCE, here, so every downstream feature (FC and FCD)
    # sees the same trimmed series -- mirrors vbi's model-level `t_cut` and prevents the
    # FC-cut-but-FCD-uncut inconsistency. r is left uncut (different time base: rv_decimate, not tr).
    bold = jnp.array(bold_d[args.cut:].T)
    r = jnp.array(r)

    if args.do_fc_r:
        FC_r = FCD_jax.get_fc(r)
        fc_r_filename = os.path.join(results_path, f"FC_r_G_{float(g_val):.3f}_tend_{t_end_val}.npz")
        np.savez(fc_r_filename, FC_r=FC_r)
        logging.info(f"Saved FC_r to {fc_r_filename}")

    if args.do_fc_bold:
        FC_bold = FCD_jax.get_fc(bold)
        fc_filename = os.path.join(results_path, f"FC_G_{float(g_val):.3f}_tend_{t_end_val}.npz")
        np.savez(fc_filename, FC=FC_bold)
        logging.info(f"Saved FC to {fc_filename}")

    if args.do_fcd_bold:
        wwidth = 30
        maxNwindows = 200
        olap = 0.94
        FCD, FCvec, _ = FCD_jax.extract_FCD(bold, wwidth=wwidth,
                                            maxNwindows=maxNwindows, olap=olap)
        fcd_filename = os.path.join(results_path, f"FCD_G_{float(g_val):.3f}_tend_{t_end_val}.npz")
        np.savez(fcd_filename, FCD=FCD, FCvec=FCvec)
        logging.info(f"Saved FCD to {fcd_filename}")

