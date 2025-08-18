import jax
import jax.numpy as jnp
import numpy as np
import mpr_jax
import FCD_jax
import utils
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, entropy
import logging
import argparse
import os

parser = argparse.ArgumentParser(description="Run simulation and save output")
parser.add_argument('--true_g', type=float, required=True, help='Coupling constant for simulation of the observed data',default=0.5)
parser.add_argument('--t_end', type=float, required=True, help='length of the simulation',default=300_000)
parser.add_argument('--grid', type=int, required=True, help='numbers of g_values', default=20)

def ks_distance_jax(x, y):
    x = jnp.sort(x)
    y = jnp.sort(y)
    n1 = x.shape[0]
    n2 = y.shape[0]
    data_all = jnp.sort(jnp.concatenate([x, y]))
    
    cdf1 = jnp.searchsorted(x, data_all, side='right') / n1
    cdf2 = jnp.searchsorted(y, data_all, side='right') / n2
    return jnp.max(jnp.abs(cdf1 - cdf2))


def grid_search_FCD_vmap(sde, obs_flat, G_values, wwidth, maxNwindows, olap):
    """
    Grid search over G (vmap) to compute KS and KL distances between simulated and observed FCD.
    
    Args:
        sde: an initialized SDE object (mpr_jax.MPR_sde)
        FCD_obs: observed FCD matrix (jax or numpy array)
        G_values: 1D jax array of candidate G values
    
    Returns:
        KS: array of KS distances for each G
        best_idx_KS: index of best G value identified by KS
        best_G_KS: G value with min KS
        KL: array of KL distances for each G
        best_idx_KL: index of best G values identified by KL
        best_G_KL: G value with min KL
    """
    

    def single_run(G):
        # Run single simulation
        data = sde.run({"G": G})
        bold_d = data["bold_d"]
        
        # Extract FCD
        FCD_sim, _, _ = FCD_jax.extract_FCD(
            bold_d.T,
            wwidth=wwidth,
            maxNwindows=maxNwindows,
            olap=olap
        )
        
        return jnp.ravel(FCD_sim)  # returns flatteed FCD


    # Vectorize the single run over G values (run for all G values)
    FCD_sims_flat = jax.vmap(single_run)(G_values)

    # Convert to NumPy for scipy functions
    FCD_sims_flat_np = np.array(FCD_sims_flat)
    obs_flat_np = np.array(obs_flat)

    # KS distance using scipy
    KS = np.array([ks_2samp(obs_flat_np, sim_flat)[0] for sim_flat in FCD_sims_flat_np])

    # KL divergence using scipy
    eps = 1e-8
    bins = 50
    p_hist, edges = np.histogram(obs_flat_np, bins=bins, density=True)
    p = (p_hist + eps) / (p_hist + eps).sum()

    KL = np.array([
        entropy(
            p,
            (np.histogram(sim_flat, bins=edges, density=True)[0] + eps) /
            (np.histogram(sim_flat, bins=edges, density=True)[0] + eps).sum()
        )
        for sim_flat in FCD_sims_flat_np
    ])


    
    best_idx_KS = KS.argmin()
    best_G_KS = G_values[best_idx_KS]
    best_idx_KL = KL.argmin()
    best_G_KL = G_values[best_idx_KL]

    return KS, best_idx_KS, best_G_KS, KL, best_idx_KL, best_G_KL


if __name__ == "__main__":

    wwidth = 30
    maxNwindows = 200
    olap = 0.94

    results_path = utils.results_folder()

    # Load weights (SC matrix)
    weights = np.loadtxt(utils.DATA_ROOT + "/weights.txt")
    weights = jnp.array(weights)

    # import arguments
    args = parser.parse_args()
    true_g = args.true_g
    t_end = args.t_end
    grid = args.grid

    # Create the observed data 
    params = {
        "G": true_g,  
        "t_end": t_end,  
        "weights": weights / jnp.max(weights),
        "dt": 0.01,
        "eta": jnp.array([-4.6]),
        "rv_decimate": 10,
        "noise_amp": 0.037,
        "tr": 300.0,
        "seed": 42
    }
    sde_obs = mpr_jax.MPR_sde.create(params)
    
    data = sde_obs.run({})
    bold_d = data["bold_d"]
    bold_d_T = bold_d.T
    
    # Extract FCD from observed data
    FCD_obs, FCvec, _ = FCD_jax.extract_FCD(
        bold_d_T,
        wwidth=wwidth,
        maxNwindows=maxNwindows,
        olap=olap
    )

    obs_flat = jnp.ravel(FCD_obs)

    # Define grid of G values
    G_values = jnp.linspace(0.0, 0.95, grid)

    # Run grid search
    sde_model = mpr_jax.MPR_sde.create(params)
    KS, best_idx_ks, G_KS, KL, best_idx_kl, G_KL = grid_search_FCD_vmap(sde_model, obs_flat, G_values, wwidth, maxNwindows, olap)

    FCD_KS = {
        "KS_distances": KS,
        "best_G": G_KS,         
        "best_idx": best_idx_ks,
        "g_values": G_values,
        "true_g": true_g  
    }

    FCD_KL = {
        "KL_distances": KL,
        "best_G": G_KL,
        "best_idx": best_idx_kl,
        "g_values": G_values,
        "true_g": true_g  
    }

    # Save them
    np.savez(
        os.path.join(results_path, f"FCD_KS_KL_g_{true_g}_t_end_{int(t_end)}_grid_{grid}.npz"),
        FCD_KS=FCD_KS,
        FCD_KL=FCD_KL
    )

    print("Results saved in FCD_results.npz")






    # Print results
    print("KS distances:", KS)
    print("Best G (KS):", G_KS)
    print("KL distances:", KL)
    print("Best G (KL):", G_KL)
    #plt.plot(KL)
    #plt.plot(KS)
    #plt.show()




