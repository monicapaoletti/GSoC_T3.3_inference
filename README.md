# GSoC_T3.3_inference

## Project Overview

This repository implements simulations and inference of the Montbrió et al. neural mass model to study whole-brain dynamics. The focus is on estimating the global coupling parameter \(G\) by comparing simulated functional connectivity (FC) and functional connectivity dynamics (FCD) to empirical data. Both Numba and JAX implementations of the model are provided, along with benchmarking tools and inference scripts.

---

## Repository Structure
GSoC_T3.3_inference/
├── data/ # Input data (e.g., structural connectivity matrices)
├── results/ # Figures and outputs from simulations and analyses
├── scripts/
│ ├── notebook/ # Example notebooks
│ │ └── mpr_jax.ipynb # Jupyter notebook with JAX simulations
│ └── functions/ # Core simulation and inference scripts
│ ├── mpr.py # Montbrió model implemented in Numba
│ ├── mpr_jax.py # Montbrió model implemented in Python/JAX
│ ├── benchmarking_simulation.py # Compare runtime of Numba vs JAX
│ ├── grid_search_fc.py # Infer G by minimizing FC differences
│ ├── grid_search_fcd.py # Infer G by minimizing FCD differences
│ └── mpr_fc_fcd.py # Run simulations for fixed G and t_end, store and plot results
└── README.md # Project documentation
