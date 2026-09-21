# GSoC_T3.3_inference

## Project Overview


This project is part of the **Google Summer of Code (GSoC)** program. The repository provides an implementation and inference of the **Montbrió et al. neural mass model** to study whole-brain dynamics. The focus is on estimating the **global coupling parameter (G)** by comparing simulated **functional connectivity (FC)** and **functional connectivity dynamics (FCD)** to empirical data. Both **Numba** and **JAX** implementations of the model are provided, along with benchmarking tools and inference scripts.

Contributor / Author: Monica Paoletti (mpaolett@sissa.it)  
Mentors: Meysam Hashemi and Daniele Marinazzo


## Repository Structure
- GSoC_T3.3_inference/
  - data/  # Input data (e.g., structural connectivity matrices)
  - results/  # Figures and outputs from simulations and analyses
  - scripts/
    - notebook/  # Example notebooks
      - mpr_jax.ipynb  # Jupyter notebook with JAX simulations
      - mpr_jax_inference.ipynb #notebook for optimization and Bayesian inference
      - mpr_jax_inference_smallSC.ipynb #notebook for optimization with parameter sweeping for both small and whole network
    - functions/  # Core simulation and inference scripts
      - mpr.py  # Montbrió model implemented in Numba
      - mpr_jax.py  # Montbrió model implemented in Python/JAX
      - benchmarking_simulation.py  # Compare runtime of Numba vs JAX
      - grid_search_fc.py  # Infer G by minimizing FC differences
      - grid_search_fcd.py  # Infer G by minimizing FCD differences
      - mpr_fc_fcd.py  # Run simulations for fixed G and t_end, store and plot results
      - mpr_jax_numpyro.py # Broken attempt of gradient-based inference
      - mpr_jax_pymc.py # Gradient-free inference
      - utils.py # Utilities
  - README.md  # Project documentation
  - GSoC_final_submission.md # Submission document
  - container/  # Pinned environment: laptop dev container and the cluster Apptainer image

## If you cloned this before 2026-09-20: the history was rewritten

The commits written by Claude between 2026-08-05 and 2026-09-10 were re-authored from
`Monica Paoletti` to `Claude <noreply@anthropic.com>`, so the record says who wrote what.
**No file content changed** — every tree is byte-identical, and every commit message and
date is the same — but the commit hashes from `Add SBC driver; FCD lag banding`
(2026-08-05) onward are new.

An existing clone will therefore not `git pull` cleanly. If you have no local commits:

```bash
git fetch origin
git reset --hard origin/main      # discards local changes -- stash them first if you have any
```

If you do have local commits to keep, replay them onto the new history instead:

```bash
git fetch origin
git rebase --onto origin/main <the-commit-you-branched-from> <your-branch>
```


