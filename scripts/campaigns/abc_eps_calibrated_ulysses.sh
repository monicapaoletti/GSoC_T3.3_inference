#!/bin/bash
# Calibrated-epsilon ABC-SMC on the pymc/CPU path: changes ONE variable.
#
# Every existing CPU ABC cell ran at epsilon=10 because benchmark_sweep_ulysses.sh never
# passed --epsilon and mpr_jax_pymc.py defaults to 10. pm.Simulator's default distance is
# gaussian, -0.5*((obs-sim)/epsilon)^2 -- the same form as a Gaussian log-likelihood of
# sd epsilon -- and the simulator draws rng.normal(mu, obs_err) with obs_err=1.0. So
# epsilon=10 is a tolerance ten times the noise it is tolerating, and epsilon=obs_err=1.0
# is the calibrated choice, exactly as in the JAX path.
#
# Every other flag is copied verbatim from the sweep's pymc branch, so the ONLY difference
# from the existing epsilon=10 cells is epsilon. That isolates the calibration effect,
# which the concurrent JAX-on-CPU run (batch 64) cannot do since it also changes framework
# and batch width.
#
#SBATCH --job-name=abc_eps1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --partition=long1
#SBATCH --array=0-1
set -uo pipefail

REPO=$HOME/GSoC_T3.3_inference
cd "$REPO/scripts/functions" || exit 1
DATE=$(date +%F)
RESULTS_DIR=$REPO/results/$DATE
mkdir -p "$RESULTS_DIR"

STATS=(FC FCD)
STAT=${STATS[$SLURM_ARRAY_TASK_ID]}

echo "[abc_eps1] task=$SLURM_ARRAY_TASK_ID stat=$STAT start $(date)"
srun --unbuffered python3 mpr_jax_pymc.py \
    --sampler smcabc --which_stat "$STAT" \
    --epsilon 1.0 \
    --G 0.2 --SC_type data --SC_size 10 --fast_bold --cut 10 \
    --t_end 30000 --n_warmup 1000 --n_samples 1000 --n_chains 2 \
    --sample_cores 2 --n_prior 1000 --seed 42
echo "[abc_eps1] task=$SLURM_ARRAY_TASK_ID stat=$STAT done $(date) rc=$?"
