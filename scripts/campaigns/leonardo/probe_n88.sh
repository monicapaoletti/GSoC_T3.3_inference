#!/usr/bin/env bash
# Feasibility + cost probe for the 88-node campaign on Leonardo Booster (A100 64GB).
#
# WHY THIS RUNS BEFORE bench_n88_submit.sh / nuts_n88_submit.sh. The UGent hour budgets
# (rwmh 11.2h, demc 11.5h, smc_abc/smc_abc_demc 1.4h, smc_lik 0.4h, per G per statistic --
# see scripts/campaigns/ugent/bench_n88_submit.sh) were MEASURED on an H100 NVL. An A100
# is a different card (less memory, different throughput curve) and copying those hour
# budgets unchanged would be a guess, not a measurement -- exactly the mistake the UGent
# README warns against repeating. Run this probe, read its batch-sweep + per-replicate
# numbers, and scale the --time= values in bench_n88_submit.sh / nuts_n88_submit.sh from
# THIS cluster's own cost model (runtime = n_steps x t_call(batch)) before submitting the
# real matrix.
#
# Task 0: batch sweep (forward throughput vs batch size) -- same sweep UGent ran, to find
#   where the A100 saturates (64 GB is a third of the H100 NVL's; the occupancy point
#   that forced CUDA MPS on doslis's 23 GB L4s may or may not reappear here).
# Task 1-4: one real SMC/MCMC replicate each (FC/FCD x a cheap sampler + an expensive one),
#   at the campaign's actual settings, to anchor the cost model with a real number instead
#   of an extrapolation from the sweep alone.
#
#SBATCH --job-name=probe88_leo
# account: pass it on the command line -- sbatch --account=EUHPC_B40_021 <this script>
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --output=%x-%A_%a.out
set -uo pipefail

module load openblas
eval "$(micromamba shell hook --shell=bash)"
micromamba activate jax_conda

LEO_HOME="/leonardo/home/userexternal/mpaolett"
GSOC_REPO="$LEO_HOME/GSoC_T3.3_inference"
OUT="$LEO_HOME/results/probe_n88_leonardo"
mkdir -p "$OUT"

cd "$GSOC_REPO/scripts/functions" || exit 1
echo "[probe] task $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"
nvidia-smi -L

case "$SLURM_ARRAY_TASK_ID" in
  0)
    # batch sweep: forward-only throughput, G-/stat-independent
    python run_gpu_pool.py --suite forward --SC_size 88 --gpu_only --gpu_concurrency 1 \
      --results_dir "$OUT"
    ;;
  1)
    time python mpr_jax_numpyro.py --sampler smc_lik --which_stat FCD --G 0.5 \
      --n_stages 50 --n_mcmc 5 --SC_size 88 --save_dir "$OUT"
    ;;
  2)
    time python mpr_jax_numpyro.py --sampler rwmh --which_stat FCD --G 0.5 \
      --n_chains 1024 --n_warmup 1000 --n_samples 1000 --SC_size 88 --save_dir "$OUT"
    ;;
  3)
    time python mpr_jax_numpyro.py --sampler smc_lik --which_stat FC --G 0.5 \
      --n_stages 50 --n_mcmc 5 --SC_size 88 --save_dir "$OUT"
    ;;
  4)
    # gradient-based arm: one NUTS replicate, fd gradient (default, safe -- see
    # memory/grad-horizon-zeroes-gradient and gradient-based-inference-mpr; autodiff
    # explodes/vanishes in the chaotic regime, do NOT default to it), FCD needs
    # fc_eps>0 or the corrcoef gradient through a zero-variance window is NaN.
    time python mpr_jax_numpyro.py --sampler nuts --chain_method vectorized \
      --n_chains 32 --n_warmup 500 --n_samples 500 --which_stat FCD --fc_eps 1e-6 \
      --G 0.5 --SC_size 88 --save_dir "$OUT"
    ;;
esac
echo "[probe] task $SLURM_ARRAY_TASK_ID done at $(date)"
