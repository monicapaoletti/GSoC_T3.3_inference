#!/usr/bin/env bash
# Cost + memory probe for the 88-node campaign on Leonardo Booster (A100 64 GB).
#
# THIS IS scripts/campaigns/ugent/probe_n88.sh, TASK FOR TASK, so every number it prints
# divides directly by the UGent (H100 NVL) one measured 2026-09-08:
#   task 0  forward sweep, s/call:  1:3.08 4:4.61 16:4.71 64:4.84 256:4.18 1024:4.59
#                                   4096:5.93 8192:7.72 16384:12.03; peak 30.9 GB
#   task 1-4  one SBC replicate (smc_lik/abc x FC/FCD): 1658-1681 s;
#             peak 5.1 GB (FC), 33.7-37.8 GB (FCD)
# The ratio at batch 4096 and the replicate ratio are the A100/H100 factor that sets every
# walltime and the GPU-hour budget. The peak memory says whether FCD at 4096 particles
# fits in 64 GB.
#
# NUTS is not here: its cost is unknown, and nuts_n88_submit.sh PILOT=1 is its probe.
#
# Submit from anywhere:  sbatch --account=EUHPC_B40_021 probe_n88.sh
# (the account cannot be set in an #SBATCH line from a variable; config.sh's default is
# the same project)
#SBATCH --job-name=probe88L
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --output=%x-%A_%a.out
set -uo pipefail

GSOC_REPO="${GSOC_REPO:-/leonardo/home/userexternal/mpaolett/GSoC_T3.3_inference}"
source "$GSOC_REPO/scripts/campaigns/leonardo/config.sh"
eval "$LEO_ENV_SETUP"
cd "$GSOC_REPO/scripts/functions" || exit 1

OUT="$LEO_RESULTS/probe_n88_leonardo"
mkdir -p "$OUT"

# JAX does not report peak device memory once preallocation is off, so sample the card.
MEMLOG="$OUT/gpumem_${SLURM_ARRAY_TASK_ID}.csv"
( while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$MEMLOG"
    sleep 5
  done ) &
SAMPLER_PID=$!
trap 'kill $SAMPLER_PID 2>/dev/null' EXIT

COMMON="--L 100 --scale 0.5 --n_particles 4096 --SC_type data --SC_size 88 \
        --t_end 30000 --tr 1 --cut 10 --out_dir $OUT --i_start 0 --i_end 1"

echo "[probe] task $SLURM_ARRAY_TASK_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c 'import jax; print("jax", jax.__version__, jax.devices())'

START=$SECONDS
case "$SLURM_ARRAY_TASK_ID" in
  0) python microbench_batch.py --sc_size 88 --reps 3 \
            --batches 1 4 16 64 256 1024 4096 8192 16384 \
            --outdir "$OUT" ;;
  1) python sbc_run.py $COMMON --sampler smc_lik --which_stat FC  ;;
  2) python sbc_run.py $COMMON --sampler smc_abc --which_stat FC  ;;
  3) python sbc_run.py $COMMON --sampler smc_lik --which_stat FCD ;;
  4) python sbc_run.py $COMMON --sampler smc_abc --which_stat FCD ;;
esac
RC=$?
ELAPSED=$((SECONDS - START))

kill $SAMPLER_PID 2>/dev/null
PEAK=$(sort -n "$MEMLOG" 2>/dev/null | tail -1)
echo "[probe] task $SLURM_ARRAY_TASK_ID rc=$RC elapsed=${ELAPSED}s peak_gpu_mem=${PEAK}MiB"
echo "$SLURM_ARRAY_TASK_ID,$RC,$ELAPSED,$PEAK" >> "$OUT/probe_summary.csv"
