#!/usr/bin/env bash
# Feasibility probe for the 88-node campaign on HPC-UGent (litleo, H100 NVL 93.6 GiB).
#
# WHY THIS RUNS BEFORE ANY CAMPAIGN. On doslis the 88-node run was rejected outright:
# "88x4096 needs ~44 GB against a 23 GB card" (chain_sbc_fc.sh:8), which forced 1024
# particles and would have changed network size AND budget at once. An H100 NVL has
# 93.6 GiB, so 4096 particles should now fit -- but "should fit" is a calculation, not a
# measurement, and the whole campaign is sized off the answer. Two things are unknown and
# only one of them is memory:
#   (a) does 88x4096 actually fit, and where is the throughput optimum on this card
#       (the analogue of the BN ~ 640 CPU working set and the 4096-on-L4 occupancy);
#   (b) what does ONE real replicate cost at 88 nodes, per sampler and per statistic.
# (b) is the number that decides everything downstream: the campaign is 100 replicates x
# 2 samplers, the Tier-2 walltime cap is 71h59m, and the known ~17 min/replicate non-SMC
# overhead (fresh subprocess per replicate, sbc_run.py:115) is charged 100 times. If a
# replicate costs more than ~40 min the campaign cannot be a single job at any width and
# has to become a job array, which is a different script.
#
# Task 0 is the batch sweep; tasks 1-4 are one real replicate each, at the campaign's own
# settings, over the four sampler x statistic corners. They are one array job because
# litleo has 16 GPUs and an empty queue, so the corners cost the same wall time as one.
#
# FCD is run with the lag cut k=30 unchanged and is reported as a DIFFERENT statistic
# from the 10-node FCD, not a comparable one -- a deliberate choice, not an oversight.
#SBATCH --job-name=probe88
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --array=0-4
#SBATCH --output=%x-%A_%a.out
set -uo pipefail

source "$VSC_DATA_VO_USER/env.sh"
cd "$GSOC_REPO/scripts/functions" || exit 1

OUT="$VSC_DATA_VO_USER/results/probe_n88"
mkdir -p "$OUT"

# Peak device memory is the point of task 0 and a guardrail for the rest, and JAX will not
# report it once preallocation is off -- so sample the card instead.
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

START=$SECONDS
case "$SLURM_ARRAY_TASK_ID" in
  0) # Batch sweep. 4096 is the campaign budget; 8192/16384 probe how much headroom is
     # left above it, which is what decides whether replicates can share a card later.
     python microbench_batch.py --sc_size 88 --reps 3 \
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
