#!/usr/bin/env bash
# One SBC replicate at 88 nodes. Submitted as a 200-task array by sbc_n88_submit.sh.
#
# WHY AN ARRAY AND NOT THE DOSLIS SHAPE. chain_sbc_fc.sh ran eight background workers
# pinned to two cards with MPS, because doslis had exactly two GPUs and no scheduler, so
# packing them was the only parallelism available. Neither half of that reasoning holds
# here. litleo has 16 GPUs behind Slurm, so the parallelism comes from the scheduler; and
# the probe showed 4096 particles at 88 nodes is nowhere near saturating an H100
# (throughput still climbing at batch 16384), so the occupancy gap MPS existed to fill is
# not there. One replicate per GPU, no MPS, no CUDA_VISIBLE_DEVICES juggling.
#
# It also removes the ranks-file race the doslis README warns about: shards no longer run
# concurrently against one filename, because each task writes only its own draws_*.npz.
# The full-range pass still runs, as a dependent job, and is still what writes the
# complete ranks file -- see sbc_n88_submit.sh.
#
# Measured 2026-09-08 on litleo (H100 NVL): 1658-1681 s per replicate across all four
# sampler x statistic corners, peak device memory 5.1 GB (FC) / 37.8 GB (FCD). The
# 1h30 walltime is ~3x the measured cost, which is margin for a slower card, not slack.
#SBATCH --job-name=sbc88
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --output=logs/sbc88-%A_%a.out
set -uo pipefail

source "$VSC_DATA_VO_USER/env.sh"
cd "$GSOC_REPO/scripts/functions" || exit 1

: "${STAT:?set STAT=FC or FCD}"
: "${MODE:?set MODE=1d or 2d}"
: "${OUT:?set OUT}"

# Tasks 0-99 are smc_lik replicates 0-99; tasks 100-199 are smc_abc replicates 0-99.
# sbc_run.py draws the ground truths for the full --L before slicing, so replicate k is
# the same truth in every task -- that is what makes this decomposition legitimate.
T=$SLURM_ARRAY_TASK_ID
if [ "$T" -lt 100 ]; then SAMPLER=smc_lik; I=$T; else SAMPLER=smc_abc; I=$((T - 100)); fi

COMMON="--L 100 --scale 0.5 --n_particles 4096 --which_stat $STAT \
        --SC_type data --SC_size 88 --t_end 30000 --tr 1 --cut 10 --out_dir $OUT"
# sigma = 0.1, not the 0.5 default: eta_prior_scale is a MULTIPLICATIVE spread on |eta|,
# and 0.5 would draw most SBC truths outside the window where the data constrain eta at
# all, making the ranks uniform by construction and the test vacuous. See chain_sbc_2d.sh.
[ "$MODE" = "2d" ] && COMMON="$COMMON --infer_eta --eta_prior_scale 0.1"

# ADAPT THE SMC RANDOM-WALK SCALE (ADAPT=0 to disable). The 2026-09-09 campaign ran with
# the fixed rw_step=0.1 and the kernel collapsed: smc_lik accepted 0.2% of moves in 2-D and
# the 4096-particle cloud held 254-419 unique values, so its rank histogram described the
# rejuvenation, not the posterior. The weight ESS read ~4000 throughout and gave no warning.
# Runs with this on are NOT comparable against that campaign; re-run both modes together.
[ "${ADAPT:-1}" = "1" ] && COMMON="$COMMON --adapt_rw"

echo "[sbc88] task $T -> $SAMPLER replicate $I ($STAT, $MODE) on $(hostname) $(date)"
START=$SECONDS
python sbc_run.py $COMMON --sampler "$SAMPLER" --i_start "$I" --i_end "$((I + 1))"
RC=$?
echo "[sbc88] task $T rc=$RC elapsed=$((SECONDS - START))s $(date)"
exit $RC
