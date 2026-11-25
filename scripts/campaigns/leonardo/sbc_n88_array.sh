#!/usr/bin/env bash
# One SBC replicate at 88 nodes on Leonardo. Submitted as a 200-task array by
# sbc_n88_submit.sh -- do not sbatch this directly (account/partition come from there).
#
# A port of scripts/campaigns/ugent/sbc_n88_array.sh: same sbc_run.py flags, so a
# Leonardo FCD campaign compares replicate-for-replicate with UGent's
# sbc_n88_{1d,2d}_FCD_adapt (sbc_run.py draws all L truths before slicing, so replicate k
# has the same ground truth on both clusters).
#
# UGent measured 1658-1681 s per replicate on an H100 NVL, peak device memory 5.1 GB (FC)
# and 37.8 GB (FCD). Both fit an A100 64 GB. The 2 h walltime allows for a card up to
# ~4x slower.
#SBATCH --job-name=sbc88L
#SBATCH --time=02:00:00
set -uo pipefail

: "${GSOC_REPO:?}"; : "${STAT:?set STAT=FC or FCD}"; : "${MODE:?set MODE=1d or 2d}"; : "${OUT:?}"
source "$GSOC_REPO/scripts/campaigns/leonardo/config.sh"
eval "$LEO_ENV_SETUP"
cd "$GSOC_REPO/scripts/functions" || exit 1

# Tasks 0-99: smc_lik replicates 0-99; tasks 100-199: smc_abc replicates 0-99.
T=$SLURM_ARRAY_TASK_ID
if [ "$T" -lt 100 ]; then SAMPLER=smc_lik; I=$T; else SAMPLER=smc_abc; I=$((T - 100)); fi

echo "[sbc88L] task $T -> $SAMPLER replicate $I ($STAT, $MODE) on $(hostname) $(date)"
nvidia-smi -L
START=$SECONDS
# shellcheck disable=SC2086  # SBC_FLAGS is a deliberately word-split flag list
python sbc_run.py $SBC_FLAGS --sampler "$SAMPLER" --i_start "$I" --i_end "$((I + 1))"
RC=$?
echo "[sbc88L] task $T rc=$RC elapsed=$((SECONDS - START))s $(date)"
exit $RC
