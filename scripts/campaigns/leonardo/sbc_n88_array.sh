#!/usr/bin/env bash
# One SBC replicate at 88 nodes on Leonardo. Submitted as a 200-task array by
# sbc_n88_submit.sh -- do not sbatch this directly (account/partition come from there).
#
# A port of scripts/campaigns/ugent/sbc_n88_array.sh: same sbc_run.py flags, so a
# Leonardo FCD campaign compares replicate-for-replicate with UGent's
# sbc_n88_{1d,2d}_FCD_adapt (sbc_run.py draws all L truths before slicing, so replicate k
# has the same ground truth on both clusters).
#
# Measured on Leonardo's A100 (probe, 2026-11-25): 2225-2290 s per replicate (H100: 1658-
# 1681 s). FCD at 4096 particles fits the 64 GB card. 1h15 is ~2x the measured time.
#SBATCH --job-name=sbc88L
#SBATCH --time=01:15:00
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
