#!/usr/bin/env bash
# Calibration test 1 (2026-11-26): the EXACT 1-D posterior on a grid of G, for the same
# 100 truths and seeds as the n=88 1-D SBC campaigns, with the feature standardisation
# both at each replicate's truth (as every campaign ran) and pinned at STD_REF_G.
# Separates "the SMC sampler is wrong" from "the setup depends on the truth" -- see the
# docstring of scripts/functions/sbc_grid.py.
#
# Usage:  ./grid_n88_submit.sh FCD            # one statistic, 100-task array + aggregation
#         ./grid_n88_submit.sh FC FCD         # both
#         DRY=1 / DEP=afterany:<id> / STD_REF_G=0.33 (default) / CONC=16
#
# Cost: per task 4 calls of mpr_jax_numpyro.py --profile_grid (coarse + fine, two
# standardisations), each ~2 x 4001 batched forward runs plus compile and the 8 reference
# simulations. An SMC replicate does ~300 batched calls of 4096 in ~2250 s on the A100, so
# a task should take ~10 min; 1 h walltime is generous. 4001 < 4096 particles, which is
# known to fit the 64 GB card for FCD.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

STATS=("${@:-FCD}")
REF="${STD_REF_G:-0.33}"
CONC="${CONC:-16}"
mapfile -t FLAGS < <(leo_sbatch_flags)
DEPARG=(); [ -n "${DEP:-}" ] && DEPARG=(--dependency="$DEP")
ALL=""
for STAT in "${STATS[@]}"; do
  case "$STAT" in FC|FCD) ;; *) echo "STAT must be FC or FCD" >&2; exit 2;; esac
  OUT="$LEO_RESULTS/grid_n88_1d_${STAT}"
  SMC="$LEO_RESULTS/sbc_n88_1d_${STAT}_adapt_leonardo"
  ARGS="--which_stat $STAT --out_dir $OUT --std_ref_G $REF"
  TASK="$LEO_ENV_SETUP
cd $GSOC_REPO/scripts/functions
python sbc_grid.py run $ARGS --i \$SLURM_ARRAY_TASK_ID"
  AGG="$LEO_ENV_SETUP
cd $GSOC_REPO/scripts/functions
python sbc_grid.py aggregate $ARGS --smc_dir $SMC"
  if [ -n "${DRY:-}" ]; then
    echo "[dry] $STAT -> $OUT"; echo "[dry] task: python sbc_grid.py run $ARGS --i <0-99>"
    echo "[dry] aggregate: python sbc_grid.py aggregate $ARGS --smc_dir $SMC"; continue
  fi
  mkdir -p "$OUT/logs"
  AID=$(sbatch --parsable "${DEPARG[@]}" "${FLAGS[@]}" --array="0-99%${CONC}" \
          --job-name="grid88_${STAT}" --time=01:00:00 \
          --output="$OUT/logs/grid88-%A_%a.out" --wrap="$TASK" | cut -d';' -f1)
  BID=$(sbatch --parsable --dependency="afterany:${AID}" "${FLAGS[@]}" \
          --job-name="grid88agg_${STAT}" --time=00:30:00 \
          --output="$OUT/logs/grid88agg-%j.out" --wrap="$AGG" | cut -d';' -f1)
  echo "[submit] $STAT: array $AID, aggregation $BID -> $OUT"
  ALL="${ALL:+$ALL:}$AID:$BID"
done
echo "JOBIDS=$ALL"
