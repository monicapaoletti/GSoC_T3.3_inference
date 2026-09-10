#!/usr/bin/env bash
# Submit one 88-node SBC campaign: the 200-replicate array plus the aggregation pass that
# depends on it.
#
# Usage:  ./sbc_n88_submit.sh FC 1d [concurrency]
#         ./sbc_n88_submit.sh FCD 2d 8
#
# The aggregation job is NOT redundant and must not be dropped to save time. sbc_run.py
# writes sbc_ranks_<sampler>_<stat>.npz from whatever draws it can see, and the array
# tasks each see only their own replicate -- so the complete ranks file, and the KS test
# and plot built from it, only exist after a pass that runs over the full range with
# every draws_*.npz already on disk. Resume logic makes that pass cheap: every replicate
# is skipped, and only the ranking is recomputed.
#
# afterany, not afterok: a single failed replicate should not sink the aggregation of the
# other 199. Check the array's exit codes and re-run the stragglers, which resume cleanly.
set -euo pipefail

STAT="${1:?usage: sbc_n88_submit.sh <FC|FCD> <1d|2d> [concurrency]}"
MODE="${2:?usage: sbc_n88_submit.sh <FC|FCD> <1d|2d> [concurrency]}"
CONC="${3:-16}"        # litleo has 16 GPUs; throttle so one campaign cannot take them all

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# SUFFIX names a NEW campaign in a NEW directory. It exists because sbc_run.py RESUMES:
# a replicate whose draws_*.npz is already there is skipped. Re-submitting into a finished
# campaign's directory therefore skips all 200 replicates and merely re-aggregates the old
# draws -- producing a "new" ranks file, on a new date, that is the old result. The guard
# below refuses that outright; SUFFIX is how you ask for a genuinely fresh run.
OUT="$VSC_DATA_VO_USER/results/sbc_n88_${MODE}_${STAT}${SUFFIX:+_${SUFFIX}}"
if [ -n "$(ls "$OUT"/draws_*.npz 2>/dev/null | head -1)" ] && [ -z "${RESUME:-}" ]; then
  echo "ERROR: $OUT already holds $(ls "$OUT"/draws_*.npz | wc -l) draws files." >&2
  echo "       Those replicates would be SKIPPED, not re-run, and the aggregation would" >&2
  echo "       rebuild the ranks from them. Set SUFFIX=<name> for a fresh campaign, or" >&2
  echo "       RESUME=1 if you really are finishing an interrupted one." >&2
  exit 1
fi
mkdir -p "$OUT/logs"
cd "$OUT"

DEPARG=(); [ -n "${DEP:-}" ] && DEPARG=(--dependency="$DEP")

AID=$(sbatch --parsable "${DEPARG[@]}" --array="0-199%${CONC}" \
        --export=ALL,STAT="$STAT",MODE="$MODE",OUT="$OUT" \
        "$HERE/sbc_n88_array.sh" | cut -d';' -f1)
echo "[submit] replicate array: $AID  ($STAT, $MODE, ${CONC} concurrent)"

# Build the 2-D flags before the sbatch call rather than inside the --wrap string:
# a command substitution nested in the quoting of --wrap is evaluated at the wrong time
# and silently yields an empty string, which would turn a 2-D aggregation into a 1-D one.
# These flags must match sbc_n88_array.sh exactly. The aggregation normally skips every
# replicate, but if one is missing it RUNS it -- and a replicate run with a different
# kernel or a different prior than its 199 siblings silently corrupts the ranks file.
EXTRA=""
[ "$MODE" = "2d" ] && EXTRA="--infer_eta --eta_prior_scale 0.1"
[ "${ADAPT:-1}" = "1" ] && EXTRA="$EXTRA --adapt_rw"

AGG="source \$VSC_DATA_VO_USER/env.sh
cd \$GSOC_REPO/scripts/functions
for S in smc_lik smc_abc; do
  python sbc_run.py --L 100 --scale 0.5 --n_particles 4096 --which_stat $STAT \
    --SC_type data --SC_size 88 --t_end 30000 --tr 1 --cut 10 \
    --out_dir $OUT $EXTRA --sampler \$S
done"

BID=$(sbatch --parsable --dependency="afterany:${AID}" \
        --job-name="sbc88agg" --time=02:00:00 --ntasks=1 --gpus-per-node=1 \
        --cpus-per-task=24 --mem=120G --output="$OUT/logs/sbc88agg-%j.out" \
        --wrap="$AGG" | cut -d';' -f1)
echo "[submit] aggregation:     $BID  (after $AID)"
echo "[submit] out_dir:         $OUT"
echo "JOBIDS=${AID}:${BID}"
