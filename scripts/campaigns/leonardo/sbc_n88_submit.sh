#!/usr/bin/env bash
# Submit one 88-node SBC campaign on Leonardo: a 200-replicate array (100 smc_lik +
# 100 smc_abc) plus the aggregation pass that depends on it.
#
# Usage:  (account EUHPC_B40_021 is the default, see config.sh)
#         ./sbc_n88_submit.sh FC 1d [concurrency]  # NEW: FC has never been SBC-tested
#         ./sbc_n88_submit.sh FC 2d
#         ./sbc_n88_submit.sh FCD 1d               # repeat of UGent sbc_n88_1d_FCD_adapt
#         ./sbc_n88_submit.sh FCD 2d               # repeat of UGent sbc_n88_2d_FCD_adapt
#         DRY=1 / DEP=afterany:<id> / ADAPT=0 / SUFFIX=<name> / RESUME=1
#         STD_REF_G=0.33 ./sbc_n88_submit.sh FCD 1d   # standardisation pinned at G=0.33
#           for every replicate instead of at each one's own truth (calibration test 2,
#           2026-11-26). Goes to its own _stdref<G> directory unless SUFFIX is set.
#         LIK_ONLY=1 ...   # smc_lik only: a 100-task array, half the cost
#
# Flags are built ONCE here and handed to both the array and the aggregation, so the two
# cannot drift. That matters: the aggregation normally skips every replicate, but RUNS any
# that is missing -- and a replicate run with a different kernel or prior than its 199
# siblings silently corrupts the ranks file.
#
# The aggregation is not redundant: each array task sees only its own draws_*.npz, so the
# complete ranks file (and the KS test built from it) exists only after a pass over the
# full range. afterany, not afterok: one failed replicate must not sink the other 199 --
# check the array's exit codes, re-run stragglers with RESUME=1.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

STAT="${1:?usage: sbc_n88_submit.sh <FC|FCD> <1d|2d> [concurrency]}"
MODE="${2:?usage: sbc_n88_submit.sh <FC|FCD> <1d|2d> [concurrency]}"
CONC="${3:-16}"
case "$STAT" in FC|FCD) ;; *) echo "STAT must be FC or FCD" >&2; exit 2;; esac
case "$MODE" in 1d|2d) ;; *) echo "MODE must be 1d or 2d" >&2; exit 2;; esac

# Naming matches the UGent dirs (sbc_n88_<mode>_<stat>_adapt) plus _leonardo, so the two
# can sit side by side under results/. Do NOT point make_paper_assets.py at both at once:
# _sbc_files does not dedupe, and it would pool the two clusters as one sampler.
TAG=""; [ "${ADAPT:-1}" = "1" ] && TAG="_adapt"
# The draws file names do not record the standardisation reference, so a pinned campaign
# must never share a directory with an unpinned one.
[ -n "${STD_REF_G:-}" ] && SUFFIX="${SUFFIX:-stdref${STD_REF_G}}"
OUT="$LEO_RESULTS/sbc_n88_${MODE}_${STAT}${TAG}_leonardo${SUFFIX:+_${SUFFIX}}"

# sbc_run.py RESUMES: replicates whose draws_*.npz exist are skipped. Submitting into a
# finished campaign's directory would re-aggregate the old draws under a new date.
if [ -n "$(ls "$OUT"/draws_*.npz 2>/dev/null | head -1)" ] && [ -z "${RESUME:-}" ]; then
  echo "ERROR: $OUT already holds $(ls "$OUT"/draws_*.npz | wc -l) draws files." >&2
  echo "       Set SUFFIX=<name> for a fresh campaign, or RESUME=1 to finish this one." >&2
  exit 1
fi

SBC_FLAGS="--L 100 --scale 0.5 --n_particles 4096 --which_stat $STAT \
--SC_type data --SC_size 88 --t_end 30000 --tr 1 --cut 10 --out_dir $OUT"
# eta_prior_scale 0.1 is a MULTIPLICATIVE spread on |eta|; 0.5 would put most truths where
# the data do not constrain eta and make the test vacuous. Same as UGent.
[ "$MODE" = "2d" ] && SBC_FLAGS="$SBC_FLAGS --infer_eta --eta_prior_scale 0.1"
# Without adaptation the n=88 SMC kernel collapses (accept 0.002, 6-10% unique particles
# in 2-D). On by default, as in the UGent _adapt campaigns this one is compared against.
[ "${ADAPT:-1}" = "1" ] && SBC_FLAGS="$SBC_FLAGS --adapt_rw"
[ -n "${STD_REF_G:-}" ] && SBC_FLAGS="$SBC_FLAGS --std_ref_G $STD_REF_G"
# Tasks 0-99 are smc_lik, 100-199 smc_abc (sbc_n88_array.sh), so LIK_ONLY just stops at 99.
LAST=199; SAMPLERS="smc_lik smc_abc"
[ -n "${LIK_ONLY:-}" ] && { LAST=99; SAMPLERS="smc_lik"; }

if [ -n "${DRY:-}" ]; then
  echo "[dry] out: $OUT"; echo "[dry] array 0-${LAST}%$CONC, flags: $SBC_FLAGS"; exit 0
fi
mkdir -p "$OUT/logs"
mapfile -t FLAGS < <(leo_sbatch_flags)
DEPARG=(); [ -n "${DEP:-}" ] && DEPARG=(--dependency="$DEP")
# Through the environment, not --export=K=V: SBC_FLAGS holds spaces.
export GSOC_REPO LEO_ACCOUNT LEO_PARTITION LEO_QOS LEO_MAXH LEO_HOME LEO_RESULTS JAX_MODULE
export STAT MODE OUT SBC_FLAGS

AID=$(sbatch --parsable "${DEPARG[@]}" "${FLAGS[@]}" --array="0-${LAST}%${CONC}" \
        --output="$OUT/logs/sbc88L-%A_%a.out" --export=ALL \
        "$HERE/sbc_n88_array.sh" | cut -d';' -f1)
echo "[submit] replicate array: $AID  ($STAT, $MODE, ${CONC} concurrent)"

AGG="$LEO_ENV_SETUP
cd $GSOC_REPO/scripts/functions
for S in $SAMPLERS; do python sbc_run.py $SBC_FLAGS --sampler \$S; done"
BID=$(sbatch --parsable --dependency="afterany:${AID}" "${FLAGS[@]}" \
        --job-name="sbc88Lagg" --time=02:00:00 \
        --output="$OUT/logs/sbc88Lagg-%j.out" --wrap="$AGG" | cut -d';' -f1)
echo "[submit] aggregation:     $BID  (after $AID)"
echo "[submit] out_dir:         $OUT"
echo "JOBIDS=${AID}:${BID}"
