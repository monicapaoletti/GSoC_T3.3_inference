#!/usr/bin/env bash
# Re-run the benchmark points that finished sampling but never wrote a benchmark row.
#
# WHY THIS EXISTS. A cell writes summary_<TAG>.csv right after sampling and
# benchmark_<TAG>.csv several steps later, past the plots. Between 2026-09-09 and
# 2026-09-10 an unguarded plot_trace_chains sat in that gap: arviz's per-chain KDE raises
# `OverflowError: cannot convert float infinity to integer` on a chain holding exactly two
# distinct values (one that accepted a single move -- a fully constant chain is caught by
# arviz's own guard and plots fine), and the process died before the benchmark write. 22
# cells, 2.5-3.5 h of sampling each, left a summary and no row. The plots are guarded now,
# so this only has to clean up what was already lost.
#
# THE GAP IS DERIVED, NOT HARDCODED. Every summary_*.csv without a matching
# benchmark_*.csv is a lost point, so this script rescans at submit time and always asks
# for exactly what is missing right now. Re-running it after a partial recovery is safe
# and idempotent; running it while cells are still in flight is not (a point that is
# mid-run has a summary and no row yet, and would be resubmitted on top of itself) --
# chain it behind the current jobs with DEP, which is what pipeline_n88.sh does.
#
# ONE JOB PER (statistic, sampler, G), asking only for the missing batch sizes via
# --batch_sizes, so the points that did land are not recomputed. That is the whole reason
# run_gpu_pool.py takes --batch_sizes: a cell is a 4-point sweep, and without the override
# recovering one lost point means paying for four.
#
# Usage:  ./rerun_missing_n88.sh                     # submit whatever is missing now
#         DRY=1 ./rerun_missing_n88.sh               # print the plan, submit nothing
#         DEP=afterany:123:456 ./rerun_missing_n88.sh # hold until those finish
set -euo pipefail

OUT="$VSC_DATA_VO_USER/results/bench_n88"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT/logs"

# Emit one "STAT SAMPLER G BATCH[,BATCH...]" line per group that is missing points.
# The batched axis lives in a different filename field per sampler family (nc for the
# chain-vmapped MCMC, np for SMC), which is why this is parsed rather than globbed.
# SYSTEM python, not the venv's. The scan needs nothing but the stdlib, while jax_GSoC's
# interpreter is a tuned build that SIGILLs (exit 132) on some of the VSC login nodes --
# and `ssh login.hpc.ugent.be` round-robins across them, so it fails on one connection and
# works on the next. Nothing here should depend on which node answered.
#
# The scan is also guarded rather than left to `set -e`: a bare PLAN=$(...) that dies takes
# the whole script down with no message and no submission, which is indistinguishable from
# "nothing was missing" -- exactly how the first attempt at this looked.
PY_SCAN="${PY_SCAN:-/usr/bin/python3}"
set +e
PLAN=$("$PY_SCAN" - "$OUT" <<'PYEOF'
import os, re, sys
from collections import defaultdict

out = sys.argv[1]
pat = re.compile(
    r"^summary_G(?P<G>[0-9.]+)_.*?_nc(?P<nc>\d+)_SC_88_sampler_(?P<sampler>[a-z_]+)"
    r"_which_stat_(?P<stat>[A-Z]+)_gpu_cmvectorized_np(?P<np>\d+)\.csv$")

missing = defaultdict(set)
for fn in sorted(os.listdir(out)):
    m = pat.match(fn)
    if not m:
        continue
    if os.path.exists(os.path.join(out, fn.replace("summary_", "benchmark_", 1))):
        continue
    d = m.groupdict()
    # n_chains is the swept axis for rwmh/demc/slice; n_particles for every SMC flavour.
    batch = d["nc"] if d["sampler"] in ("rwmh", "demc", "slice") else d["np"]
    missing[(d["stat"], d["sampler"], d["G"])].add(int(batch))

for (stat, sampler, g), batches in sorted(missing.items()):
    print(f"{stat} {sampler} {g} {','.join(str(b) for b in sorted(batches))}")
PYEOF
)
SCAN_RC=$?
set -e
if [ "$SCAN_RC" -ne 0 ]; then
  echo "[rerun] ERROR: scan of $OUT failed (exit $SCAN_RC); nothing submitted." >&2
  echo "[rerun] retry -- a different login node usually works." >&2
  exit 1
fi

if [ -z "$PLAN" ]; then echo "[rerun] nothing missing -- every summary has a benchmark row"; exit 0; fi

SUBMITTED=""
while read -r STAT SAMPLER G BATCHES; do
  [ -z "${STAT:-}" ] && continue
  BL="${BATCHES//,/ }"
  N=$(echo "$BL" | wc -w)
  case "$SAMPLER" in
    rwmh|demc|slice)
      # ~3.5 h at the top of the sweep; bill every point at 4 h and never dip under 6.
      HOURS=$(( N * 4 )); [ "$HOURS" -lt 6 ] && HOURS=6
      ARGS="--suite mcmc --mcmc_samplers $SAMPLER --n_warmup 1000 --n_samples 1000" ;;
    smc_abc_demc)
      HOURS=$(( N * 2 )); [ "$HOURS" -lt 4 ] && HOURS=4
      ARGS="--suite smc --smc_samplers smc_abc --smc_move demc --n_stages 50 --n_mcmc 5" ;;
    smc_lik|smc_abc)
      HOURS=$(( N * 2 )); [ "$HOURS" -lt 4 ] && HOURS=4
      ARGS="--suite smc --smc_samplers $SAMPLER --n_stages 50 --n_mcmc 5" ;;
    *) echo "[rerun] SKIP unknown sampler '$SAMPLER'" >&2; continue ;;
  esac

  NAME="r88_${SAMPLER}_${STAT}_G${G}"
  CMD="source \$VSC_DATA_VO_USER/env.sh
cd \$GSOC_REPO/scripts/functions
python run_gpu_pool.py $ARGS --which_stat $STAT --G $G --batch_sizes $BL \
  --SC_size 88 --gpu_only --gpu_concurrency 1 --results_dir $OUT"

  if [ -n "${DRY:-}" ]; then
    echo "[dry] $NAME (${HOURS}h): batches=$BL"
    continue
  fi
  dep=(); [ -n "${DEP:-}" ] && dep=(--dependency="$DEP")
  id=$(sbatch --parsable "${dep[@]}" --job-name="$NAME" --time="${HOURS}:00:00" \
        --ntasks=1 --gpus-per-node=1 --cpus-per-task=24 --mem=120G \
        --output="$OUT/logs/${NAME}-%j.out" --wrap="$CMD" | cut -d';' -f1)
  SUBMITTED="${SUBMITTED:+$SUBMITTED:}$id"
  echo "[submit] $id  $NAME  batches=$BL  (${HOURS}h)"
done <<< "$PLAN"

echo "[rerun] out_dir: $OUT"
echo "JOBIDS=$SUBMITTED"
