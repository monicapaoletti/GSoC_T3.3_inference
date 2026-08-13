#!/usr/bin/env bash
# Do N concurrent processes on ONE GPU overlap, or do they time-slice?
#
# Runs microbench_batch.py at the SMC's real batch (4096) as 1 process, then as
# 4 concurrent processes on the same card, and reports aggregate throughput.
#   time-slicing  -> 4-proc aggregate ~= solo throughput  (no gain)
#   real overlap  -> 4-proc aggregate ~= up to 2.4x solo  (the occupancy headroom)
#
# Usage: mps_bench.sh <label>     (label: nomps | mps)
# With MPS, export CUDA_MPS_PIPE_DIRECTORY before calling; it is inherited.
set -u
source /CNSdata/mpaolett/env.sh >/dev/null 2>&1
cd /CNSdata/mpaolett/GSoC_T3.3_inference/scripts/functions || exit 1

LABEL="${1:-nomps}"
D=/CNSdata/mpaolett/mps_bench/$LABEL
rm -rf "$D"; mkdir -p "$D"

export CUDA_VISIBLE_DEVICES=0              # isolate a single card
export XLA_PYTHON_CLIENT_PREALLOCATE=false # same as the SBC run
PY=/CNSdata/mpaolett/jax_GSoC/bin/python3
ARGS="--batches 4096 --reps 3 --sc_size 10"

echo "===== [$LABEL] MPS_PIPE=${CUDA_MPS_PIPE_DIRECTORY:-<unset>} ====="

echo "--- 1 process ---"
t0=$SECONDS
$PY microbench_batch.py $ARGS --outdir "$D/solo" > "$D/solo.log" 2>&1
solo_wall=$((SECONDS - t0))
solo_line=$(grep -E "^  4096" "$D/solo.log" || echo "  4096  FAILED")
echo "wall=${solo_wall}s | $solo_line"

echo "--- 4 concurrent processes ---"
t0=$SECONDS
for k in 0 1 2 3; do
  $PY microbench_batch.py $ARGS --outdir "$D/p$k" > "$D/p$k.log" 2>&1 &
done
wait
four_wall=$((SECONDS - t0))
echo "wall=${four_wall}s"
for k in 0 1 2 3; do
  echo "  p$k: $(grep -E '^  4096' "$D/p$k.log" || echo FAILED)"
done

# aggregate throughput = sum of per-process evals_per_s (col 4 of the 4096 row)
awk -v lbl="$LABEL" '
  FILENAME ~ /solo\.log$/ && $1==4096 { solo=$4 }
  FILENAME ~ /p[0-3]\.log$/ && $1==4096 { agg+=$4; n++ }
  END {
    if (solo>0 && n>0)
      printf "\n[%s] solo=%.1f evals/s | %d-proc aggregate=%.1f evals/s | SPEEDUP=%.2fx\n",
             lbl, solo, n, agg, agg/solo
    else print "\n[" lbl "] could not parse throughput"
  }' "$D/solo.log" "$D"/p[0-3].log
