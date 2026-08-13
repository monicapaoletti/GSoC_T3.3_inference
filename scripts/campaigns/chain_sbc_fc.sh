#!/usr/bin/env bash
# Wait for the calibrated CPU ABC run to finish, then run the SBC campaign on FC.
#
# The existing campaign covers FCD only, which is the standing limit in the manuscript's
# scope note. FC changes exactly ONE variable -- same 10-node network, same 4096
# particles, same HalfNormal(0.5) prior, same 100 replicates per sampler -- so the result
# is directly comparable to the FCD one and turns "FCD only" into "both features". The
# 88-node alternative was rejected: device memory forces 1024 particles there (88x4096
# needs ~44 GB against a 23 GB card), changing network size AND budget at once, and the
# FCD lag cut k=30 does not describe the same statistic at a different window count.
#
# Cost: FC extraction is cheaper than FCD (45 features vs 231 at 10 nodes) while the
# forward integration is identical, so expect ~= the FCD campaign, roughly 11-12 h.
set -u
source /CNSdata/mpaolett/env.sh
cd /CNSdata/mpaolett/GSoC_T3.3_inference/scripts/functions || exit 1

OUT=/CNSdata/mpaolett/GSoC_T3.3_inference/results/sbc
mkdir -p "$OUT"
DEADLINE=$(( $(date +%s) + 12*3600 ))

echo "[sbc_fc] armed $(date); waiting for the calibrated CPU ABC run to exit"
while pgrep -u "$(id -u)" -f "[a]bc_cpu_calibrated" >/dev/null 2>&1; do
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    echo "[sbc_fc] ABORT: ABC run still going after 12 h; not starting"
    exit 1
  fi
  sleep 300
done
echo "[sbc_fc] ABC finished $(date)"

# MPS is what makes 4 workers per card worth running at all: without it four processes
# time-slice and cost 40% of throughput (measured 0.60x), with it they reach 2.35x.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_MPS_PIPE_DIRECTORY=/CNSdata/mpaolett/mps_pipe
export CUDA_MPS_LOG_DIRECTORY=/CNSdata/mpaolett/mps_log
if ! pgrep -u "$(id -u)" -f nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo "[sbc_fc] MPS daemon not running -- starting it"
  /CNSdata/mpaolett/mps_ctl.sh start
fi

# Identical to sbc_4x.sh except --which_stat FC. Filenames and the ranks file both embed
# the statistic, so FC and FCD results coexist in the same directory without collision.
COMMON="--L 100 --scale 0.5 --n_particles 4096 --which_stat FC \
        --SC_type data --SC_size 10 --t_end 30000 --tr 1 --cut 10 --out_dir $OUT"

echo "[sbc_fc] start $(date)"
PIDS=""
for k in 0 1 2 3; do
  S=$((k * 25)); E=$((S + 25))
  CUDA_VISIBLE_DEVICES=0 python3 sbc_run.py $COMMON --sampler smc_lik \
      --i_start $S --i_end $E > "$OUT/fc_lik_$k.log" 2>&1 &
  PIDS="$PIDS $!"
  CUDA_VISIBLE_DEVICES=1 python3 sbc_run.py $COMMON --sampler smc_abc \
      --i_start $S --i_end $E > "$OUT/fc_abc_$k.log" 2>&1 &
  PIDS="$PIDS $!"
done
echo "[sbc_fc] launched 8 workers:$PIDS"
for p in $PIDS; do wait $p; done
echo "[sbc_fc] all workers done $(date)"

# REQUIRED, not cosmetic: the 4 shards of a sampler all write the same
# sbc_ranks_<sampler>_<stat>.npz (sbc_run.py:180 has no shard suffix), so during the
# parallel phase they race and the last to exit leaves a partial file. This full-range
# pass re-ranks everything that is now on disk and writes the complete one.
for S in smc_lik smc_abc; do
  python3 sbc_run.py $COMMON --sampler $S
done
echo "[sbc_fc] finished $(date)"
