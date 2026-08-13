#!/usr/bin/env bash
# Wait for the 1-D FC SBC, then run the 2-D (G, eta) SBC: 100 (G*, eta*) pairs per
# sampler, both inferred jointly.
#
# WHY sigma = 0.1 AND NOT THE 0.5 DEFAULT. eta_prior_scale is the sigma of
# LogNormal(log 4.6, sigma) on |eta| -- a MULTIPLICATIVE spread -- so 0.5 puts 95% of the
# prior on |eta| in [1.7, 12.5] while the live window is a few percent wide. That is not
# merely wasteful here, it would INVALIDATE the test: SBC draws its truths from the prior,
# so most replicates would land where the data cannot constrain eta, the posterior would
# return the prior, and the ranks would come out uniform BY CONSTRUCTION. A pass would
# mean nothing. sigma = 0.1 spans |eta| in about [3.8, 5.6], keeping the truths inside the
# region the eta sweeps show is alive.
#
# Ranks are kept per parameter: a joint posterior can be calibrated in G and not in eta,
# and that asymmetry is the interesting outcome, not something to average away.
set -u
source /CNSdata/mpaolett/env.sh
cd /CNSdata/mpaolett/GSoC_T3.3_inference/scripts/functions || exit 1

OUT=/CNSdata/mpaolett/GSoC_T3.3_inference/results/sbc
mkdir -p "$OUT"
DEADLINE=$(( $(date +%s) + 20*3600 ))

echo "[sbc2d] armed $(date); waiting for the 1-D FC SBC to exit"
while pgrep -u "$(id -u)" -f "[c]hain_sbc_fc" >/dev/null 2>&1; do
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    echo "[sbc2d] ABORT: 1-D FC SBC still running after 20 h"; exit 1
  fi
  sleep 600
done
echo "[sbc2d] 1-D FC SBC finished $(date)"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_MPS_PIPE_DIRECTORY=/CNSdata/mpaolett/mps_pipe
export CUDA_MPS_LOG_DIRECTORY=/CNSdata/mpaolett/mps_log
if ! pgrep -u "$(id -u)" -f nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo "[sbc2d] MPS daemon not running -- starting it"
  /CNSdata/mpaolett/mps_ctl.sh start
fi

run_stat () {
  STAT=$1
  COMMON="--L 100 --scale 0.5 --n_particles 4096 --which_stat $STAT \
          --infer_eta --eta_prior_scale 0.1 \
          --SC_type data --SC_size 10 --t_end 30000 --tr 1 --cut 10 --out_dir $OUT"
  echo "[sbc2d] === $STAT start $(date) ==="
  PIDS=""
  for k in 0 1 2 3; do
    S=$((k * 25)); E=$((S + 25))
    CUDA_VISIBLE_DEVICES=0 python3 sbc_run.py $COMMON --sampler smc_lik \
        --i_start $S --i_end $E > "$OUT/2d_${STAT}_lik_$k.log" 2>&1 &
    PIDS="$PIDS $!"
    CUDA_VISIBLE_DEVICES=1 python3 sbc_run.py $COMMON --sampler smc_abc \
        --i_start $S --i_end $E > "$OUT/2d_${STAT}_abc_$k.log" 2>&1 &
    PIDS="$PIDS $!"
  done
  echo "[sbc2d] $STAT launched 8 workers:$PIDS"
  for p in $PIDS; do wait $p; done
  # Required: the 4 shards of a sampler share one ranks filename and race during the
  # parallel phase, so this full-range pass is what writes the complete one.
  for S in smc_lik smc_abc; do
    python3 sbc_run.py $COMMON --sampler $S
  done
  echo "[sbc2d] === $STAT done $(date) ==="
}

# FCD first: it is the feature the 1-D calibration result used and the one the paper
# argues from. FC follows for completeness if there is time.
run_stat FCD
run_stat FC

echo "[sbc2d] finished $(date)"
