#!/usr/bin/env bash
# Consolidated "finish the GPU matrix" campaign. Waits for the FC mcmc campaign to
# free the GPUs, then runs EVERYTHING remaining at high concurrency (~3 jobs/GPU,
# since each job uses ~250 MiB):
#   - demc FCD (all G)                       [DE-MC converges at 200/500]
#   - slice FCD (G=0.2)                       [eval-heavy -> one G]
#   - rwmh converged FC+FCD (all G, 1000/1000)[proper warm-up -> fair Metropolis number]
#   - ABC fixed-epsilon FC+FCD (all G, full particle sweep) [eps=obs_err, calibrated]
# All outputs pinned to one results folder (no midnight split).
source /CNSdata/mpaolett/env.sh   # activate the GPU venv (jax/numpyro/...); MUST come first
set -uo pipefail
FUNCS=/CNSdata/mpaolett/GSoC_T3.3_inference/scripts/functions
LOG=/CNSdata/mpaolett/pool_gpu_final.log
OUT=/CNSdata/mpaolett/GSoC_T3.3_inference/results/final_matrix
CONC="--gpu_only --gpu_concurrency 3 --results_dir $OUT"
cd "$FUNCS"

run(){ python3 run_gpu_pool.py "$@" $CONC >> "$LOG" 2>&1; }

echo "[wait] for mcmc campaign to finish... $(date)" | tee -a "$LOG"
while tmux has-session -t mcmc 2>/dev/null; do sleep 60; done
echo "[go] GPUs free, starting consolidated matrix $(date)" | tee -a "$LOG"

# 1) DE-MC on FCD, all G
for G in 0.2 0.33 0.5 0.7; do
  run --suite mcmc --mcmc_samplers demc --which_stat FCD --G "$G" --n_warmup 200 --n_samples 500
done
# 2) slice on FCD, G=0.2 only
run --suite mcmc --mcmc_samplers slice --which_stat FCD --G 0.2 --n_warmup 15 --n_samples 30
# 3) rwmh CONVERGED, FC+FCD, all G (1000 tune / 1000 draws)
for STAT in FC FCD; do for G in 0.2 0.33 0.5 0.7; do
  run --suite mcmc --mcmc_samplers rwmh --which_stat "$STAT" --G "$G" --n_warmup 1000 --n_samples 1000
done; done
# 4) ABC fixed-epsilon, FC+FCD, all G, full particle sweep
for STAT in FC FCD; do for G in 0.2 0.33 0.5 0.7; do
  run --suite smc --smc_samplers smc_abc --which_stat "$STAT" --G "$G" --n_stages 50 --n_mcmc 5
done; done

echo "[done] consolidated GPU matrix finished $(date)" | tee -a "$LOG"
