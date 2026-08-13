#!/usr/bin/env bash
# Same-code CPU baseline for the chain-vmapped MCMC samplers.
#
# The controlled CPU/GPU comparison -- identical JAX code, only the device changes --
# currently exists ONLY for the SMC family (smc_lik at nc 64/256, smc_abc at 64). For
# rwmh, demc and slice the only CPU evidence is PyMC, so their apparent CPU/GPU gap
# confounds device with framework. run_gpu_pool.mcmc_jobs already emits "one small-nc CPU
# baseline per sampler" (line 13-14) but every multi-G campaign passed --gpu_only to stop
# the slow CPU cells bottlenecking the matrix, so these were never run.
#
# nc=64 matches the narrowest GPU point of each sweep. The BUDGET is matched per sampler
# to that sampler's own GPU cell at nc=64, which is what makes each pair controlled:
#   rwmh, demc : 1000 warmup / 1000 draws
#   slice      :   50 warmup /  100 draws  (slice was swept at the reduced budget)
# Using one budget for all three would break the pairing it is meant to establish.
set -u
source /CNSdata/mpaolett/env.sh
cd /CNSdata/mpaolett/GSoC_T3.3_inference/scripts/functions || exit 1
OUT=/CNSdata/mpaolett/GSoC_T3.3_inference/results/$(date +%F)
mkdir -p "$OUT"

export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

COMMON="--grad_method fd --grad_horizon 100 --SC_type data --SC_size 10 --fast_bold \
        --cut 10 --tr 1 --t_end 30000 --seed 42 --G 0.2"

run () {  # sampler warmup draws stat
  echo "[cpu_mcmc] === $1 $4  nw=$2 ns=$3  $(date) ==="
  python3 mpr_jax_numpyro.py --sampler "$1" --n_chains 64 \
      --n_warmup "$2" --n_samples "$3" --which_stat "$4" \
      --save_dir "$OUT" $COMMON \
      || echo "[cpu_mcmc] FAILED $1 $4 (rc=$?)"
}

echo "[cpu_mcmc] start $(date)"
# rwmh and demc first: they carry the GPU numbers the text actually argues from, so if
# this is cut short the useful half is already on disk. slice is known impractical.
for STAT in FCD FC; do
  run rwmh  1000 1000 "$STAT"
  run demc  1000 1000 "$STAT"
done
for STAT in FCD FC; do
  run slice 50 100 "$STAT"
done
echo "[cpu_mcmc] finished $(date); results in $OUT"
