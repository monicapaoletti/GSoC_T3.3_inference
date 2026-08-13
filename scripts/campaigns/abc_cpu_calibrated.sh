#!/usr/bin/env bash
# Calibrated ABC-SMC on CPU: the missing half of a controlled CPU/GPU comparison.
#
# As it stands the ABC CPU-vs-GPU contrast in the paper differs in FOUR ways at once:
#   CPU cell = pymc,  batch 1000, epsilon=10 (uncalibrated), host
#   GPU cell = numpyro, batch 4096, eps_target=obs_err (calibrated), device
# so it cannot separate "GPU helps" from "the CPU run used a 10x too-loose epsilon".
#
# These runs are the same JAX code, same calibrated epsilon, same flags as the GPU
# sweep (COMMON in run_gpu_pool.py) -- only the backend differs. Batch 64 is chosen
# because the GPU sweep already has a batch-64 cell for every (G, stat), giving a
# MATCHED pair; batch 4096 on CPU would be ~63 h per cell (Table 3: CPU throughput
# collapses past B=64), which is why the CPU arm was originally skipped.
set -u
source /CNSdata/mpaolett/env.sh
cd /CNSdata/mpaolett/GSoC_T3.3_inference/scripts/functions || exit 1
OUT=/CNSdata/mpaolett/GSoC_T3.3_inference/results/$(date +%F)
mkdir -p "$OUT"

export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

COMMON="--grad_method fd --grad_horizon 100 --SC_type data --SC_size 10 --fast_bold \
        --cut 10 --tr 1 --t_end 30000 --seed 42"

echo "[abc_cpu] start $(date)"
for STAT in FC FCD; do
  for G in 0.2 0.33 0.5 0.7; do
    echo "[abc_cpu] === G=$G $STAT $(date) ==="
    python3 mpr_jax_numpyro.py --sampler smc_abc --n_particles 64 \
        --n_stages 50 --n_mcmc 5 --which_stat "$STAT" --G "$G" \
        --save_dir "$OUT" $COMMON \
        || echo "[abc_cpu] FAILED G=$G $STAT (rc=$?)"
  done
done
echo "[abc_cpu] finished $(date); results in $OUT"
