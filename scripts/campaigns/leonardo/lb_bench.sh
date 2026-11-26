#!/usr/bin/env bash
# Larter-Breakspear: van der Vlag's CUDA kernel vs lb_jax.py, on one A100.
# See scripts/functions/bench_lb_cuda_vs_jax.py for what each test measures.
#
# ONE-TIME SETUP, on a login node (adds CuPy to jax_n88 without touching its pins):
#   micromamba activate jax_n88
#   python -m pip install --no-deps cupy-cuda12x==13.3.0 fastrlock==0.8.2
#
# Submit:
#   LB_JAX_DIR=<dir holding your lb_jax.py> \
#     sbatch --account=EUHPC_B40_021 --export=ALL lb_bench.sh
# Optional: KERNEL=<path to .c>, BENCH_ARGS="--skip T3 --batches 1 64 1024"
# Quick look instead (one simulation per code, plotted; a few minutes):
#   QUICK=1 LB_JAX_DIR=<dir> sbatch --account=EUHPC_B40_021 --time=00:20:00 --export=ALL lb_bench.sh
#   optional QUICK_ARGS="--t_ms 5000 --G 0.3 --own_noise"
#
# The kernel file is only READ; the noise-injecting variant is built in memory.
#SBATCH --job-name=lb_bench
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
set -uo pipefail

GSOC_REPO="${GSOC_REPO:-/leonardo/home/userexternal/mpaolett/GSoC_T3.3_inference}"
source "$GSOC_REPO/scripts/campaigns/leonardo/config.sh"
: "${LB_JAX_DIR:?set LB_JAX_DIR to the directory holding the lb_jax.py to test}"
KERNEL="${KERNEL:-/leonardo_work/EUHPC_B40_021/mvanderv/tvbSBI/tvbgpu/models/larter_breakspear2.c}"
OUT="$LEO_RESULTS/lb_bench_$(date +%Y%m%d_%H%M)"

set +u
module load cuda/12.2                 # nvcc, which CuPy's nvcc backend calls
set -u
export CUDA_PATH="$(dirname "$(dirname "$(command -v nvcc)")")"
eval "$LEO_ENV_SETUP"

echo "[lb_bench] $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "[lb_bench] kernel : $KERNEL  (md5 $(md5sum "$KERNEL" | cut -c1-12))"
echo "[lb_bench] lb_jax : $LB_JAX_DIR/lb_jax.py  (md5 $(md5sum "$LB_JAX_DIR/lb_jax.py" | cut -c1-12))"

cd "$GSOC_REPO/scripts/functions" || exit 1
if [ -n "${QUICK:-}" ]; then
  mkdir -p "$OUT"
  # shellcheck disable=SC2086
  python lb_quick_compare.py --kernel "$KERNEL" --lb_jax_dir "$LB_JAX_DIR" \
    --out "$OUT/lb_quick.png" ${QUICK_ARGS:-}
  echo "[lb_bench] quick rc=$? out=$OUT/lb_quick.png $(date)"
  exit 0
fi
# shellcheck disable=SC2086  # BENCH_ARGS is a deliberately word-split flag list
python bench_lb_cuda_vs_jax.py --kernel "$KERNEL" --lb_jax_dir "$LB_JAX_DIR" \
  --out "$OUT" ${BENCH_ARGS:-}
echo "[lb_bench] rc=$? out=$OUT $(date)"
