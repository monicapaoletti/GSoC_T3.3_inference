#!/bin/bash
# CPU benchmark sweep on ulysses (SISSA HPC). PRODUCTION sizes (1000 tune / 1000 draws).
# Matrix: {pymc,numpyro} x samplers x {FC,FCD} x grad_method  -> SLURM job array.
# Each array task = one (impl,sampler,stat,grad) cell = its OWN node, all run in parallel.
# Sizes are env-overridable (NW=tune, NS=draws) so a cheap validation can reuse this script.
# NOTE: ulysses is CPU-only; account is just the user (no --account/--partition=boost).
#SBATCH --job-name=mpr_bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --partition=long1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mpaolett@sissa.it
#SBATCH --array=0-35

set -uo pipefail

# Production sizes (override for validation: sbatch --export=ALL,NW=20,NS=20 ...)
NW=${NW:-1000}     # tune / warmup
NS=${NS:-1000}     # draws / samples
GVAL=${GVAL:-0.2}  # true G for the synthetic observation (0.2 = calmer regime than 0.33)
TEND=${TEND:-30000}
NNODES=${NNODES:-10} # data SC cut to NNODES x NNODES (10 keeps forward ~0.88s/eval w/ fast_bold)
CUT=${CUT:-10}       # transient BOLD frames dropped; 10 (not 30) keeps FCD non-degenerate
                     # at t_end=30000: cut=10 -> 80 frames, 51 windows, 231 FCD features (k=30);
                     # cut=30 -> only 1 FCD feature. Pinned same for pymc & numpyro fairness.
TR=${TR:-1}          # numpyro ONLY subsamples BOLD by tr (bold[cut::tr]); tr=1 -> keep all 80
                     # frames to match pymc (which ignores tr). tr=5 -> only 16 frames, FCD invalid.

# --- task matrix: "IMPL SAMPLER WHICH_STAT GRAD" (GRAD=none for gradient-free) ---
TASKS=(
  # pymc gradient-free (12)
  "pymc slice FC none"          "pymc slice FCD none"
  "pymc metropolis FC none"     "pymc metropolis FCD none"
  "pymc demetropolisz FC none"  "pymc demetropolisz FCD none"
  "pymc demetropolis FC none"   "pymc demetropolis FCD none"
  "pymc smclik FC none"         "pymc smclik FCD none"
  "pymc smcabc FC none"         "pymc smcabc FCD none"
  # pymc gradient-based x {fd,autodiff} (12)
  "pymc nuts FC fd"        "pymc nuts FCD fd"        "pymc nuts FC autodiff"     "pymc nuts FCD autodiff"
  "pymc blackjax FC fd"    "pymc blackjax FCD fd"    "pymc blackjax FC autodiff" "pymc blackjax FCD autodiff"
  "pymc numpyro FC fd"     "pymc numpyro FCD fd"     "pymc numpyro FC autodiff"  "pymc numpyro FCD autodiff"
  # numpyro impl gradient-based x {fd,autodiff} (12)
  "numpyro nuts FC fd"       "numpyro nuts FCD fd"       "numpyro nuts FC autodiff"       "numpyro nuts FCD autodiff"
  "numpyro pathfinder FC fd" "numpyro pathfinder FCD fd" "numpyro pathfinder FC autodiff" "numpyro pathfinder FCD autodiff"
  "numpyro blackjax FC fd"   "numpyro blackjax FCD fd"   "numpyro blackjax FC autodiff"   "numpyro blackjax FCD autodiff"
)

read -r IMPL SAMPLER STAT GRAD <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"

DATE=$(date +%Y-%m-%d)
REPO=$HOME/GSoC_T3.3_inference
RESULTS_DIR=$REPO/results/$DATE
OUT_DIR=$RESULTS_DIR/out
mkdir -p "$OUT_DIR"
LOG=$OUT_DIR/bench_${IMPL}_${SAMPLER}_${STAT}_${GRAD}

exec > >(tee -a "${LOG}.out") 2> >(tee -a "${LOG}.err" >&2)
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID impl=$IMPL sampler=$SAMPLER stat=$STAT grad=$GRAD start=$(date)"

# --- environment (built with micromamba, see ulysses_env_and_slurm memory) ---
module load openblas 2>/dev/null || true
export MAMBA_ROOT_PREFIX=$HOME/micromamba
eval "$("$HOME/bin/micromamba" shell hook -s bash)"
micromamba activate jax_conda
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

cd "$REPO/scripts/functions"

# --- smoke-test sizes (PROVISIONAL: validated on one cell before full submit) ---
GRAD_HORIZON=100   # truncated-BPTT for autodiff stability; ignored by fd/gradient-free

if [ "$IMPL" = "pymc" ]; then
  GRAD_ARG=""
  [ "$GRAD" != "none" ] && GRAD_ARG="--grad_method $GRAD --grad_horizon $GRAD_HORIZON"
  PCHAINS=2
  [ "$SAMPLER" = "demetropolis" ] && PCHAINS=4  # DEMetropolis is a population sampler: needs >=3 chains
  srun --unbuffered python3 mpr_jax_pymc.py \
      --sampler "$SAMPLER" --which_stat "$STAT" $GRAD_ARG \
      --G "$GVAL" --SC_type data --SC_size "$NNODES" --fast_bold --cut "$CUT" \
      --t_end "$TEND" --n_warmup "$NW" --n_samples "$NS" --n_chains "$PCHAINS" \
      --sample_cores 2 --n_prior 1000 --seed 42
else
  srun --unbuffered python3 mpr_jax_numpyro.py \
      --sampler "$SAMPLER" --which_stat "$STAT" --grad_method "$GRAD" --grad_horizon $GRAD_HORIZON \
      --G "$GVAL" --SC_type data --SC_size "$NNODES" --fast_bold --cut "$CUT" --tr "$TR" \
      --t_end "$TEND" --n_warmup "$NW" --n_samples "$NS" --n_chains 1 \
      --save_dir "$RESULTS_DIR" --seed 42
fi

echo "task=$SLURM_ARRAY_TASK_ID exit=$? end=$(date)"
