#!/usr/bin/env bash
# The gradient-based arm, at 88 nodes, on Leonardo -- NEW, never run at this scale before
# (the n=88 UGent campaign covered only the gradient-free suite: rwmh/demc/smc_*; see
# memory/n88-campaign-outcome and scripts/campaigns/ugent/bench_n88_submit.sh).
#
# WHY THIS IS A SEPARATE SCRIPT FROM bench_n88_submit.sh AND DOESN'T GO THROUGH
# run_gpu_pool.py --suite nuts. run_gpu_pool.py's nuts_jobs() (scripts/functions/
# run_gpu_pool.py:101) builds its mpr_jax_numpyro.py argv INTERNALLY and does not forward
# extra CLI flags from its own invocation, so there is no way to hand it --fc_eps from the
# outside. That matters here specifically: FCD's windowed corrcoef gradient is NaN at
# fc_eps=0 (the default) whenever a window has near-zero variance, and finite at
# fc_eps=1e-6 -- confirmed in memory/gradient-based-inference-mpr ("FCD DIVERGENCES
# DIAGNOSED"). So this script calls mpr_jax_numpyro.py directly, one job per
# (sampler, n_chains, G, stat) cell, and sets --fc_eps itself for FCD.
#
# GRADIENT METHOD: --grad_method fd (finite-difference; mpr_jax_numpyro.py defaults to
# this already, set explicitly here for the record). Do NOT switch to --grad_method
# autodiff for the main sweep: the autodiff gradient through this SDE explodes in the
# chaotic regime (G gtrsim 0.37, d<r>/dG ~1e12-1e26) and on the real FC/FCD statistics it
# is simply wrong even under truncated BPTT (grad_horizon), which separately was found to
# zero the gradient outright rather than tame it -- see memory/gradient-based-inference-
# mpr and memory/grad-horizon-zeroes-gradient. fd costs 2 forward sims per leapfrog step
# but is the only gradient that has been validated correct in this model.
#
# *** --time budgets below are UNCALIBRATED GUESSES -- there is no prior n=88 NUTS/fd
# measurement on ANY cluster to scale from (10-node NUTS runs exist but do not predict
# 88-node cost the way the MCMC/SMC budgets could be scaled from UGent's H100 numbers). ***
# Run probe_n88.sh's task 4 first (one nc=32 FCD replicate) and use ITS wall-clock,
# divided/multiplied by the real chain-count and tree-depth ratios, before trusting
# anything here. Until then, submit PILOT=1 first (one cell, G=0.5, nc=32, FCD) rather
# than the full sweep.
#
# Usage:  ./nuts_n88_submit.sh FCD              # one statistic, full G x nc sweep
#         ./nuts_n88_submit.sh FC FCD           # both
#         PILOT=1 ./nuts_n88_submit.sh FCD      # single calibration cell only
#         DRY=1 ./nuts_n88_submit.sh FCD        # print sbatch lines, submit nothing
#         DEP=afterany:123 ./nuts_n88_submit.sh FCD
#         INCLUDE_AUTODIFF_PROBE=1 ./nuts_n88_submit.sh FCD
#           # adds ONE extra nc=1 --grad_method autodiff cell per G/stat, purely to
#           # reconfirm the known explosion/vanishing-gradient result on this hardware --
#           # expected to fail, cheap (nc=1), off by default because it has no positive
#           # outcome, only a confirmatory one.
#
# JAX_MODULE=<name> runs under a CINECA module instead of the pinned jax_conda env (find
# candidates with probe_jax_modules.sh). With it set, PILOT also submits an nc=1 autodiff
# cell: the fd gradient does not depend on the JAX build, the autodiff one is what a
# different build could change. Expectation to test, not assume: the autodiff blow-up is
# the true derivative of a chaotic trajectory, so a newer JAX should NOT tame it. Module
# runs go to a separate directory and are never mixed with the pinned-env comparison.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

STATS=("${@:-FCD}")
GS=(0.2 0.33 0.5 0.7)
NCS=(1 8 32 128)
OUT="$LEO_RESULTS/nuts_n88_leonardo${JAX_MODULE:+_module_${JAX_MODULE//[\/ ]/_}}"
mkdir -p "$OUT/logs"
mapfile -t FLAGS < <(leo_sbatch_flags)

SUBMITTED=""
submit () {          # submit <name> <hours> <mpr_jax_numpyro.py args...>
  local name="$1" hours; hours=$(leo_hours "$2"); shift 2
  local dep=(); [ -n "${DEP:-}" ] && dep=(--dependency="$DEP")
  local cmd="$LEO_ENV_SETUP
python -c 'import jax; print(\"jax\", jax.__version__, jax.devices())'
cd $GSOC_REPO/scripts/functions
python mpr_jax_numpyro.py --sampler nuts --chain_method vectorized $* --SC_size 88 --save_dir $OUT"
  if [ -n "${DRY:-}" ]; then echo "[dry] ${name} (${hours}h): $*"; return; fi
  local id
  id=$(sbatch --parsable "${dep[@]}" "${FLAGS[@]}" --job-name="$name" --time="${hours}:00:00" \
         --output="$OUT/logs/${name}-%j.out" --wrap="$cmd" | cut -d';' -f1)
  SUBMITTED="${SUBMITTED:+$SUBMITTED:}$id"
  echo "[submit] $id  $name"
}

eps_flag () { [ "$1" = "FCD" ] && echo "--fc_eps 1e-6" || echo ""; }

if [ -n "${PILOT:-}" ]; then
  STAT="${STATS[0]}"
  submit "nutsL_pilot_${STAT}" 12 \
    --n_chains 32 --n_warmup 500 --n_samples 500 --grad_method fd \
    --which_stat "$STAT" --G 0.5 $(eps_flag "$STAT")
  if [ -n "${JAX_MODULE:-}" ]; then
    submit "nutsL_pilot_autodiff_${STAT}" 6 \
      --n_chains 1 --n_warmup 200 --n_samples 200 --grad_method autodiff \
      --which_stat "$STAT" --G 0.5 $(eps_flag "$STAT")
  fi
  echo "[submit] out_dir: $OUT"
  echo "JOBIDS=$SUBMITTED"
  exit 0
fi

for STAT in "${STATS[@]}"; do
  EPS=$(eps_flag "$STAT")
  for G in "${GS[@]}"; do
    for NC in "${NCS[@]}"; do
      submit "nutsL_fd_nc${NC}_${STAT}_G${G}" 24 \
        --n_chains "$NC" --n_warmup 500 --n_samples 500 --grad_method fd \
        --which_stat "$STAT" --G "$G" $EPS
    done
    if [ -n "${INCLUDE_AUTODIFF_PROBE:-}" ]; then
      submit "nutsL_autodiff_nc1_${STAT}_G${G}" 6 \
        --n_chains 1 --n_warmup 200 --n_samples 200 --grad_method autodiff \
        --which_stat "$STAT" --G "$G" $EPS
    fi
  done
done
echo "[submit] out_dir: $OUT"
echo "JOBIDS=$SUBMITTED"
