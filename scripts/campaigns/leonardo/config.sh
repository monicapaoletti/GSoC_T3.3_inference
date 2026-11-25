#!/usr/bin/env bash
# Shared settings for every Leonardo driver in this directory. Sourced, not run.
#
# Everything cluster-specific lives here so it is set once. The account is the EuroHPC
# project EUHPC_B40_021 -- NOT Sis25_piasini, which the old scripts hardcoded. Override
# only if hours move to another project; `saldo -b` on a login node lists the accounts
# and their remaining hours.
#
#   export LEO_ACCOUNT=...                  # default EUHPC_B40_021
#   export LEO_PARTITION=boost_usr_prod     # default; check `sinfo -s`
#   export LEO_QOS=...                      # optional, e.g. a longer-walltime QOS
#   export LEO_MAXH=24                      # walltime cap of that partition/QOS, hours

LEO_ACCOUNT="${LEO_ACCOUNT:-EUHPC_B40_021}"
LEO_PARTITION="${LEO_PARTITION:-boost_usr_prod}"
LEO_QOS="${LEO_QOS:-}"
LEO_MAXH="${LEO_MAXH:-24}"
LEO_GPU_MEM="${LEO_GPU_MEM:-120G}"     # host RAM per 1-GPU job

LEO_HOME="${LEO_HOME:-/leonardo/home/userexternal/mpaolett}"
GSOC_REPO="${GSOC_REPO:-$LEO_HOME/GSoC_T3.3_inference}"
# Code runs from the $HOME clone (the one wired to GitHub). Results go to a personal
# folder in the project's $WORK, which has the space $HOME lacks. $WORK is shared with
# the other EUHPC_B40_021 members -- and $WORK/GSoC_T3.3_inference is SOMEONE ELSE'S
# clone (git reports dubious ownership), so never write into or run from it.
LEO_RESULTS="${LEO_RESULTS:-${WORK:+$WORK/mpaolett/results}}"
LEO_RESULTS="${LEO_RESULTS:-$LEO_HOME/results}"     # fallback if $WORK is unset

# The pinned environment `jax_n88` (built by setup_env.sh): the UGent package versions, which is what makes
# the Leonardo numbers comparable to the UGent ones. JAX_MODULE swaps in a CINECA module
# instead -- for the NUTS/module experiment only, never for the comparison runs.
#
# The activation runs with `set -u` suspended: jax_conda's own activate.d hook
# (activate-binutils_linux-64.sh line 68) reads $ADDR2LINE unset, and under the drivers'
# `set -u` that killed every job in 3 s (first probe, 2026-11-25). The caller's -u state
# is restored afterwards. JAX_MODULE may be "profile/deeplrn cineca-ai/4.3.0": cineca-ai
# is only visible once its profile is loaded.
_LEO_SET_U='case $- in *u*) _leo_u=1;; *) _leo_u=;; esac; set +u'
_LEO_RESTORE_U='if [ -n "$_leo_u" ]; then set -u; fi'
if [ -n "${JAX_MODULE:-}" ]; then
  LEO_ENV_SETUP="$_LEO_SET_U
module load $JAX_MODULE
$_LEO_RESTORE_U"
else
  # unset PYTHONPATH: the default-loaded python/3.11.7 module sets it to its own 3.11
  # site-packages, which every job inherits and which jax_n88 (Python 3.10) must not see.
  LEO_ENV_SETUP="$_LEO_SET_U
unset PYTHONPATH
module load openblas
eval \"\$(micromamba shell hook --shell=bash)\"
micromamba activate ${LEO_CONDA_ENV:-jax_n88}
$_LEO_RESTORE_U"
fi

# Common sbatch flags for a one-GPU job. --qos only when set.
leo_sbatch_flags () {
  local f=(--account="$LEO_ACCOUNT" --partition="$LEO_PARTITION"
           --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem="$LEO_GPU_MEM")
  [ -n "$LEO_QOS" ] && f+=(--qos="$LEO_QOS")
  printf '%s\n' "${f[@]}"
}

# Clamp a requested walltime (hours) to the cap, and say so: a silently clamped budget is
# how a cell dies at the limit with no warning.
leo_hours () {
  local h="$1"
  if [ "$h" -gt "$LEO_MAXH" ]; then
    echo "[warn] requested ${h}h > cap ${LEO_MAXH}h, clamped" >&2; h="$LEO_MAXH"
  fi
  echo "$h"
}
