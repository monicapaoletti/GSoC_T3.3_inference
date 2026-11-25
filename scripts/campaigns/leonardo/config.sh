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
# Results default to $HOME next to the repo. If $HOME quota is tight, point this at
# $WORK or $SCRATCH (scratch is purged, so copy results off before it is).
LEO_RESULTS="${LEO_RESULTS:-$LEO_HOME/results}"

# The pinned environment: identical package versions to doslis/UGent, which is what makes
# the Leonardo numbers comparable to the UGent ones. JAX_MODULE swaps in a CINECA module
# instead -- for the NUTS/module experiment only, never for the comparison runs.
if [ -n "${JAX_MODULE:-}" ]; then
  LEO_ENV_SETUP="module load $JAX_MODULE"
else
  LEO_ENV_SETUP='module load openblas
eval "$(micromamba shell hook --shell=bash)"
micromamba activate jax_conda'
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
