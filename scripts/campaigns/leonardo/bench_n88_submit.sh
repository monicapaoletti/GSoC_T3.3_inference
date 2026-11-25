#!/usr/bin/env bash
# The n=88 sampler-comparison matrix on Leonardo Booster -- the same matrix the UGent
# campaign ran (scripts/campaigns/ugent/bench_n88_submit.sh, results pulled to
# results/2026-09-24_paper_assets_n88/), with ONE deliberate change and one control.
#
# THE CHANGE: SMC cells run with --adapt_rw. At UGent the SMC kernel used the fixed
# rw_step=0.1 and the FC posteriors collapsed (G_sd exactly 0 in most cells, truth >3 sd
# away in 43 of 48). Those numbers cannot go in the paper, so this run fixes the kernel.
#
# THE CONTROL (EXACT=1, default): smc_lik + smc_abc at G=0.5 on FC and FCD, re-run with
# the UGent fixed step, into a separate directory. That cell is the cross-cluster
# reproducibility check for the SMC arm. The MCMC cells (rwmh, demc) need no separate
# control: --adapt_rw does not touch them, so every MCMC cell here IS an exact replica of
# its UGent counterpart and compares one-to-one.
#
# ONE JOB PER (sampler, stat, G, batch) FOR MCMC. A UGent MCMC cell was a 4-point sweep
# over n_chains taking ~11 h on an H100 NVL; on an A100 that may not fit a 24 h cap. One
# point per job (~3.4 h on H100 at nc4096) leaves room for a slower card, and a lost point
# is a one-job resubmit. SMC cells stay as 4-point sweeps (~1.8 h on H100).
#
# Walltimes set from the Leonardo probe (2026-11-25): A100/H100 = 1.20-1.37 over batches
# 64-4096, applied to UGent's recorded run times. Longest predicted job: MCMC point 4.6 h
# (7 h asked), SMC cell 3.6 h (5 h), SMC-DEMC cell 1.8 h (3 h).
#
# SLICE IS ABSENT, as at UGent: at rwmh's 1000/1000 budget it costs ~64 h per batch size
# at 88 nodes. Report its absence as a result about the sampler.
#
# Usage:  (account EUHPC_B40_021 is the default, see config.sh)
#         ./bench_n88_submit.sh FC FCD      # both statistics (the full matrix)
#         DRY=1 ./bench_n88_submit.sh FC FCD
#         EXACT=0 ./bench_n88_submit.sh FCD # skip the fixed-step control
#         ONLY=smc ./bench_n88_submit.sh FC # just the SMC arm (or ONLY=mcmc)
#         DEP=afterany:123 ./bench_n88_submit.sh FC
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

if [ $# -eq 0 ]; then STATS=(FC FCD); else STATS=("$@"); fi
GS=(0.2 0.33 0.5 0.7)
BATCHES=(64 256 1024 4096)
OUT="$LEO_RESULTS/bench_n88_leonardo"            # adaptive SMC + all MCMC
OUT_EXACT="$LEO_RESULTS/bench_n88_leonardo_fixedstep"
mkdir -p "$OUT/logs" "$OUT_EXACT/logs"
mapfile -t FLAGS < <(leo_sbatch_flags)

SUBMITTED=""
submit () {          # submit <name> <hours> <outdir> <run_gpu_pool args...>
  local name="$1" hours; hours=$(leo_hours "$2"); local out="$3"; shift 3
  local dep=(); [ -n "${DEP:-}" ] && dep=(--dependency="$DEP")
  local cmd="$LEO_ENV_SETUP
cd $GSOC_REPO/scripts/functions
python run_gpu_pool.py $* --SC_size 88 --gpu_only --gpu_concurrency 1 --results_dir $out"
  if [ -n "${DRY:-}" ]; then echo "[dry] ${name} (${hours}h) -> ${out##*/}: $*"; return; fi
  local id
  id=$(sbatch --parsable "${dep[@]}" "${FLAGS[@]}" --job-name="$name" \
         --time="${hours}:00:00" --output="$out/logs/${name}-%j.out" \
         --wrap="$cmd" | cut -d';' -f1)
  SUBMITTED="${SUBMITTED:+$SUBMITTED:}$id"
  echo "[submit] $id  $name"
}

for STAT in "${STATS[@]}"; do
  for G in "${GS[@]}"; do
    if [ "${ONLY:-}" != "smc" ]; then
      for S in rwmh demc; do
        for B in "${BATCHES[@]}"; do
          submit "b88L_${S}_${STAT}_G${G}_nc${B}" 7 "$OUT" \
            --suite mcmc --mcmc_samplers "$S" --batch_sizes "$B" \
            --which_stat "$STAT" --G "$G" --n_warmup 1000 --n_samples 1000
        done
      done
    fi
    if [ "${ONLY:-}" != "mcmc" ]; then
      submit "b88L_smc_${STAT}_G${G}"     5 "$OUT" --suite smc --smc_samplers smc_lik smc_abc \
        --adapt_rw --which_stat "$STAT" --G "$G" --n_stages 50 --n_mcmc 5
      submit "b88L_smcdemc_${STAT}_G${G}" 3 "$OUT" --suite smc --smc_samplers smc_abc \
        --smc_move demc --adapt_rw --which_stat "$STAT" --G "$G" --n_stages 50 --n_mcmc 5
    fi
  done
  # Fixed-step control: same flags as UGent, no --adapt_rw, separate directory.
  if [ "${EXACT:-1}" = "1" ] && [ "${ONLY:-}" != "mcmc" ]; then
    submit "b88L_smcFIXED_${STAT}_G0.5" 5 "$OUT_EXACT" --suite smc \
      --smc_samplers smc_lik smc_abc --which_stat "$STAT" --G 0.5 --n_stages 50 --n_mcmc 5
  fi
done
echo "[submit] out_dir: $OUT  (control: $OUT_EXACT)"
echo "JOBIDS=$SUBMITTED"
