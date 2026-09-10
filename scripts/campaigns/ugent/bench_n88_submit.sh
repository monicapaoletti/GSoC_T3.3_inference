#!/usr/bin/env bash
# The sampler-comparison matrix at 88 nodes, one Slurm job per (sampler, statistic, G).
#
# Each job is a run_gpu_pool.py invocation with --gpu_concurrency 1, so the code path is
# byte-for-byte the one that produced the paper's 10-node matrix; only --SC_size differs.
# That is deliberate -- the point of the 88-node campaign is to change the network and
# nothing else, so the runner is reused rather than reimplemented for Slurm.
#
# WHY ONE JOB PER CELL AND NOT ONE POOL. run_gpu_pool.py exists to pack many cells onto a
# fixed pair of local GPUs; under Slurm that job is the scheduler's, and a single pool job
# would serialise the matrix behind one allocation. One cell per job also means a cell
# that dies takes only itself down, and re-running it is a one-line resubmit.
#
# WALLTIMES come from the measured 10-node runtimes scaled by the per-call ratio to 88
# nodes on an H100 (0.90x at batch 1 up to 2.02x at batch 4096). Each cell is a 4-point
# sweep over the batched axis (n_chains for MCMC, n_particles for SMC), which is why an
# mcmc cell is ~11 h and not ~3 h.
#
# SLICE IS ABSENT, DELIBERATELY. The 10-node matrix ran it at 50/100 because 1000/1000
# took 31.5 h there -- but 50/100 is not a comparison, it is a different question asked of
# a sampler that is being judged on where its chains get to, so the only budget worth
# reporting is rwmh's 1000/1000. At 88 nodes that projects to 63.7 h PER BATCH SIZE
# against a 71h59m cap: 255 GPU-h per G per statistic, 1019 GPU-h for four G on FCD alone,
# 2037 h on both -- three to five times the entire rest of the campaign, with every job
# sitting minutes from the cap. So slice is dropped rather than run at a budget that would
# not mean anything. That absence is a result about the sampler (it is eval-hungry enough
# to price itself out at whole-brain scale), not a hole in the matrix, and should be
# reported as one.
#
# Usage:  ./bench_n88_submit.sh FCD          # one statistic
#         ./bench_n88_submit.sh FC FCD       # both
#         DRY=1 ./bench_n88_submit.sh FCD    # print the sbatch lines, submit nothing
#         DEP=afterany:123:456 ./bench_n88_submit.sh FC   # hold until those finish
#
# Prints a final JOBIDS=<colon-separated> line so a caller can chain the next stage onto
# this one without having to scrape the per-job output.
set -euo pipefail

STATS=("${@:-FCD}")
GS=(0.2 0.33 0.5 0.7)
OUT="$VSC_DATA_VO_USER/results/bench_n88"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT/logs"

SUBMITTED=""
submit () {          # submit <name> <hours> <run_gpu_pool args...>
  local name="$1" hours="$2"; shift 2
  local dep=(); [ -n "${DEP:-}" ] && dep=(--dependency="$DEP")
  local cmd="source \$VSC_DATA_VO_USER/env.sh
cd \$GSOC_REPO/scripts/functions
python run_gpu_pool.py $* --SC_size 88 --gpu_only --gpu_concurrency 1 --results_dir $OUT"
  if [ -n "${DRY:-}" ]; then echo "[dry] ${name} (${hours}h): $*"; return; fi
  local id
  id=$(sbatch --parsable "${dep[@]}" --job-name="$name" --time="${hours}:00:00" \
         --ntasks=1 --gpus-per-node=1 --cpus-per-task=24 --mem=120G \
         --output="$OUT/logs/${name}-%j.out" --wrap="$cmd" | cut -d';' -f1)
  SUBMITTED="${SUBMITTED:+$SUBMITTED:}$id"
  echo "[submit] $id  $name"
}

for STAT in "${STATS[@]}"; do
  for G in "${GS[@]}"; do
    # MCMC: rwmh and demc at the converged 1000/1000 budget. No slice -- see above.
    submit "b88_rwmh_${STAT}_G${G}"  24 --suite mcmc --mcmc_samplers rwmh  --which_stat "$STAT" --G "$G" --n_warmup 1000 --n_samples 1000
    submit "b88_demc_${STAT}_G${G}"  24 --suite mcmc --mcmc_samplers demc  --which_stat "$STAT" --G "$G" --n_warmup 1000 --n_samples 1000
    # SMC: both flavours in one job (~1.8 h total), then tempered DE-MC as its own
    # sampler row -- --smc_move demc is what makes it a separate label, not a variant.
    submit "b88_smc_${STAT}_G${G}"      6 --suite smc --smc_samplers smc_lik smc_abc --which_stat "$STAT" --G "$G" --n_stages 50 --n_mcmc 5
    submit "b88_smcdemc_${STAT}_G${G}"  6 --suite smc --smc_samplers smc_abc --smc_move demc --which_stat "$STAT" --G "$G" --n_stages 50 --n_mcmc 5
  done
done
echo "[submit] out_dir: $OUT"
echo "JOBIDS=$SUBMITTED"
