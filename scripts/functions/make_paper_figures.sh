#!/usr/bin/env bash
# Regenerate EVERY figure the manuscript \includegraphics, from code.
#
# Written after an audit found that four of the seven paper figures could not be traced
# to a generator by filename: two are built from f-string tags (so the literal name never
# appears in the source), and the two eta sweeps had been copied and RENAMED by hand, so
# no command reproduced the file the paper actually inputs. This script is the missing
# mapping: figure in the paper <- exact command that produces it.
#
# Usage:  bash make_paper_figures.sh [--slow]
#   default : regenerate only the figures that read stored CSV/NPZ measurements (seconds)
#   --slow  : ALSO re-run the simulations behind Figs. 1 and the eta sweeps (~1 h, CPU)
#
# The simulation-backed figures are gated because they re-integrate the SDE; the stored
# measurements they would overwrite are the ones the manuscript already cites.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
REPO="$(cd ../.. && pwd)"
PAPER="$REPO/paper"
FIGS="$PAPER/figures"
TODAY="$REPO/results/$(date +%F)"
PY="${PY:-/home/monica/anaconda3/envs/myenv/bin/python}"
SLOW=0
[ "${1:-}" = "--slow" ] && SLOW=1
mkdir -p "$FIGS" "$TODAY"

echo "== 1. benchmark figures (recovery, accuracy-vs-cost, calibration) + all tables =="
# Writes figures/recovery_vs_G.png, figures/accuracy_vs_cost.png,
# figures/calibration_sd_vs_err.png and every tables/*.tex, all from master_results.csv.
"$PY" make_paper_assets.py --results "$REPO/results" --out "$PAPER" \
      --table_label tab:accuracy
# make_paper_assets writes straight into the paper; mirror its outputs into the dated
# results folder too, so every figure the manuscript uses also has a copy filed under
# the day it was produced -- the same convention steps 2-4 follow.
for f in recovery_vs_G accuracy_vs_cost calibration_sd_vs_err sbc_ranks throughput_ess; do
  [ -f "$FIGS/$f.png" ] && cp "$FIGS/$f.png" "$TODAY/"
done
[ -f "$PAPER/tables/sbc_table.tex" ] && cp "$PAPER/tables/sbc_table.tex" "$TODAY/"

echo "== 2. forward throughput (Fig. 2) + its companion table =="
# Reads the stored measurement from the day it was MEASURED; writes into today's folder
# so an old dated folder is never rewritten, then copies the figure into the paper.
"$PY" plot_forward_throughput.py --outdir "$REPO/results/2026-07-31" --out "$TODAY"
cp "$TODAY/forward_throughput.png" "$FIGS/"
cp "$TODAY/forward_throughput_table.tex" "$PAPER/tables/"

echo "== 3. ESS scaling (Fig. 4) =="
"$PY" plot_ess_scaling.py --master "$PAPER/master_results.csv" --out "$TODAY" \
      --name ess_scaling.png
cp "$TODAY/ess_scaling.png" "$FIGS/"

echo "== 4. simulation runtime (Fig. 3) =="
"$PY" plot_sim_runtime.py --npz "$REPO/results/2026-07-31/sim_benchmark/benchmarking_GPU_tend30000.png.npz" \
      --t_end 30000 --out "$TODAY"
cp "$TODAY/sim_runtime_tend30000.png" "$FIGS/"

if [ "$SLOW" -eq 1 ]; then
  echo "== 5. model dynamics (Fig. 1) + its companion table  [SLOW: re-simulates] =="
  "$PY" plot_model_dynamics.py --out "$TODAY"          # 88 nodes, eta=-4.6, G sweep
  cp "$TODAY/model_dynamics_sweepG_tend300000.png" "$FIGS/"
  cp "$TODAY/model_dynamics_sweepG_tend300000_stats.tex" \
     "$PAPER/tables/model_dynamics_stats.tex"

  echo "== 6. eta sweeps (supplementary figures)  [SLOW: re-simulates] =="
  # The paper inputs these under SHORT names; the generator emits tag-based ones. This
  # copy is the rename the audit found undocumented -- keep the two in step.
  ETAS="-6.0 -5.5 -5.0 -4.6 -4.2 -3.8"
  "$PY" plot_model_dynamics.py --G 0.5 --eta $ETAS --out "$TODAY"
  cp "$TODAY/model_dynamics_sweepeta_tend300000.png" "$FIGS/eta_sweep_G0.5_SC88.png"
  "$PY" plot_model_dynamics.py --G 0.5 --eta $ETAS --SC_size 10 --out "$TODAY"
  cp "$TODAY/model_dynamics_sweepeta_tend300000_SC10.png" "$FIGS/eta_sweep_G0.5_SC10.png"
else
  echo "== 5-6. model dynamics + eta sweeps SKIPPED (pass --slow to re-simulate) =="
fi

echo
echo "== audit: every figure the paper inputs, and whether it now exists =="
grep -oE "figures/[a-zA-Z0-9_.]+\.png" "$PAPER/main.tex" | sort -u | while read -r f; do
  if [ -f "$PAPER/$f" ]; then echo "  OK      $f"; else echo "  MISSING $f"; fi
done
