#!/usr/bin/env bash
# Pull benchmark results from doslis (/CNSdata, which is NOT under the home mount)
# into the local repo, via a temporary sshfs mount that is always unmounted at the end.
# Usage:  bash scripts/pull_results.sh [DATE]     # DATE like 2026-07-27; default: all dates
#
# Copies benchmark_*.csv, summary_*.csv and *.png from the doslis results tree into the
# local results/ tree, so make_paper_assets.py can run locally. Read-only w.r.t. doslis.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MP="${HOME}/doslis_cns_mount"
REMOTE_RESULTS="GSoC_T3.3_inference/results"          # relative to /CNSdata/mpaolett
LOCAL_RESULTS="${REPO}/results"
DATE="${1:-}"

cleanup() { mountpoint -q "$MP" && fusermount3 -u "$MP" 2>/dev/null || true; }
trap cleanup EXIT

mkdir -p "$MP"
mountpoint -q "$MP" && fusermount3 -u "$MP"
echo "[pull] mounting doslis:/CNSdata/mpaolett ..."
sshfs doslis:/CNSdata/mpaolett "$MP"

SRC="$MP/$REMOTE_RESULTS"
[ -d "$SRC" ] || { echo "[pull] remote results dir not found: $SRC"; exit 1; }

dirs=()
if [ -n "$DATE" ]; then dirs=("$SRC/$DATE"); else dirs=("$SRC"/*/); fi
for d in "${dirs[@]}"; do
  [ -d "$d" ] || continue
  base="$(basename "$d")"
  dst="$LOCAL_RESULTS/$base"; mkdir -p "$dst"
  echo "[pull] $base"
  # only lightweight artifacts (CSVs + figures), never the big .nc / raw traces
  find "$d" -maxdepth 2 \( -name "benchmark_*.csv" -o -name "summary_*.csv" -o -name "*.png" \) \
    -exec cp -u {} "$dst/" \; 2>/dev/null || true
done

echo "[pull] done -> $LOCAL_RESULTS  (unmounting)"
