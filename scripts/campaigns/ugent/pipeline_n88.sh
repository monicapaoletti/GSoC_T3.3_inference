#!/usr/bin/env bash
# Chain the whole 88-node campaign so it runs unattended, in the order the results are
# wanted, with no stage competing with another for GPUs.
#
# WHY DEPENDENCIES AND NOT HOLDS. A held job needs someone to release it; a dependency
# does not. Every stage below starts itself when the previous one finishes, so the
# campaign survives nobody logging in for two days.
#
# WHY SERIAL AND NOT ALL AT ONCE. Slurm gives each job its own GPU -- jobs never share a
# card, so concurrency would not make any single job slower. But litleo is contended and
# we only get ~8-10 of its 16 GPUs, so parallel stages would split that pool and every
# stage would finish later. Total throughput is identical; serialising just decides which
# results exist first. FCD before FC is Monica's ordering.
#
# ORDER
#   1. SBC 1-D FCD          (already running: array $SBC1D)
#   2. benchmark FCD        (already submitted, constrained here to wait for stage 1)
#   3. SBC 2-D (G, eta) FCD (submitted here, waits for stage 2)
#   4. benchmark FC         (submitted here, waits for stage 3)
#
# Stage 3 lands ~12 h after stage 1 finishes, which is deliberate: it is the last point at
# which the 1-D ranks can be inspected and the 92 GPU-h 2-D run cancelled if 88 nodes
# turns out to be broken in some way the probe could not see.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBC1D=43112770                       # the running 1-D FCD array
BENCH_FCD="43112781 43112782 43112783 43112784 43112785 43112786 43112787 43112788 \
           43112789 43112790 43112791 43112792 43112793 43112794 43112795 43112796"

module swap cluster/litleo >/dev/null 2>&1

# litleo's controller was refusing RPCs when this was written; jobs kept running, but
# scontrol/sbatch could not reach it. Wait it out rather than failing the campaign.
echo "[pipeline] waiting for the litleo controller to answer..."
until squeue -M litleo -u "$USER" -h -o "%i" >/dev/null 2>&1; do sleep 60; done
echo "[pipeline] controller up $(date)"

# Two copies of this script run on purpose -- one nohup'd on the login node, one as a
# doduo job -- so that neither a login-node restart nor a doduo queue delay can leave the
# chain unbuilt. They race here, AFTER the wait, so both are still live candidates; mkdir
# is atomic, so exactly one proceeds and the other exits without submitting anything.
# Taking the lock before the wait would defeat the redundancy: the winner would hold it
# while blocked, and the loser would exit while the controller was still down.
LOCK="$VSC_DATA_VO_USER/.pipeline_n88.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "[pipeline] another instance already built the chain ($LOCK); nothing to do"
  exit 0
fi
echo "[pipeline] lock acquired by $(hostname)"

# --- stage 2: make the FCD benchmark wait for the 1-D SBC array ---
DEPLIST=""
for J in $BENCH_FCD; do
  scontrol update JobId="$J" Dependency="afterany:${SBC1D}" 2>&1 | head -1
  DEPLIST="${DEPLIST:+$DEPLIST:}$J"
done
echo "[pipeline] benchmark FCD now waits on $SBC1D"

# --- stage 3: 2-D (G, eta) SBC on FCD, after the FCD benchmark ---
OUT3=$(DEP="afterany:${DEPLIST}" "$HERE/sbc_n88_submit.sh" FCD 2d 16 | tee /dev/stderr \
        | grep '^JOBIDS=' | cut -d= -f2)
echo "[pipeline] SBC 2-D FCD -> $OUT3"

# --- stage 4: FC benchmark, after the 2-D SBC ---
OUT4=$(DEP="afterany:${OUT3}" "$HERE/bench_n88_submit.sh" FC | tee /dev/stderr \
        | grep '^JOBIDS=' | cut -d= -f2)
echo "[pipeline] benchmark FC -> $OUT4"

echo "[pipeline] chain complete $(date); nothing further needs a human"
