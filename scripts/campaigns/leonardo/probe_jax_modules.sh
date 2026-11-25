#!/usr/bin/env bash
# Module recon for Leonardo, run ONCE before anything else in this directory.
#
# WHY THIS EXISTS. Every other script here assumes the micromamba env `jax_conda`
# (the one scripts/functions/benchmarking_simuations_gpu.sh already uses on this
# cluster). CINECA also ships its own curated module stack for AI/deep-learning work
# (commonly surfaced via a `profile/deeplrn`-style module that unlocks modules named
# things like `python/<ver>--gcc--<ver>`, `cineca-ai`, or framework-specific modules).
# If one of those already carries a CUDA-12 JAX build, it could be faster to point at
# than micromamba's own resolve -- but NONE of the module names below are verified
# against a live `module avail` (no cluster access when this was written: 2FA gate,
# nobody authenticated). Treat every candidate as a guess to be falsified, not a known
# answer, and do not wire any other script to one of these module names until this
# probe has actually printed a working `jax.devices()` under it.
#
# This is diagnostic only -- it changes nothing about how bench/nuts submission
# actually runs. The pinned stack (micromamba jax_conda here, or the Apptainer
# `science` image from container/build_sif.sh, built the same way for doslis/UGent)
# stays the default specifically so results are comparable across clusters; see
# container/HOWTO.md. Switching the campaign to a module-provided JAX is a decision
# to make AFTER reading this probe's output, not before.
#
#SBATCH --job-name=probe_jax_mod
# account: pass it on the command line -- sbatch --account=EUHPC_B40_021 <this script>
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.out
set -uo pipefail

echo "=== module system ==="
echo "[probe] hostname $(hostname), date $(date)"
module avail 2>&1 | tee /tmp/module_avail.$$ >/dev/null

echo
echo "=== candidate AI/deep-learning profiles ==="
grep -iE "profile|deeplrn|^ *ai[/-]|cineca-ai" /tmp/module_avail.$$ || echo "(none matched)"

echo
echo "=== candidate jax/python/cuda modules (pre-profile) ==="
grep -iE "jax|cuda|python/|anaconda" /tmp/module_avail.$$ || echo "(none matched)"

# A profile module, if one exists, usually has to be loaded before its modules show up
# at all -- so re-list after trying the most likely name. This is the part most likely
# to need hand-correction once real `module avail` output is seen.
for PROFILE in profile/deeplrn profile/candidate profile/astro; do
  if module load "$PROFILE" >/dev/null 2>&1; then
    echo
    echo "=== after 'module load $PROFILE' ==="
    module avail 2>&1 | grep -iE "jax|cuda|python/|cineca-ai" || echo "(none matched)"
    module unload "$PROFILE" >/dev/null 2>&1
  fi
done
rm -f /tmp/module_avail.$$

echo
echo "=== does any module-provided python already import a GPU jax? ==="
# cineca-ai only appears after its profile is loaded (first run, 2026-11-25: deeplrn lists
# cineca-ai/3.0.0-4.3.0, candidate lists cineca-ai/2026-09). python/3.11 alone has
# jax 0.7.0 CPU-only. `set +u` because module/activate scripts read unset variables.
set +u
for CAND in "profile/deeplrn cineca-ai/4.3.0" "profile/deeplrn cineca-ai/4.1.1" \
            "profile/candidate cineca-ai/2026-09"; do
  echo "--- modules: $CAND ---"
  ( module purge >/dev/null 2>&1
    module load $CAND >/dev/null 2>&1 || { echo "(load failed)"; exit 0; }
    which python3
    python3 -c "import jax, jaxlib; print('jax', jax.__version__, 'jaxlib', jaxlib.__version__, jax.devices())" 2>&1 | tail -2
    python3 -c "import numpyro; print('numpyro', numpyro.__version__)" 2>&1 | tail -1 )
done

echo
echo "=== baseline: the pinned env this campaign actually runs under ==="
module load openblas
eval "$(micromamba shell hook --shell=bash)"
micromamba activate jax_conda
set -u
nvidia-smi -L
python3 -c "import jax, numpyro; print('jax_conda jax', jax.__version__, 'numpyro', numpyro.__version__, jax.devices())"

echo
echo "[probe] done. The baseline must list a CudaDevice, or nothing else here will run."
