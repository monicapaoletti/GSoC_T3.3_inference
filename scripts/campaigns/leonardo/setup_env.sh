#!/usr/bin/env bash
# Build the micromamba env `jax_n88` for the Leonardo n=88 campaign. Run ONCE, on a LOGIN
# node (compute nodes have no internet):   bash setup_env.sh
#
# WHY A NEW ENV. The existing `jax_conda` has no numpyro (probe 2026-11-25), and it is what
# the older Leonardo scripts in scripts/functions/ use, so it is left alone.
#
# WHY THESE VERSIONS. They are the ones the UGent n=88 env ran (built 2026-09-08 to
# replicate doslis), so Leonardo results compare against UGent's on hardware alone. Note
# they are NOT requirements-gpu.txt (jax 0.6.2 / numpyro 0.19), which no cluster ran.
#
# ONE pip resolve, with the transitive pins that matter: installed in several calls, chex
# (via optax) upgraded jax to 0.7.1 at UGent, orphaned the CUDA plugin, and pulled a
# matplotlib/pandas that broke arviz 0.20 -- an env that still imports, without a GPU.
# chex and orbax-checkpoint are pinned to releases contemporary with jax 0.4.27 for that
# reason.
set -eo pipefail
ENV_NAME="${ENV_NAME:-jax_n88}"

eval "$(micromamba shell hook --shell=bash)"
if micromamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "env $ENV_NAME already exists -- remove it first (micromamba env remove -n $ENV_NAME)" >&2
  exit 1
fi
micromamba create -y -n "$ENV_NAME" -c conda-forge python=3.10 pip
micromamba activate "$ENV_NAME"

python -m pip install --no-cache-dir \
  "jax[cuda12]==0.4.27" "jaxlib==0.4.27" \
  numpyro==0.16.1 blackjax==1.2.1 flax==0.8.2 optax==0.2.2 \
  chex==0.1.86 orbax-checkpoint==0.5.9 jaxopt==0.8.3 \
  arviz==0.20.0 numpy==1.26.4 scipy==1.13.1 networkx==3.3 seaborn==0.13.2 \
  matplotlib==3.8.4 pandas==2.2.2 xarray==2024.3.0 h5py==3.11.0 tqdm

# Fail loudly if anything drifted. GPU visibility can only be checked on a compute node
# (probe_jax_modules.sh does it); here, versions only.
python - <<'EOF'
import jax, jaxlib, numpyro, blackjax, flax, optax, arviz, numpy, scipy
got = {"jax": jax.__version__, "jaxlib": jaxlib.__version__, "numpyro": numpyro.__version__,
       "blackjax": blackjax.__version__, "flax": flax.__version__, "optax": optax.__version__,
       "arviz": arviz.__version__, "numpy": numpy.__version__, "scipy": scipy.__version__}
want = {"jax": "0.4.27", "jaxlib": "0.4.27", "numpyro": "0.16.1", "numpy": "1.26.4"}
print(got)
bad = {k: (got[k], v) for k, v in want.items() if got[k] != v}
assert not bad, f"version drift: {bad}"
from importlib.metadata import version
print("cuda plugin:", version("jax-cuda12-plugin"))
print("OK -- now run probe_jax_modules.sh to confirm the GPU is seen")
EOF
