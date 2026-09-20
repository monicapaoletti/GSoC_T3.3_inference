# Two targets from one definition:
#   science : Python 3.10 + the pinned MPR inference stack. The cluster Apptainer image is
#             built from this, so a fit there runs the versions the laptop developed against.
#   claude  : science + Claude Code. The container to develop in.
#
# Unlike the choices_history container, this one is NOT a safety boundary. The structural
# connectivity is public and everything else is simulated, so there is no firewall, no
# dangling data symlinks and nothing to keep Claude away from. It exists so that the
# environment is reproducible and identical on the laptop and on the clusters.

FROM python:3.10.15-slim-bookworm AS science

# jax==0.6.2         -> CPU wheels; the laptop has no GPU, so this is the default.
# jax[cuda12]==0.6.2 -> self-contained CUDA-12 wheels for doslis (L4) and UGent (H100).
#                       Needs an NVIDIA driver >= 525, via CUDA minor-version forward
#                       compatibility (validated on driver 535 / L4).
ARG JAX_SPEC="jax==0.6.2"

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# g++ and OpenBLAS are runtime requirements, not just build ones: pytensor compiles its C
# backend on first use and links BLAS then. Without them pymc silently falls back to a slow
# NumPy C-API BLAS.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libopenblas-dev \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /opt/env \
 && printf '[blas]\nldflags = -lopenblas\n' > /opt/env/pytensorrc
# A config file rather than PYTENSOR_FLAGS, so jobscripts can still set their own flags.
ENV PYTENSORRC=/opt/env/pytensorrc

COPY requirements.in /opt/env/
RUN pip install --upgrade pip \
 && pip install "${JAX_SPEC}" -r /opt/env/requirements.in \
 && python -c "import jax, numpyro, pymc, arviz; \
print('jax', jax.__version__, '| numpyro', numpyro.__version__, \
      '| pymc', pymc.__version__, '| arviz', arviz.__version__)"


FROM science AS claude

ARG USER_UID=1001
ARG USER_GID=1001

# The laptop has no GPU; without this JAX prints a warning on every import.
ENV JAX_PLATFORMS=cpu

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git curl ca-certificates openssh-client less procps jq \
 && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid ${USER_GID} dev \
 && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash dev \
 && mkdir -p /home/dev/.claude/projects/-workspaces-GSoC-T3-3-inference/memory \
 && chown -R dev:dev /home/dev

COPY entrypoint.sh /usr/local/bin/
RUN chmod 755 /usr/local/bin/entrypoint.sh

USER dev
WORKDIR /home/dev

# Downloaded first so a failed download fails the build (a pipe into bash would not).
RUN curl -fsSL https://claude.ai/install.sh -o /tmp/install-claude.sh \
 && bash /tmp/install-claude.sh \
 && rm /tmp/install-claude.sh \
 && /home/dev/.local/bin/claude --version
ENV PATH=/home/dev/.local/bin:$PATH \
    CLAUDE_CONFIG_DIR=/home/dev/.claude \
    DISABLE_AUTOUPDATER=1

# A git identity, unlike the choices_history container which deliberately has none. The
# repo's CLAUDE.md says Claude's commits are Claude's own, so a commit made in here is
# authored correctly without anyone having to remember a flag. Monica still commits from
# the host, as herself.
RUN git config --global user.name "Claude" \
 && git config --global user.email "noreply@anthropic.com" \
 && git config --global --add safe.directory /workspaces/GSoC_T3.3_inference

WORKDIR /workspaces/GSoC_T3.3_inference
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["claude"]
