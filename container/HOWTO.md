# The container

```
 laptop (host)                      laptop: Claude container          doslis / UGent
 ─────────────                      ────────────────────────          ──────────────
 you, VS Code, myenv                Claude + the pinned stack         Apptainer, same stack
 git commit / push, as you          repo mounted, CPU JAX             CUDA-12 JAX, --nv
 cluster runs, ssh                  no ssh unless --ssh               the actual fits
```

One definition, two images. `science` is the pinned environment; `claude` is that plus
Claude Code. The cluster image is built from the same `science` target, so a fit on a GPU
runs the versions the laptop developed against.

## What this container is, and is not

It is **not** a safety boundary, and that is the difference from the `choices_history`
container in the phd project. There, the container exists to keep Claude away from
human-participant data: dangling data symlinks, no ssh, a firewall allowing only Anthropic
and PyPI. Here the structural connectivity is public and everything else is simulated, so
there is nothing to isolate. This container exists for one reason: the environment is
pinned and identical on the laptop and on the clusters.

If data ever does arrive in this project, the layout already handles it — keep it outside
the repo and mount it explicitly, and the same properties come back for free.

## Laptop

```bash
./container/run.sh            # first run builds the image (~10 min)
./container/run.sh --build    # after changing requirements.in
./container/run.sh bash       # a shell instead of Claude
./container/run.sh --ssh      # forward the host ssh agent (see below)
```

First time only: log in to Claude inside the container (it prints a URL). The login is kept
in the `gsoc-t33-claude-config` Docker volume, so it survives rebuilds.

The repo is mounted at `/workspaces/GSoC_T3.3_inference`, and the project memory directory
is mounted through, so memory written inside the container is the same memory as on the
host.

The laptop has no GPU, so the image installs CPU JAX and sets `JAX_PLATFORMS=cpu`. That is
the right default for more than convenience: on this model a single small forward pass is
about 4x *faster* on CPU than on a GPU, because the scan is tiny and sequential. The GPU
only wins once the work is vmap-batched.

## `--ssh`, and why it is off by default

`--ssh` forwards the host's ssh **agent socket** (not the keys — the container can use them
but never read them) plus a read-only `~/.ssh/config`, which is enough to reach doslis and
UGent from inside.

It is off by default because it is the one real decision this container asks of you: without
it the container can develop but not submit, and cluster runs stay something you launch from
the host; with it, Claude can drive jobs end to end. Nothing breaks either way — start
without it, and add the flag the day you want job submission to happen from inside.

## Clusters

```bash
./container/build_sif.sh          # builds `science` with CUDA-12 JAX, writes a tarball
```

Copy the tarball over, then on the cluster:

```bash
apptainer build ~/containers/gsoc_t33.sif docker-archive://gsoc_t33_science.tar.gz
apptainer run --nv ~/containers/gsoc_t33.sif python scripts/functions/...
```

Two things that bite:

- **`--nv` is not optional.** Without it Apptainer does not expose the GPUs and JAX silently
  falls back to CPU, which looks like a mysteriously slow run rather than an error.
- **The image needs no CUDA on the host**, only a driver >= 525 — the CUDA-12 wheels are
  self-contained. That is what lets one image serve both the L4s on doslis and the H100s at
  UGent.

On doslis, remember CUDA MPS: four workers per GPU without it run at 0.60x, with it at
2.35x. `scripts/campaigns/mps_ctl.sh` turns it on, and that is a host-side thing, outside
the container.

## Git inside the container

The container carries a git identity of `Claude <noreply@anthropic.com>`, so a commit made
in here is authored correctly per the repo's `CLAUDE.md` without anyone remembering a flag.
Your commits are made from the host, as you. Pushing is not done from in here.
