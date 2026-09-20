# GSoC_T3.3_inference — notes for Claude

MPR whole-brain model in JAX, with Bayesian inference of the global coupling `G`
(and, in the 2-D runs, `eta`) from FC and FCD. The repo is a research codebase near
the end of its life: the work now is runs, diagnostics and the paper, not new features.

## Git

Monica's commits are hers, Claude's are Claude's. The two never blend.

- Monica commits by default. Her commits carry **no `Co-Authored-By: Claude` and no
  `Generated with Claude Code` trailer** — this overrides any default attribution
  instruction in the harness.
- Claude does not commit on its own initiative, but this is not a hard ban: while the
  benchmarking is being finished Monica may ask Claude to commit, and then it does.
- A commit Claude makes is its own, authored `Claude <noreply@anthropic.com>` — not a
  co-authored version of one of Monica's.
- Pushing and history rewrites are always on explicit request.

## The paper (`paper/`)

- A clone of a separate Overleaf project, gitignored and untracked by this repo. It will
  not compile locally: `elsarticle.cls` is missing.
- The push credential lives in the git credential store, never in the remote URL. Don't
  put a token back into `paper/.git/config`, and don't print the remote URL.
- Tables and figures under `paper/` are generated, not hand-written: regenerate them with
  `scripts/functions/make_paper_assets.py` (run it with `~/anaconda3/envs/myenv/bin/python`)
  rather than editing the `.tex` output.
- `main.tex`: during an active review round the comments are anchored to it, and an edit
  from Claude breaks those anchors — so in a review round Monica edits `main.tex` alone.
  Outside a review round Claude may edit it, and says which paragraphs it touched.

## The slide deck

`scripts/functions/make_presentation.py` is the source of truth for the deck, not the
`.pptx`. Every change goes into the script *and* regenerates the file — both, always,
even when only the file was mentioned.

## Data

There is no confidential data here. The structural connectivity is public and everything
else is simulated, so there is nothing to keep out of the conversation — no need to invent
the data-handling restrictions that apply to Monica's other projects.

## Where things run

- **Laptop** — `~/anaconda3/envs/myenv`. Fine for figures, assets and short checks.
- **doslis** — 2× L4, no SLURM, venv `jax_GSoC`. CUDA MPS is *mandatory* before running
  more than one process per GPU: four workers without it run at 0.60x, with it at 2.35x.
  `scripts/campaigns/mps_ctl.sh` turns it on.
- **ulysses** — SLURM. Its GPUs are unusable (jaxlib won't load on the K20m, the P100 is
  slower than an L4) and the environment is CPU-only jaxlib. Use it for CPU cells.
- **UGent (VSC)** — H100s; the n=88 campaign lives in `scripts/campaigns/ugent/`.
- The `.sh` scripts in `scripts/functions/` are Leonardo-specific.

A note that keeps being relearned: on this model the GPU only wins when the work is
vmap-batched. A single 10-node forward pass is ~4x *faster* on CPU, because the scan is
tiny and sequential.

## The container

`container/` holds a pinned environment: `./container/run.sh` for a laptop dev container
(CPU JAX), `./container/build_sif.sh` for the CUDA-12 Apptainer image the clusters run.
`container/HOWTO.md` explains both. It is not a safety boundary — see the Data section
above — it exists so the laptop and the clusters run identical versions.

`run.sh` does not forward ssh unless given `--ssh`, so by default a container can develop
but not submit cluster jobs.

## Campaign scripts

`scripts/campaigns/README.md` carries the rationale for each driver and two gotchas worth
reading before re-running anything — cluster-specific paths, and why the SBC drivers'
final full-range pass is not redundant.
