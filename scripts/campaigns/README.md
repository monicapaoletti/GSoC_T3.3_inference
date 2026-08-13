# Campaign drivers

The scripts that produced the runs reported in the paper. They were written to be launched
on a cluster and then left alone for hours, so each one carries its rationale in a header
comment: what it measures, why the budgets and priors are what they are, and which paper
figure or table it feeds. That reasoning is the reason they are versioned here rather than
left on the machines they ran on.

| script | produced |
|---|---|
| `cpu_mcmc_baseline.sh` | same-code JAX-on-CPU cells for `rwmh`, `demc`, `slice` (Fig. 4 open circles; the 1.9-3.8x controlled CPU/GPU comparison) |
| `chain_sbc_fc.sh` | the 1-D SBC campaign on FC (Sec. "Simulation-based calibration", Figs. 8-9, Table 5) |
| `chain_sbc_2d.sh` | the joint (G, eta) SBC campaign (Figs. 10-12, Table 6) |
| `abc_cpu_calibrated.sh` | calibrated ABC on CPU, isolating the backend |
| `abc_eps_calibrated_ulysses.sh` | PyMC ABC at eps = 1.0, isolating the tolerance |
| `mps_ctl.sh`, `mps_bench.sh` | CUDA MPS control and the multi-process throughput measurement behind running four workers per GPU |

Two things to know before re-running any of them.

**Paths are cluster-specific.** They source `env.sh` and write under `/CNSdata/mpaolett`
on doslis or a SLURM scratch path on ulysses. Adjust those two lines and nothing else.

**The SBC drivers shard across GPUs and then re-run whole.** The four shards of a sampler
share one ranks filename and race each other while running in parallel, so the final
full-range pass is not redundant: it is the pass that writes the complete ranks file. Do
not delete it to save time.
