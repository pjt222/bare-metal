# Documentation index

Single navigation entry for all documentation in this repository.
Files are listed in dependency order: read top to bottom and each
section assumes only what precedes it.

## Project entry points

| File                                       | Role                                                       |
|--------------------------------------------|------------------------------------------------------------|
| [`../README.md`](../README.md)             | Repository overview, headline numbers, hardware            |
| [`../SETUP.md`](../SETUP.md)               | Environment install and `make reproduce` walkthrough       |
| [`../AGENTS.md`](../AGENTS.md)             | Canonical agent-facing reference                           |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | Contributor guide and PR checklist                         |
| [`../CHANGELOG.md`](../CHANGELOG.md)       | Structural reorganizations and audit history               |

## Kernel reference

| File                                                       | Content                                                |
|------------------------------------------------------------|--------------------------------------------------------|
| [`inventory.md`](inventory.md)                             | Kernel inventory by family with peak numbers           |
| [`comparison_to_sota.md`](comparison_to_sota.md)           | Measured gap to cuBLAS / cuDNN / cuSPARSELt            |
| [`roofline_measured.md`](roofline_measured.md)             | NCU-measured roofline per profiled kernel              |
| [`sass_histogram.md`](sass_histogram.md)                   | Per-kernel SASS instruction mix                        |
| [`register_audit.md`](register_audit.md)                   | Per-kernel register usage                              |
| [`ncu_metrics.md`](ncu_metrics.md)                         | NCU profiling harness reference                        |

Underlying data lives in `../data/` (`baselines.json`,
`reference_baselines.json`, `sass_histogram.csv`,
`register_audit.csv`).

## Hardware and SASS reference

| File                                               | Content                                                  |
|----------------------------------------------------|----------------------------------------------------------|
| [`ampere_sass_reference.md`](ampere_sass_reference.md) | sm_86 instruction reference                          |
| [`control_codes.md`](control_codes.md)             | Stall counts, barriers, yield, scoreboards               |
| [`memory_hierarchy.md`](memory_hierarchy.md)       | GA104 memory subsystem and the 50 KB cliff               |
| [`cuasm_r.md`](cuasm_r.md)                         | Local R package for SASS byte-level hand-edits           |

## Tutorial series

`tutorial/` contains a six-chapter prose walkthrough of the work in
this repository. Chapters can be read in any order; the
recommended dependency chain is 02 → 03 → 04 → 05, with 01 as a
prerequisite and 06 as synthesis.

| Chapter | Title                                                          |
|---------|----------------------------------------------------------------|
| 01      | [SASS Hello World](tutorial/01-sass-hello-world.md)            |
| 02      | [GEMM from Scratch](tutorial/02-gemm-from-scratch.md)          |
| 03      | [INT8 Tensor Cores](tutorial/03-int8-tensor-cores.md)          |
| 04      | [Software Pipelining](tutorial/04-software-pipelining.md)      |
| 05      | [Flash Attention](tutorial/05-flash-attention.md)              |
| 06      | [The Four Laws](tutorial/06-the-four-laws.md)                  |

## Analyses and postmortems

| File                                                                                 | Content                                          |
|--------------------------------------------------------------------------------------|--------------------------------------------------|
| [`gpu_reflections.md`](gpu_reflections.md)                                           | Observation catalogue. The first-person voice is a deliberate stylistic experiment; see file preamble. |
| [`fragment_shfl_reductions.md`](fragment_shfl_reductions.md)                         | Reusable Tensor Core reduction pattern           |
| [`cymatic_memory_mapping.md`](cymatic_memory_mapping.md)                             | Chladni-pattern memory layout: theory and bench  |
| [`diffusion_primitives.md`](diffusion_primitives.md)                                 | UNet primitive inventory                         |
| [`int8_sparse_4096_regression_analysis.md`](int8_sparse_4096_regression_analysis.md) | Companion to Observation HH                      |
| [`polyhedral_spring_networks.md`](polyhedral_spring_networks.md)                     | Literature scoping for issue #32 (research-grade)|
| [`troubleshooting.md`](troubleshooting.md)                                           | Common pitfalls and remedies                     |

## Publishing pipeline

`analysis/` is the [Quarto](https://quarto.org/) project used to
render long-form analyses for publication. Rendered HTML lands in
`analysis/_output/`. To render:

```bash
cd docs/analysis
quarto render
```

Current documents:

| File                                                                   | Role                                          |
|------------------------------------------------------------------------|-----------------------------------------------|
| [`analysis/kernel_architecture.qmd`](analysis/kernel_architecture.qmd) | Kernel architecture analysis (issue #22)      |

## Visualizations

`../viz/` holds interactive visualizations built with web tooling.
Each subdirectory is a standalone application with its own
`package.json`.

| Subdirectory                                         | Stack          | Purpose                                            |
|------------------------------------------------------|----------------|----------------------------------------------------|
| [`../viz/research-map/`](../viz/research-map/)       | Vite + three.js | Interactive knowledge graph of the project's optimization observations and their cross-references |

To run a viz locally:

```bash
cd viz/research-map
npm install
npm run dev          # serves on localhost
npm run build        # produces viz/research-map/dist/
```

Generated `dist/` and `node_modules/` directories are gitignored.

## Figures

`figures/` contains all PNG figures referenced from the
documentation. The headline figures are regenerated by
`Rscript scripts/audit/generate_readme_figures.R`. The
`figures/cymatic/` subdirectory holds the per-mode Chladni-layout
visualizations produced by `scripts/cymatic/cymatic_visualize.R`.

## Session handoff

[`CONTINUE_HERE.md`](CONTINUE_HERE.md) is a per-author session
scratchpad. It records the work completed in the most recent
working session and the next concrete steps. Not durable
documentation; expected to churn between sessions.
