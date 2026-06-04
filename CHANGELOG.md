# CHANGELOG

Structural reorganizations, audit passes, and policy changes that
affect the on-disk layout or build/test interface. Per-kernel
performance changes are recorded in `docs/gpu_reflections.md`;
per-issue closures live in the GitHub issue tracker.

The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), with
"Tier N" referring to internal audit episodes preserved here for
historical reference.

## Unreleased

### Added
- **`make bench-all` full-corpus runner (#124).** New on-demand "run
  everything" pass: `scripts/bench/bench_all.R` discovers the whole
  `$(BENCH_EXES)` corpus, runs every bench, and records every attempt +
  per-config summary + run metadata to
  `results/bench_all/<timestamp>/{results.json,summary.md,samples.jsonl}`.
  Skip nothing, record everything (docs/benchmark_methodology.md). Reuses
  the cuasmR measurement API (no reimplementation). Per-bench invocation +
  output-parse hints live in `scripts/bench/bench_all.yml` (all 48 corpus
  exes specced). Benches that emit no single number (A/B sweep tables,
  ms-only pipelines, correctness harnesses) are tagged `non-measurable`
  and run once; un-runnable ones (cuDNN-SDPA stub, cymatic needs data
  files) are documented-`skipped` — neither is reported as a kernel
  `failed`. Each entry carries a `verified`/`infer` flag separating
  baseline-confirmed specs from source-inferred ones. The fast regression
  gate (`make bench` / `bench_regress.R`) is unchanged. GPU-free unit
  tests in `tests/bench_all/`.

### Changed
- **cuasmR measurement-API migration (#134, cuasmR 0.1.0 → 0.2.0).** The
  benchmark run → parse → validate → regress logic was migrated out of the
  `scripts/{bench,probe}` harnesses into the `cuasmR` package as 12 new exports
  (`run_bench`, `parse_throughput`, `validate_sample`, `collect_valid_samples`,
  `report_median_metrics`, `check_regression`, `append_jsonl_row`, `read_jsonl`,
  `capture_gpu_state`, `classify_meta`, `decode_throttle`, `summarise_meta`).
  Seven harness scripts now `library(cuasmR)`; the package gained roxygen
  `man/`, declared `Imports`, and a clean `R CMD check --as-cran` (0/0/1 on
  Linux R 4.6.0). Landed as stacked PRs #136 + #137.
- **bench_flash_all revival (#138, PR #139).** `bench_flash_all.R` rewired onto
  `cuasmR::run_bench` + `parse_throughput` (removing a duplicated run-and-parse
  primitive and a can't-launch crash handler) and revived: its dead
  `phase3/flash_attention` discovery path → `kernels/attention/flash_attention`,
  `REPO_ROOT` derivation fixed to a `.git`/`renv.lock` marker-search, and the
  `--build` target `make phase3` → `make attention`. The `bench_imma_s02/s04.R`
  `run_bench_grep` grep-extract helper was documented as an intentionally
  distinct shape (not migrated).
- Documentation review pass. Voice and provenance leakage cleaned
  up across user-facing docs; "Tier N" jargon retained only in
  this file. Created `AGENTS.md` as the canonical agent-facing
  reference; `CLAUDE.md` and `.github/copilot-instructions.md`
  rewritten as thin pointers. Regenerable CSV/JSON moved from
  `docs/` to `data/`. `docs/kernels.md` removed in favor of a
  single family-axis inventory at `docs/inventory.md`. Per-kernel
  READMEs de-phased. Created `docs/index.md` as the documentation
  map. `Makefile` per-family `bench_%` rules collapsed to a single
  pattern.

### Removed
- `.github/issues/*.md` and `scripts/create_issues.sh`. All 16
  seed files corresponded to GitHub issues that have been
  open-then-closed (#35–#44, #55–#69). The seed-and-push workflow
  has run; the content lives in GitHub issue history.
- Empty `tools/` directory.
- `.github/SESSION_INSIGHTS_2026-05-05.md`. Content absorbed into
  the audit-history section below.

## Audit history

### Tier 13 — 2026-05-10 — kernel tree by family

Reorganized `kernels/` from `phaseN/` to family directories. Each
kernel directory now contains its `.cu` source, one or more
`bench*.cu` harnesses, and a `README.md` with measured results.

| Step | Commit  | Move                                              |
|-----:|---------|---------------------------------------------------|
|  1   | bf278c6 | `phase1/` → `kernels/tutorial/`                   |
|  2   | 50f31c4 | `phase2/common/` → `kernels/_common/`             |
|  3   | 3bd16c3 | `phase2/{sgemm,hgemm,hgemm_sparse,igemm}/` → `kernels/gemm/` |
|  4   | cde3d40 | reductions family                                 |
|  5   | c82d44e | attention family                                  |
|  6   | 24e6810 | convolution family                                |
|  7   | 5ba383b | elementwise family                                |
|  8   | 260072b | memory_layout family (cymatic)                    |
| 9–10 | b18dc1b | composition family; `phase{1..5}/` directories deleted |
| follow | efbe90c | README link audit; 29 broken cross-refs fixed   |

### Tier 12 — 2026-05-10 — speculative tag dropped

Removed the "speculative" / second-class distinction from kernel
directories. Added `docs/kernels_by_family.md`. Commit 329e80f.

### Tier 11 — 2026-05-10 — reproducibility orchestration

Rewrote `SETUP.md`; added `make reproduce` as the single
setup → verify → all → bench entry point. Commit dae64ce.

### Tier 10 — 2026-05-10 — fair-run capture

`scripts/bench/bench_regress.R` captures GPU and host state around
each bench run and skips unfair runs (thermal throttle, power cap)
instead of failing them. Commit a092b4a.

### Tier 9 — 2026-05-10 — bench_regress parser repair

Parser fixes and a `testthat` suite for
`scripts/bench/bench_regress.R`; baseline schema extended. Commit
a179a6a.

### Tier 8 — 2026-05-10 — bench filename normalization

Renamed three bench files to drop redundant dir-name prefixes
(`kernels/gemm/hgemm/bench_hgemm_persistent.cu` →
`kernels/gemm/hgemm/bench_persistent.cu`, etc.). Commit c2366ab.

### Tier 7 — 2026-05-10 — results centralization

Moved per-run artifacts into a unified `results/` tree; grouped
`docs/figures/cymatic/`. Commit a50cf5c.

### Tier 6 — 2026-05-10 — bench-variant naming convention

Documented the `bench.cu` / `bench_<variant>.cu` naming convention;
dropped redundant demo files and stale binaries. Commit 4a878b8.

### Tier 5 — 2026-05-10 — scripts regrouping

`scripts/` regrouped into five purpose-named subdirectories
(`audit`, `bench`, `cymatic`, `model`, `profile`) plus top-level
setup drivers. Added `scripts/README.md`. Commit e7e4428.

### Tier 4 — 2026-05-10 — top-level rename

`phase6/` → `experiments/`; `setup.md` → `SETUP.md`; `CONTINUE_HERE.md`
moved under `docs/`. Commit 96967f8.

### Tiers 1–3

Pre-naming refactors. See git log between commit fecc775 (initial
commit, 2026-03-29) and 96967f8 for the full sequence.

## Session insights (2026-05-05, GA104)

Notable findings from the 2026-05-05 working session, retained
because they document repository-level rather than per-kernel
lessons. Originally filed at `.github/SESSION_INSIGHTS_2026-05-05.md`.

### Infrastructure

- GitHub issues are stored in GitHub's database, not in the repo;
  local issue `.md` files do not sync automatically and require
  `gh issue create` or an API call to materialize.
- The `.gitignore` `bench_*` pattern is aggressive enough to catch
  `bench_driver.h` and friends. Explicit `!`-exceptions are
  required for any artifact that matches but should be tracked.
- Hand-tuned `.cubin` binaries (e.g.
  `igemm_tiled_handtuned.sm_86.cubin`) are neither committed nor
  fully ignored. Each such artifact needs an explicit decision to
  commit-as-artifact or `.gitignore` plus a documented rebuild
  path.

### GPU architecture

- The 4 MB L2 cache is a hard limit for metadata-heavy kernels:
  sparse 2:4 GEMM metadata at 4096³ totals 4.1 MB, exactly at
  the thrashing threshold. The crossover is sharp, not gradual.
- Instruction-mix counts (e.g. 160 PRMT vs 64 PRMT, manual INT8
  pack vs LDSM path) explain the constant-factor cost of one
  implementation versus another, but size-scaling analysis is
  what reveals memory-hierarchy bottlenecks.
- `ncu` was at the time blocked on GeForce GPUs by
  `ERR_NVGPUCTRPERM`. The accessible profiling stack was
  `cuobjdump` plus `nvcc --cubin -res-usage` plus the in-tree
  benches. (`ncu` access was later unblocked; see Observation U
  in `docs/gpu_reflections.md`.)
- Register count alone is misleading: the sparse kernel uses 64
  regs (good) vs dense's 126, but the real bottleneck is the
  memory hierarchy, not occupancy.

### Project hygiene

- Bench boilerplate was the largest contributor to codebase size:
  26 bench files × ~100 duplicate lines = ~2,600 lines.
  `bench_driver.h` reduced the first three refactored files from
  1,739 to 297 lines (-83%).
- `Makefile` rules needed per-family specificity; a single
  `bench*` wildcard was too broad for Flash Attention's ten-plus
  variants.
- A `CHECK_CU` macro had a missing brace inside its `do { ... }
  while(0)` body, causing `cuCtxCreate` compilation failures.
  Fixed in PR #54.
