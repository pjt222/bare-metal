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
- **`make test-r` — the GPU-free R suites are now a gate (#163).** They were
  invoked by nothing before this: not the pre-push hook, not the Makefile, not
  CI. `scripts/audit/run_r_tests.R` discovers every `tests/**/test[-_]*.R` plus
  the `cuasmR` package suite (glob, not a manifest — a manifest would recreate
  the bug being fixed; both separators, because `test-` is what
  `usethis::use_test()` emits) and runs each in its own child `Rscript`,
  serially. Separate
  processes because `scripts/bench/bench_all.R` and `bench_regress.R` both define
  `main` and `parse_args` and the suites `source()` into `globalenv()`; serially
  because concurrent R on the 9p `/mnt/d` mount is what turned a 75 s suite into
  an hour (#162). Wired into `.githooks/pre-push` (blocking, after the renv check
  and before the CUDA build) and into a new `.github/workflows/tests.yml` with an
  `renv.lock`-keyed cache. 115 s of tests on AC; AGENTS.md records the
  measurement conditions, and why the battery figure is only 1.16× higher
  despite a 2.1× slower R startup — and that ~7/8 of the local cost is the 9p
  mount, measured against the same repo on ext4 on the same box (14 s vs 115 s).
  Proven able to fail two ways before landing, against a clean tree's exit 0:
  a real dead suite, and a deliberate mutation of `normalise_clock`'s lower
  bound. The dead-suite demonstration is not reproducible from the resulting
  tree, since this change deletes that suite.
  Three guards exist against the gate quietly hollowing out, because every part
  of its verdict is otherwise self-reported: `--expect N` (passed by both the
  hook and CI) supplies the suite-count denominator from outside, since "all
  discovered suites passed" is a ratio against whatever was found and so always
  100%; skipped tests are counted and reported per suite rather than folded into
  a pass, without which the `cuasmR` byte-identical roundtrip check would stop
  running the moment `nvdisasm` left `PATH`; and a plumbing canary runs one child
  that exits 3 and requires a non-zero back, because the whole verdict flows
  through one `set -o pipefail` — measured, a pipeline without it reports 0 for
  a child that exited 3, which would pass every suite silently.
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

### Fixed
- **The regression gate no longer reports `PASSED` having measured nothing
  (#176).** `bench_regress.R` decided its verdict on `regressions > 0L` alone, so
  a run in which every config skipped printed `RESULT: PASSED -- all benchmarks
  within tolerance` and exited 0 with zero kernels compared against zero
  baselines. It did so on five consecutive real pushes at 7 of 7 skipped, and the
  pre-push hook rendered that as a green "All benchmarks within tolerance. Push
  allowed." Two effects stack to make the empty run routine rather than
  occasional: #156 deliberately put three of the seven configs behind a host-side
  clock lock the hook never applies, and throttle-skip takes the rest whenever
  the laptop is warm. There is now a third outcome, `INCONCLUSIVE`, with its own
  exit code (`0` PASSED / `1` FAILED / `2` INCONCLUSIVE), and every verdict line
  names the fraction measured — `PASSED -- 3 of 7 config(s) measured, all within
  tolerance (4 skipped)`, which is the sentence the old one should have been. A
  measured regression still outranks everything: 1 config measured and regressed
  is `FAILED`, not `INCONCLUSIVE`. Configs belonging to a kernel with no built
  executable are now counted as skipped instead of vanishing from the
  denominator, so an unbuilt corpus reports `0 of 7`, not `Total: 0` followed by
  a pass. **On all-skip the hook warns and allows the push** — a deliberate
  choice, argued in `AGENTS.md` next to the hook step table: blocking would
  reject ordinary pushes on this machine and its escape hatch, `git push
  --no-verify`, disables all five hook steps including the #163 R suites. All
  three callers read the same code: hook warns, `make bench` warns,
  `run_locked_eval.ps1` propagates. Two of those callers needed repairing before
  that sentence was true, and both failed the same way the gate did — treating
  "no measurement happened" as a measurement. `run_locked_eval.ps1` sets
  `$PSNativeCommandUseErrorActionPreference`, so on PowerShell 7 any non-zero
  exit from its `wsl.exe … Rscript` call threw before `$BenchExit =
  $LASTEXITCODE` ran: it reported 1 for both FAILED and INCONCLUSIVE and wrote no
  results record at all for the runs that measured nothing (measured on pwsh
  7.6.3; a no-op on PS 5.1). And `Rscript` exits 2 when it cannot open the script
  file, which is INCONCLUSIVE's code, so `make bench` now guards on `test -r`
  first — a missing gate is an error, not an empty measurement.
  Proven both ways on real runs before landing
  — an all-skip (`Total: 2 | Measured: 0` → INCONCLUSIVE, exit 2, where the same
  command on the parent commit printed PASSED and exited 0) and a live one
  (`Total: 7 | Measured: 3` → PASSED, exit 0). The verdict is a pure function of
  four counters and is unit-tested by the new
  `tests/bench_regress/test_verdict.R`, which the #163 gate discovers
  automatically; `R_SUITES` and the `tests.yml` denominators go 3 → 4.
- **Retired the dead `tests/bench_regress/test_parser.R` (#171).** All 14 of its
  `test_that` groups had been erroring since `caeca97` (2026-06-02), which removed
  the `.pick_line` / `.parse_line` pair it exercises in favour of
  `cuasmR::parse_throughput` (#134) without updating the test — 58 days dead,
  unnoticed because nothing ran it. Two things hid it: the file ends in an
  unconditional `cat("All bench_regress parser tests passed.")` that printed while
  every group errored, and testthat's default `max_fails` of 10 truncated the
  tally so four groups never even executed. The behaviours it covered that nothing
  else does — 8 of 14, including the only section-filter case that can actually
  detect a broken filter — are tracked in #172.
- **`cuasmR` roundtrip tests now skip when `nvdisasm` is off `PATH`.** They
  guarded on the cubin existing but not on the disassembler, so a developer with
  cubins built and CUDA off `PATH` got a hard `stop()` from `disasm.R:6` — an
  environment failure blocking a push, inside a target whose whole selling point
  is that it needs no GPU.
- **The pre-push hook's repo-identity probe no longer disables everything
  (#177).** It tested for `scripts/bench/bench_regress.R` and `exit 0`'d the
  entire hook on its absence — switching off the README link audit, the renv
  sync check and the GPU-free R suites for the absence of a GPU-benchmark
  script. A worktree, a sparse checkout or a rename of that one file was enough.
  The probe now checks that this is the repo root at all (`Makefile` plus a
  `.git` directory *or* file, the latter being how worktrees present), and the
  regression step guards itself the way every other step already did.
- **Documentation drift around the gate.** `AGENTS.md` named
  `scripts/install-hooks.sh` as the gate (that script *installs* it; the gate is
  `.githooks/pre-push`), described `make reproduce` as four steps when it is five,
  and omitted `renv-check`, `bench-all` and `figures` from the target list.
  `tests/README.md` omitted `tests/bench_all/` entirely, described `test_meta.R`
  as testing "cuasmR GPU-state functions" when it sources
  `scripts/bench/bench_meta.R`, and advertised the dead parser suite as live with
  exact assertion counts. `CONTRIBUTING.md`'s hook list was missing two of the
  four steps it already had.
- **bench↔cubin name mismatch across BenchDriver-refactored benches
  (#148).** ~16 flash-attention, resblock, and attention-layer benches
  called `load_kernel("<basename>.sm_86.cubin", …)` with abbreviated
  cubin basenames the build never emits (`flash_*` instead of
  `flash_attn_*`, `resblock` instead of `resblock_fused`), so they
  crashed at runtime with `cuModuleLoad … file not found`. These
  benches are outside `make test`'s smoke loop, so the breakage was
  silent until the `make bench-all` full-corpus runner (#124) surfaced
  it. Corrected every load basename to the actual built cubin (the
  least-churn option, matching #127). Three further sub-fixes:
  `bench_br16_regpv_pad`'s two compile-time pad variants (`kv8_w0`,
  `kv0_w4`) now have explicit `-D` Makefile rules and join
  `KERNEL_CUBINS` (the default `make` cubin is the `kv8_w4` layout);
  `attention_layer/bench` also had a redundant `kernels/` segment in
  its cross-directory load paths (`../../kernels/…` → `../../…`), fixed
  so it resolves from the bench's own working directory. All 16
  affected benches now load and run; verified by direct per-bench runs.

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
