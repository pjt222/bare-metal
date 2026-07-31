# Tests

This directory contains **development and verification tests** — not benchmarks.

These files validate hardware assumptions, fragment layouts, and race conditions. They are not production kernels and do not measure performance.

| File | Purpose |
|------|---------|
| `hgemm_sparse/test_dense_manual.cu` | Sparse 2:4 layout verification (manual dense reference) |
| `hgemm_sparse/test_mma_sp.cu` | Verify `mma.sp` instruction on sm_86 |
| `hgemm_sparse/verify_wmma_ab_layout.cu` | Fragment register layout validation |
| `igemm/test_inplace_race.cu` | Reproduce WAR hazard in in-place INT8 quantization |
| `flash_attention/verify_wmma_layout.cu` | WMMA accumulator fragment layout (sm_86) |
| `bench_all/test_bench_all.R` | GPU-free unit tests for `scripts/bench/bench_all.R` and `bench_all_collect.R` (#124, #152): corpus discovery, spec merge, the taxonomy×regime planner, status classification, summary aggregation, markdown render |
| `bench_regress/test_meta.R`   | Metadata tests for `scripts/bench/bench_meta.R` — throttle-reason decode and `classify_meta` policies against canned snapshots; the live-capture smoke test skips without `nvidia-smi` |
| `bench_regress/test_verdict.R` | The run verdict in `scripts/bench/bench_regress.R` (#176): PASSED / FAILED / INCONCLUSIVE and their three exit codes, pinning that an all-skipped run can never print "all benchmarks within tolerance" again |

Build any CUDA test individually with the same `nvcc` commands used for production kernels.

## Running the R tests

The GPU-free R suites are a **pre-push gate** (#163). Run all of them with:

```bash
make test-r
```

That covers the three R files above plus the `cuasmR` package's own testthat suite
(`R/cuasmR/tests/testthat/`), which is not under `tests/` but is gated the same
way — it holds the only characterization test for the parser that
`scripts/bench/bench_regress.R` calls in the regression gate.

To see what would run without running it:

```bash
Rscript scripts/audit/run_r_tests.R --list
```

Discovery is by glob, so a new `tests/**/test[-_]*.R` is gated the day it lands —
no manifest to forget to update. The expected suite count is supplied from
outside the runner (`R_SUITES` in the `Makefile`, `--expect` in `tests.yml`), so
adding one is a deliberate two-line edit rather than something that happens
silently.

Two notes on reading the output:

- `test_meta.R` ends with an unconditional `cat("All ... tests passed.")`. That
  line prints regardless of the result. **The exit status is the verdict**,
  which is why the runner reports its own table. Do not add such a line to a new
  suite — `test_verdict.R` deliberately has none.
- Three of the `cuasmR` roundtrip tests skip unless `kernels/tutorial/vector_add.sm_86.cubin`
  has been built (it is gitignored) *and* `nvdisasm` is on `PATH`. They therefore
  always skip in CI. Run `make cubins` locally for that coverage.

Per-file assertion counts are deliberately not recorded here. They were, and they
drifted: `bench_regress/test_parser.R` was advertised as "14 groups, 32 assertions"
while every one of its groups had been erroring since 2026-06-02. It was retired
in #171; the behaviours it covered that nothing else does are tracked in #172.
