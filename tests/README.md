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
| `bench_regress/test_parser.R` | Parser tests for `scripts/bench/bench_regress.R` (14 groups, 32 assertions; canned bench-stdout fixtures) |
| `bench_regress/test_meta.R`   | GPU/host metadata tests for `scripts/bench/bench_meta.R` (Tier 10; 20 groups, 35 assertions; throttle decode + classify_meta policies; live-capture skipped without nvidia-smi) |

Build any CUDA test individually with the same `nvcc` commands used for production kernels.
Run the R tests:

```bash
Rscript tests/bench_regress/test_parser.R
Rscript tests/bench_regress/test_meta.R
```
