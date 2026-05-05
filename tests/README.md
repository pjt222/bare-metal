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

Build any test individually with the same `nvcc` commands used for production kernels.
