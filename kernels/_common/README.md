# kernels/_common — shared benchmark headers

Three header-only utilities included by every `bench.cu` across phases 2–5.

## Files

| Header | Purpose |
|---|---|
| `bench.h`        | `BenchTimer` (CUDA event wrapper), `CHECK_CU` error macro, `gflops()` helper |
| `check.h`        | `check_fp32()` AND-logic correctness verifier (abs ∧ rel tolerance) |
| `bench_driver.h` | `BenchDriver` RAII context: device alloc, host fill, warmup, timing, reference compare |

## Tolerance conventions (`check.h`)

A failure requires **both** absolute AND relative error to exceed tolerance.
Per-precision defaults used across the project:

| Precision / kernel class           | abs    | rel    |
|------------------------------------|-------:|-------:|
| FP32 scalar                        | 1e-3   | 1e-3   |
| FP16 Tensor Core (HMMA)            | 1e-2   | 1e-2   |
| `--use_fast_math` (sin/cos)        | 5e-4   | 5e-4   |
| Conv2d (9× re-accumulation)        | 1e-2   | 1e-2   |
| INT8 Tensor Core (IMMA, sym quant) | 0.5    | 0.1    |

## `BenchDriver` usage pattern

`bench_driver.h` is the post-#85 refactor target. Before: ~350 lines of
boilerplate per `bench.cu`; after: ~80 lines plus the actual launch.
The header itself documents the API; per-kernel migration into the
canonical `bench.cu` files is incremental.

## Bench-variant naming convention

Most kernel directories in phases 2–5 hold multiple bench harnesses
(per-variant comparisons, padding sweeps, persistent-kernel tests).
Project-wide convention:

| Filename pattern    | Meaning                                                       |
|---------------------|---------------------------------------------------------------|
| `bench.cu`          | Canonical primary benchmark for the dir's main kernel          |
| `bench_<variant>.cu`| Comparison driver for a kernel variant; `<variant>` is a short tag (`persistent`, `pipelined`, `v2`, `bc128`, `regpv`, etc.) |

**Don't repeat the dir name in the bench filename.** The dir is
implicit context. `kernels/gemm/hgemm/bench_persistent.cu` is correct;
`kernels/gemm/hgemm/bench_persistent_hgemm.cu` is redundant. Same logic
gives `kernels/gemm/igemm/bench_sparse.cu` (not `bench_igemm_sparse.cu`)
and `kernels/memory_layout/cymatic/bench.cu` (not `bench_cymatic.cu`).

Untracked / gitignored:

| Binary pattern  | Status |
|-----------------|--------|
| `bench`         | compiled output of `bench.cu` (gitignored) |
| `bench_*`       | compiled outputs of variant sources (gitignored) |

Deprecated names cleaned up in earlier hygiene passes (see `CHANGELOG.md`):
- `bench_refactored.cu` (BenchDriver demo files, removed once the API
  was documented in this README)
- `bench_orig` / `bench_new` / `*_orig` (stale binaries and superseded
  source variants)
- `bench_persistent_hgemm.cu`, `bench_igemm_sparse.cu`,
  `bench_cymatic.cu` (renamed to drop redundant dir-name
  prefixes).

## Cross-references

- `docs/troubleshooting.md` — common pitfalls (timer scope, async launches)
- `docs/ncu_metrics.md` — measured-roofline counter conventions
