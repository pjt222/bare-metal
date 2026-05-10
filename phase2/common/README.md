# phase2/common — shared benchmark headers

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
boilerplate per `bench.cu`; after: ~80 lines, plus the actual launch.
Migrated benchmarks: see `bench_refactored.cu` in `phase2/hgemm`,
`phase2/igemm`, `phase3/flash_attention`. Not yet promoted to the
canonical `bench.cu` — see audit Tier 1 follow-up.

## Cross-references

- `docs/troubleshooting.md` — common pitfalls (timer scope, async launches)
- `docs/ncu_metrics.md` — measured-roofline counter conventions
