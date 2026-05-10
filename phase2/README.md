# Phase 2 — ML primitives in SASS

Five kernel families covering the compute backbone of every transformer
and diffusion model: GEMM (FP32, FP16, INT8, sparse INT8), reductions
(softmax, layernorm), and elementwise activations. Each subdir is a
self-contained study with its own `README.md`.

## Subdirectory index

| Dir | Kernel | Headline |
|---|---|---|
| [`sgemm/`](sgemm/)               | naive / tiled / register-blocked SGEMM (FP32) | tiled tops 1.0 TFLOPS at 2048³ |
| [`hgemm/`](hgemm/)               | tiled FP16 HGEMM via WMMA / HMMA.16816 | 31.9 TFLOPS at 2048³ (#65 baseline) |
| [`hgemm_sparse/`](hgemm_sparse/) | 2:4 structured sparse HGEMM via `mma.sp` | dense-equivalent throughput at 2048³, see [Obs N](../docs/gpu_reflections.md) |
| [`igemm/`](igemm/)               | INT8 IMMA + 2:4 sparse INT8 + cp.async pipelining | 39.7 TOPS sparse 2048³, [Obs HH](../docs/gpu_reflections.md) on the 4096³ regression |
| [`softmax/`](softmax/)           | warp-reduce softmax via `SHFL.BFLY` + `MUFU.EX2` |  — |
| [`layernorm/`](layernorm/)       | block-reduce normalization (`SHFL.BFLY` + `MUFU.RSQ`) |  — |
| [`activations/`](activations/)   | ReLU, GELU, Swish; `MUFU.EX2` via fast_math |  — |
| [`common/`](common/)             | shared headers (`bench.h`, `check.h`, `bench_driver.h`) |  — |

## Build

Phase-wide build via the top-level `Makefile`:
```bash
make phase2          # build all phase2 cubins + benches
```

Per-kernel build instructions live in each subdir's README.

## Cross-references

- Headline numbers and rooflines: [`docs/kernels.md`](../docs/kernels.md)
- Optimization postmortems: [`docs/gpu_reflections.md`](../docs/gpu_reflections.md)
- SASS reference: [`docs/ampere_sass_reference.md`](../docs/ampere_sass_reference.md)
- Tutorial walkthroughs: [`docs/tutorial/02-gemm-from-scratch.md`](../docs/tutorial/02-gemm-from-scratch.md), [`03-int8-tensor-cores.md`](../docs/tutorial/03-int8-tensor-cores.md)
