# kernels/gemm/hgemm_sparse — 2:4 structured sparse HGEMM (`mma.sp`)

FP16 GEMM where every contiguous 4-element group of operand A has at
most 2 non-zeros. Encoded via NVIDIA's 2:4 sparsity primitive
(`mma.sp.sync.aligned.m16n8k16` PTX, `HMMA.16816.SP` SASS).
Dense-equivalent throughput is **~1.33× dense HGEMM** (clock-locked
1605 MHz, 4096³ — see [Headline](#headline-rtx-3070-ti-laptop-sm_86)
below), the structured-sparsity win (development arc:
[Obs N](../../../docs/gpu_reflections.md), [Obs HH](../../../docs/gpu_reflections.md)).

## Files

### Kernels

| File | Status | Notes |
|---|---|---|
| `hgemm_sparse_naive.cu`   | working | Scalar-load reference for correctness |
| `hgemm_sparse_tiled.cu`   | working | Tiled WMMA + `mma.sp` Tensor Core path; **canonical** |
| `sparse_meta.h`           | header  | 2:4 metadata pack/unpack helpers (E0–E3 nibble layout) |

### Verification harnesses (kept for reproducibility, not on hot path)

| File | What it verifies |
|---|---|
| `test_dense_manual.cu`        | Hand-laid WMMA fragment writes match cuBLAS dense GEMM (sanity) |
| `test_mma_sp.cu`              | `mma.sp` PTX intrinsic produces correct rows for known sparsity pattern |
| `verify_wmma_ab_layout.cu`    | A/B fragment layout matches PTX ISA spec (lane → matrix-element mapping) |

### Bench

`bench.cu` — dense vs sparse vs theoretical peak comparison.

## Build

```bash
nvcc -arch=sm_86 -O2 --cubin hgemm_sparse_tiled.cu -o hgemm_sparse_tiled.sm_86.cubin
nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../common
./bench 2048 2048 2048
```

## Headline (RTX 3070 Ti Laptop, sm_86)

2:4 sparsity skips half of A's K-elements, so **dense-equivalent**
throughput (the FLOP count the same problem would cost as dense work)
can exceed the dense HGEMM baseline — that excess is the sparsity win.
Measured under a host-side clock lock (`nvidia-smi.exe -lgc 1605,1605`,
elevated Windows shell — the only regime free of the 150 W power-cap
bimodal; [#143](https://github.com/pjt222/bare-metal/issues/143)),
`bench.cu` Dense-eq column, every cell steady at 1605 MHz:

| Shape  | Dense HGEMM (16-warp) | Sparse 2:4 (dense-eq) | Sparse-eq / dense |
|--------|----------------------:|----------------------:|------------------:|
| 2048³  | 29,259 GFLOPS | 40,980 GFLOPS | 1.40× |
| 4096³  | 31,886 GFLOPS | 42,257 GFLOPS | **1.33×** |

**Canonical: ~1.33× over dense at 4096³** (= 133%, confirming the
long-standing **131%** in [Obs N](../../../docs/gpu_reflections.md)) —
sparse dense-equivalent throughput sits between the 1× floor and 2×
ceiling of 2:4 sparsity. The 2048³ ratio (1.40×) is higher only because
the 1605 lock sits below dense's native boost: dense HGEMM is clock-bound
and is suppressed more by the lock than the power-bound sparse kernel, so
treat 4096³ as canonical, not 2048³. Locked figures are stable to <0.5%
across re-runs (vs ~1.9× bimodal at native boost — the power-cap artifact
documented for `igemm_sparse_tiled`).

> The earlier "31.9 TFLOPS / dense-parity" wording was a **category
> error**: 31.9 TFLOPS is the *dense* HGEMM baseline (the frozen
> reference line `bench.cu` prints), not the sparse result — and the
> locked dense 4096³ (31,886) confirms that literal. On a
> dense-equivalent basis the sparse kernel **beats** dense by ~1.33×; it
> does not merely match.
>
> The old "4096³ regresses to ~26 TFLOPS / 0.81×" claim is **refuted**:
> at matched 1605 MHz, sparse 4096³ (42,257) is at parity-or-above 2048³
> (40,980), not a 19% drop. The apparent regression was native-boost
> power-cap noise, not a size effect.

## Cross-references

- [Obs N](../../../docs/gpu_reflections.md) — full development arc, naive → Tensor Core path
- [Obs HH](../../../docs/gpu_reflections.md) — IMMA stall hand-tunes do not reproduce on CUDA 13.2 (sub-task A)
- [docs/int8_sparse_4096_regression_analysis.md](../../../docs/int8_sparse_4096_regression_analysis.md) — regression hypotheses (companion analysis for INT8 path; same pattern)
- [docs/tutorial/03-int8-tensor-cores.md](../../../docs/tutorial/03-int8-tensor-cores.md) — sparsity walkthrough
