# kernels/gemm/hgemm_sparse — 2:4 structured sparse HGEMM (`mma.sp`)

FP16 GEMM where every contiguous 4-element group of operand A has at
most 2 non-zeros. Encoded via NVIDIA's 2:4 sparsity primitive
(`mma.sp.sync.aligned.m16n8k16` PTX, `HMMA.16816.SP` SASS).
Dense-equivalent throughput is **~1.27× dense HGEMM** (matched-clock,
same-session — see [Headline](#headline-rtx-3070-ti-laptop-sm_86) below),
the structured-sparsity win (development arc:
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
Re-measured 2026-06-03 (#140), `bench.cu` Dense-eq column:

| Shape  | Dense HGEMM (16-warp) | Sparse 2:4 (dense-eq) |
|--------|----------------------:|----------------------:|
| 2048³  | 31,948 GFLOPS @ 1.77 GHz | 40,153 GFLOPS @ 1.41 GHz |
| 4096³  | 32,327 GFLOPS @ 1.67 GHz | 41,080 GFLOPS @ 1.69 GHz |

At **4096³** both kernels are long-running and power-bound at the 150 W
cap, at near-identical SM clock (~1.68 GHz) → a fair same-power
comparison: sparse-eq / dense = 41,080 / 32,327 = **1.27×**. This is the
clock-robust headline — sparse dense-equivalent throughput is ~1.27×
dense, between the 1× floor and 2× ceiling of 2:4 sparsity. (The 2048³
rows are at *mismatched* clocks — sparse ran un-boosted at 1.41 GHz,
dense at 1.77 GHz — so their raw ratio understates the sparse advantage;
treat 4096³ as canonical.)

> The earlier "31.9 TFLOPS / dense-parity" wording was a **category
> error**: 31.9 TFLOPS is the *dense* HGEMM baseline (the frozen
> reference line `bench.cu` prints), not the sparse result. On a
> dense-equivalent basis the sparse kernel **beats** dense; it does not
> merely match it.
>
> Absolute sparse numbers above are native-boost and power-bound
> (bimodal on the 150 W laptop, like `igemm_sparse_tiled`). A stable
> locked-clock absolute and the 2048³-vs-4096³ per-clock regression
> (old "0.81×" claim — not reproducible at native boost) need an
> elevated `nvidia-smi.exe -lgc` re-measure →
> [#143](https://github.com/pjt222/bare-metal/issues/143).

## Cross-references

- [Obs N](../../../docs/gpu_reflections.md) — full development arc, naive → Tensor Core path
- [Obs HH](../../../docs/gpu_reflections.md) — IMMA stall hand-tunes do not reproduce on CUDA 13.2 (sub-task A)
- [docs/int8_sparse_4096_regression_analysis.md](../../../docs/int8_sparse_4096_regression_analysis.md) — regression hypotheses (companion analysis for INT8 path; same pattern)
- [docs/tutorial/03-int8-tensor-cores.md](../../../docs/tutorial/03-int8-tensor-cores.md) — sparsity walkthrough
