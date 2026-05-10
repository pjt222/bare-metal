# kernels/gemm/hgemm_sparse — 2:4 structured sparse HGEMM (`mma.sp`)

FP16 GEMM where every contiguous 4-element group of operand A has at
most 2 non-zeros. Encoded via NVIDIA's 2:4 sparsity primitive
(`mma.sp.sync.aligned.m16n8k16` PTX, `HMMA.16816.SP` SASS). Throughput
matches dense HGEMM at 2048³, regresses at 4096³ (see
[Obs N](../../docs/gpu_reflections.md), [Obs HH](../../docs/gpu_reflections.md)).

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

| Shape  | Dense HGEMM | 2:4 Sparse | Sparse / dense |
|--------|------------:|-----------:|---------------:|
| 2048³  |  31.9 TFLOPS|  31.9 TFLOPS | 1.00× (dense-equiv after #65 metadata preload) |
| 4096³  |  31.9 TFLOPS|  ~26 TFLOPS | 0.81× (regression — see Obs HH) |

## Cross-references

- [Obs N](../../docs/gpu_reflections.md) — full development arc, naive → Tensor Core path
- [Obs HH](../../docs/gpu_reflections.md) — IMMA stall hand-tunes do not reproduce on CUDA 13.2 (sub-task A)
- [docs/int8_sparse_4096_regression_analysis.md](../../docs/int8_sparse_4096_regression_analysis.md) — regression hypotheses (companion analysis for INT8 path; same pattern)
- [docs/tutorial/03-int8-tensor-cores.md](../../docs/tutorial/03-int8-tensor-cores.md) — sparsity walkthrough
