---
title: "Replace scalar B-pack with LDSM for sparse INT8 GEMM"
labels: ["enhancement", "performance", "igemm"]
---

## Background
The sparse INT8 kernel manually packs B fragments from 8 scalar LDS.U8 loads per fragment. The dense kernel uses a single `wmma::load_matrix_sync` → `LDSM.16` hardware instruction.

Instruction count comparison (cuobjdump):
- Sparse: 160 PRMT (packing overhead)
- Dense: 64 PRMT

## Challenge
INT8 B fragment layout for `mma.sp.m16n8k32` is custom. Each thread needs 4 INT8 values packed little-endian into a 32-bit register. Standard `ldmatrix` delivers 16-bit elements, not 8-bit.

## Options

1. **LDSM + PRMT**: Load via `ldmatrix.m8n8.x2` (16-bit granularity), then PRMT to extract correct INT8 pairs. Still needs some PRMT but fewer scalar loads.

2. **Vectorized shared memory loads**: Use `LDS.U.64` (8 bytes) → unpack to 2×32-bit registers. Reduces 8 LDS.U8 → 4 LDS.32 or 2 LDS.64.

3. **Reformat B in smem during load**: Store B in shared memory already packed per-thread. Requires transposed smem layout during `LOAD_B_TILE`, but pays off in compute loop.

## Files
- `phase2/igemm/igemm_sparse_tiled.cu`

## Acceptance Criteria
- [ ] PRMT count reduced by ≥30% (160 → ≤112)
- [ ] Correctness passes
- [ ] 2048³ performance at least maintained (ideally improved)
