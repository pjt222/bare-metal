---
title: "Investigate INT8 sparse GEMM regression at 4096³ (46% drop)"
labels: ["bug", "performance", "igemm"]
---

## Problem
`igemm_sparse_tiled.cu` drops from 39,745 dense-equiv TOPS at 2048³ to **21,593 TOPS at 4096³** — a 46% regression. Non-sparse `igemm_pipelined_cpasync.cu` does NOT show this drop (20,688 TOPS at 4096³).

## Data
| Kernel | Size | TOPS | Notes |
|--------|------|------|-------|
| igemm_sparse_tiled | 2048³ | **39,745** | Expected performance |
| igemm_sparse_tiled | 4096³ | **21,593** | **46% regression** |
| igemm_pipelined_cpasync | 4096³ | 20,688 | Baseline (no regression) |

## Hypotheses
1. **Register spill** — sparse metadata handling adds register pressure. At 4096³, grid size increases, changing occupancy calculus.
2. **L2 cache thrashing** — 2:4 sparse metadata pattern causes irregular access, L2 hit rate degrades at larger problem sizes.
3. **Tile iteration overhead** — `K=4096 / BK=32 = 128` iterations. Metadata recompute per iteration dominates at large K.
4. **Threadblock divergence** — sparse `mma.sp` has structural constraints that create idle warps in some configurations.

## Diagnostic Commands
```bash
# Register usage
nvcc --cubin -arch=sm_86 -O2 -o igemm_sparse_tiled.sm_86.cubin kernels/gemm/igemm/igemm_sparse_tiled.cu
cuobjdump -res-usage igemm_sparse_tiled.sm_86.cubin

# Profile with Nsight Compute (if available)
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_local \
    --metrics sm__sass_average_data_reuse_per_request_mem_global_op_ld \
    ./bench_sparse 4096 4096 4096

# Check for local memory (spill)
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum \
    ./bench_sparse 4096 4096 4096
```

## Related
- `CONTINUE_HERE.md` notes: "Register spill suspected. Not investigated."
- `kernels/gemm/igemm/README.md` documents non-sparse IGEMM performance but not sparse variant.

## Files
- `kernels/gemm/igemm/igemm_sparse_tiled.cu`
- `kernels/gemm/igemm/bench_sparse.cu`
- `kernels/gemm/igemm/sparse_meta_int8.h`

## Acceptance Criteria
- [ ] Root cause identified (profile data or SASS inspection)
- [ ] Fix implemented OR documented as architectural limitation
- [ ] Performance at 4096³ within 20% of 2048³, OR limitation clearly explained
- [ ] Update `kernels/gemm/igemm/README.md` with sparse variant results

## Effort
Medium — requires profiling and potentially SASS-level analysis.
