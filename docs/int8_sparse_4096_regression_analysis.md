# INT8 Sparse GEMM 4096³ Regression Analysis

> **Status:** Hypotheses documented. GPU profiling required to confirm root cause.
> **Issue:** [#41](https://github.com/pjt222/bare-metal/issues/41)

## Observed Regression

| Size | Sparse TOPS | Dense TOPS |
|------|-------------|------------|
| 2048³ | **39,745** | ~20,688 |
| 4096³ | **21,593** (~46% drop) | ~20,688 (stable) |

The non-sparse `igemm_pipelined_cpasync` is stable across sizes. The sparse `igemm_sparse_tiled` collapses at 4096³.

---

## Structural Differences: Sparse vs Dense

| Property | Sparse (`igemm_sparse_tiled`) | Dense (`igemm_pipelined_cpasync`) |
|----------|-------------------------------|-----------------------------------|
| Block size | 512 threads (16 warps) | 128 threads (4 warps) |
| Tile | 128×128 | 64×64 |
| K-step | BK=64, WMMA_K=32 | BK=32, WMMA_K=16 |
| Blocks at 4096³ | 32×32 = 1,024 | 64×64 = 4,096 |
| Accumulators | 32 INT32 (left+right sub-tiles) | 4 WMMA accumulators |
| B loading | 8 scalar LDS.U8 + manual pack | `wmma::load_matrix_sync` (hardware) |
| Metadata traffic | ~1.28 GB at 4096³ | 0 |
| Inline asm | `mma.sp` with global meta read | WMMA API |

---

## Hypotheses (Ranked by Likelihood)

### 1. Metadata Bandwidth Bottleneck (Most Likely)

`mma.sp` requires 32-bit metadata per sub-tile per K-step. Indexed via:
```cpp
uint32_t meta = metadata[
    m_tile_idx * K_steps_total * 8 + gk_idx * 8 + gid];
```

At 4096³:
- K_steps_total = 4096/32 = 128
- Metadata per block = (128/16) × (128/16) × 128 × 8 × 4 bytes = 1.25 MB
- Total metadata traffic = 1,024 blocks × 1.25 MB = **1.28 GB**

This is *additional* traffic not present in the dense kernel. At 608 GB/s peak, 1.28 GB adds ~2.1 ms overhead. But metadata access is **random** (indexed by `m_tile_idx * K_steps_total * 8 + gk_idx * 8 + gid`), defeating L2 cache coalescing.

**To confirm:** Profile with `ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`

### 2. B-Fragment Packing Overhead

The sparse kernel manually packs B fragments from 8 scalar INT8 loads:
```cpp
fb_left0 = (uint8_t)p[0]
         | (uint8_t)p[STRIDE_B]     << 8
         | (uint8_t)p[STRIDE_B * 2] << 16
         | (uint8_t)p[STRIDE_B * 3] << 24;
```

Per fragment: 4 LDS.U8 + 3 shifts + 3 ORs. Two fragments per sub-tile × 2 sub-tiles × 2 K-steps = **32 scalar loads per warp per K-tile**.

The dense kernel uses a single `wmma::load_matrix_sync` → hardware LDSM.16 instruction.

At 4096³ (128 K-tiles), the scalar packing overhead accumulates linearly.

**To confirm:** Disassemble both kernels and count LDS instructions in the inner loop.
```bash
cuobjdump -sass igemm_sparse_tiled.sm_86.cubin | grep -c 'LDS'
cuobjdump -sass igemm_pipelined_cpasync.sm_86.cubin | grep -c 'LDS'
```

### 3. Register Pressure / Spill

| Sparse Register Budget | Count |
|------------------------|-------|
| acc_left[2][2][4] | 16 |
| acc_right[2][2][4] | 16 |
| fa0, fa1 | 2 |
| fb_left0, fb_left1, fb_right0, fb_right1 | 4 |
| Metadata index temps | ~4 |
| Loop / address registers | ~8 |
| **Total estimated** | **~50** |

50 registers × 512 threads = 25,600 per block. 64K limit → 2 blocks/SM possible. But if the compiler spills due to the inline asm constraints, local memory traffic spikes.

**To confirm:**
```bash
nvcc --cubin -arch=sm_86 -O2 -o igemm_sparse_tiled.sm_86.cubin igemm_sparse_tiled.cu
cuobjdump -res-usage igemm_sparse_tiled.sm_86.cubin
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum ./bench_igemm_sparse 4096 4096 4096
```

### 4. Grid Size / SM Saturation

At 4096³:
- Sparse: 1,024 blocks across 48 SMs = 21.3 blocks/SM
- Dense: 4,096 blocks across 48 SMs = 85.3 blocks/SM

The sparse kernel's larger tiles mean fewer total blocks. At peak, only ~21 blocks can be distributed. If metadata reads cause early blocks to stall, the SM scheduler has fewer alternative warps to switch to.

This is an **occupancy-distribution** problem, not a raw thread-count problem.

---

## Profiling Commands

Run these on the RTX 3070 Ti to identify the root cause:

```bash
# 1. Register usage and occupancy
nvcc --cubin -arch=sm_86 -O2 -o igemm_sparse_tiled.sm_86.cubin \
     phase2/igemm/igemm_sparse_tiled.cu
cuobjdump -res-usage igemm_sparse_tiled.sm_86.cubin

# 2. SASS analysis: LDS count, IMMA count
 cuobjdump -sass igemm_sparse_tiled.sm_86.cubin | grep -E 'LDS|IMMA|LDGSTS' | sort | uniq -c

# 3. Nsight Compute: local memory (spill)
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum \
    ./bench_igemm_sparse 4096 4096 4096

# 4. Nsight Compute: global load efficiency
ncu --metrics sm__sass_average_data_reuse_per_request_mem_global_op_ld.pct \
    --metrics l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
    ./bench_igemm_sparse 4096 4096 4096

# 5. Nsight Compute: instruction mix
ncu --metrics sm__inst_executed_pipe_tensor_op_hmma.sum \
    --metrics sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_elapsed \
    ./bench_igemm_sparse 4096 4096 4096
```

---

## Potential Fixes (Pending Profiling Confirmation)

| Fix | Target Hypothesis | Effort |
|-----|-------------------|--------|
| Preload metadata to smem (1 KB/tile) | Metadata bandwidth | Medium |
| Replace scalar B-pack with LDSM + PRMT | B-fragment overhead | High |
| Reduce tile to 64×64 (more blocks) | Grid saturation | Medium |
| Use `wmma::mma_sync` instead of inline asm | Register pressure | High |

---

## Notes

- The 2048³ performance (39,745 TOPS) proves the kernel design is fundamentally correct.
- The regression is **size-dependent**, suggesting a bandwidth or occupancy scaling issue rather than a correctness bug.
- L2 cache size (4 MB) vs metadata working set may be the crossover point: at 2048³, metadata fits better in L2.
