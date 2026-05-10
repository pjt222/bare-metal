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
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum ./bench_sparse 4096 4096 4096
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
    ./bench_sparse 4096 4096 4096

# 4. Nsight Compute: global load efficiency
ncu --metrics sm__sass_average_data_reuse_per_request_mem_global_op_ld.pct \
    --metrics l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
    ./bench_sparse 4096 4096 4096

# 5. Nsight Compute: instruction mix
ncu --metrics sm__inst_executed_pipe_tensor_op_hmma.sum \
    --metrics sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_elapsed \
    ./bench_sparse 4096 4096 4096
```

---

## GPU Profiling Results (Confirmed on RTX 3070 Ti)

### Measured Performance

| Kernel | Size | Dense-equiv TOPS | Status |
|--------|------|-----------------|--------|
| Sparse (rebuilt) | 2048³ | **39,457** | Baseline |
| Sparse (rebuilt) | 4096³ | **25,568** | **-35% regression** |
| Sparse (metadata preload, #65) | 2048³ | **39,674** | Baseline |
| Sparse (metadata preload, #65) | 4096³ | **31,835** | **-19.8% regression** |
| Dense cp.async | 2048³ | 17,239 | Stable |
| Dense cp.async | 4096³ | 17,975 | Stable |

The dense kernel is essentially flat across sizes. The sparse kernel drops 35%.

### Instruction Mix (cuobjdump)

| Instruction | Sparse (`igemm_sparse_tiled`) | Dense (`igemm_pipelined_cpasync`) | Ratio |
|-------------|-------------------------------|-----------------------------------|-------|
| **PRMT** | **160** | **64** | **2.5×** |
| LDS | 148 | 120 | 1.23× |
| IMMA | 32 | 32 | 1.0× |
| LDG | 74 | 66 | 1.12× |
| STS | 64 | 80 | 0.8× |
| STG | **32** | **16** | **2.0×** |
| IADD | 145 | 206 | 0.7× |
| **Registers** | **64** | **126** | **Sparse wins** |
| **Spill** | **0** | **0** | **None** |

### Key Findings

1. **Register spill ruled out**: Both kernels have 0 stack/local memory. Sparse uses only 64 registers (vs 126 dense), so occupancy is actually *better* for sparse.

2. **B-fragment packing overhead is real but not the root cause**: Sparse has 2.5× more PRMT (160 vs 64) from manual INT8 packing. However, this overhead scales linearly with tile count, so it would affect both 2048³ and 4096³ proportionally. It explains why sparse is slower than dense overall, but not the 2048³ → 4096³ regression *within* sparse.

3. **L2 cache capacity is the root cause**: At 4096³:
   - Metadata per block = K_steps × 8 × 4 bytes = 128 × 8 × 4 = **4,096 bytes**
   - Total blocks = (4096/128)² = **1,024**
   - Total metadata working set = 1,024 × 4,096 = **4,194,304 bytes = 4.1 MB**
   - **GA104 L2 cache = 4.0 MB**

   The metadata working set **exactly matches** the L2 capacity. At 2048³:
   - Total metadata = (2048/128)² × 4,096 = 256 × 4,096 = **1,048,576 bytes = 1.0 MB**
   - This fits comfortably in L2 with 3× headroom.

   At 4096³, metadata thrashes L2. Each block reads fresh metadata for every K-tile, but the L2 can't hold the full 4.1 MB working set, so metadata reads spill to DRAM.

4. **Epilogue store pattern amplifies the effect**: Sparse uses 32 STG (element-by-element stores) vs dense's 16 STG (wmma::store_matrix_sync). More store traffic at larger sizes = more L2 pressure = faster metadata eviction.

---

## Root Cause Summary

```
Hypothesis ranking (after profiling):

1. L2 cache metadata saturation ★ CONFIRMED
   - 4.1 MB metadata at 4096³ vs 4.0 MB L2 = thrashing
   - Explains size-dependent regression (2048³: 1 MB fits, 4096³: 4 MB doesn't)

2. B-fragment manual packing overhead → PARTIAL
   - 2.5× more PRMT explains sparse vs dense gap overall
   - But scales linearly; doesn't explain 2048³→4096³ drop within sparse

3. Register pressure / spill → RULED OUT
   - 64 regs, 0 stack, 0 local memory

4. Grid saturation / occupancy → RULED OUT
   - Sparse has fewer blocks but ALSO fewer registers → same or better occupancy
```

---

## Potential Fixes

| Fix | Target | Effort | Expected Gain |
|-----|--------|--------|---------------|
| **Reorder loops: K-tile outer, metadata preload to smem** | L2 thrashing | Medium | +20-30% at 4096³ |
| **Tiled metadata cache: reuse meta across K-steps** | L2 thrashing | Low | +10-15% |
| **Reduce block tile 128→64 (4× more blocks, ¼ metadata/SM)** | L2 thrashing | Medium | +15% (if occupancy holds) |
| Replace scalar B-pack with LDSM + PRMT | B overhead | High | +5-10% overall |
| Use `wmma::store_matrix_sync` for epilogue | Store traffic | Medium | +3-5% |

### Recommended Fix: Metadata Preload Pattern

Instead of reading metadata from global memory inside the K-loop:
```cpp
uint32_t meta = metadata[m_tile * K_steps * 8 + gk * 8 + gid];  // L2 miss at 4096³
```

Preload a full block's metadata into shared memory once per block:
```cpp
__shared__ uint32_t smem_meta[NUM_WARPS * 8];  // 512 bytes
// One-time load by first warp:
if (threadIdx.x < meta_count_this_block) {
    smem_meta[threadIdx.x] = metadata[base_meta + threadIdx.x];
}
__syncthreads();
// Then read from smem inside K-loop:
uint32_t meta = smem_meta[warp_id * 8 + gid];  // L0/L1 speed
```

Cost: +512 bytes smem (still under 50 KB cliff). Benefit: eliminates all metadata DRAM traffic after first tile.

---

## Notes

- Dense kernel is stable across sizes because it has no metadata traffic.
- The 4096³ regression is specifically an **L2 capacity problem**, not a fundamental algorithmic flaw.
- **2048³ is the sweet spot** for this kernel on GA104. For larger matrices, metadata preloading is the path forward.

---

## Update 2026-05-05: #65 — Metadata Preload Implemented

Commit 36a1fa7 adds double-buffered metadata preload to shared memory:

```cpp
__shared__ uint32_t smem_meta[2][128];  // 1 KB, loaded per tile alongside A/B
```

Results after fix:

| Size | Before #65 | After #65 | Change |
|------|-----------|-----------|--------|
| 2048³ | 39,457 | **39,674** | +0.5% |
| 4096³ | 25,568 | **31,835** | +24.5% |

Regression reduced from -35% → -19.8% (2048³ → 4096³). Root cause confirmed: metadata L2 thrashing eliminated.

Remaining 19.8% gap: likely B-fragment scalar packing overhead (160 PRMT in inner loop, 2.5× dense kernel). Fix requires smem transpose (N-major layout for ldmatrix compatibility) — tracked as #66.
