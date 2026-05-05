---
title: "Implement metadata preloading to smem for sparse INT8 GEMM"
labels: ["enhancement", "performance", "igemm"]
---

## Background
GPU profiling (session 2026-05-05, see `docs/int8_sparse_4096_regression_analysis.md`) confirmed that `igemm_sparse_tiled.cu` drops 35% at 4096³ because metadata thrashes L2 cache.

**The fix**: Preload a full block's metadata into shared memory once at block start, then read from smem inside the K-loop.

## Current (slow)
```cpp
uint32_t meta = metadata[m_tile * K_steps * 8 + gk * 8 + gid];  // global read per K-step
```

## Proposed
```cpp
__shared__ uint32_t smem_meta[NUM_WARPS * 8];  // 512 bytes
// Load once:
if (threadIdx.x < meta_count) {
    smem_meta[threadIdx.x] = metadata[base + threadIdx.x];
}
__syncthreads();
// Then inside K-loop:
uint32_t meta = smem_meta[warp_id * 8 + gid];  // L0/L1 speed
```

## Impact
- Extra smem: 512 bytes (total: 30.5 KB → still under 50 KB cliff)
- Eliminates all metadata DRAM traffic after first tile
- Expected gain: +20-30% at 4096³ (from 25,568 → ~33,000 dense-equiv GFLOPS)

## Files
- `phase2/igemm/igemm_sparse_tiled.cu`
- `phase2/igemm/sparse_meta_int8.h`

## Acceptance Criteria
- [ ] Metadata preload implemented
- [ ] Correctness passes at 256³, 2048³, 4096³
- [ ] 4096³ performance within 10% of 2048³ (or improvement documented)
- [ ] Update `docs/int8_sparse_4096_regression_analysis.md`
