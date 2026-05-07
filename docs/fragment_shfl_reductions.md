# Fragment-Shfl Reductions — Eliminating smem Round-Trips in Tensor Core Kernels

> Pattern documented from issue #29 / Observation P (gpu_reflections.md).
> Delivered +40% on Flash Attention regpv kernel by replacing a 16 KB FP32 smem
> round-trip with on-fragment intra-group `__shfl_xor_sync` reductions.

## When to Apply

Use this pattern when **all** of the following hold:

- Kernel uses Tensor Core WMMA (`m16n16k16`, FP32 accumulator, sm_80+)
- Output of one mma operation is reduced (max, sum, dot, etc.) before being
  consumed by the next mma operation
- Current implementation round-trips the accumulator through smem:
  - Phase A: `wmma::store_matrix_sync(accum, smem, lda, mem_row_major)`
  - Phase B: `LDS.f32` reads from smem to compute reduction
  - Phase C: write derived values (e.g. softmax weights) back to smem
- Kernel runs at ≥ 8 warps/SM (so warp scheduler can hide the shfl latency)

Skip this pattern when:

- Reductions are across-warp rather than within-warp (smem is forced)
- The accumulator is large enough that fragment register pressure becomes
  limiting (drops occupancy worse than smem traffic)
- Existing kernel already reaches Tensor Core peak (less than 5% gain expected)
- **Outer loop has fewer than ~4 iterations** — the per-iter savings can't
  amortize the fixed overheads (Q register cache prologue, pv_accum fill,
  direct-to-global epilogue). Empirically: cross-attention with seq_kv=77
  (CLIP, 2 KV iters) loses 19% to baseline; same pattern wins 1.5×+ at
  seq_kv ≥ 256. Use selection logic if both regimes are exercised in
  production: `(seq_kv * seq_q >= 200_000) ? v2 : baseline`.

## The WMMA m16n16k16 Accumulator Layout (sm_86, verified)

For `wmma::fragment<wmma::accumulator, 16, 16, 16, float>`, each lane in a
warp holds 8 FP32 elements. The mapping (verified empirically via
`tests/flash_attention/verify_wmma_layout.cu`):

```
groupID  = lane >> 2     (0..7)   → row group
in_group = lane & 3      (0..3)   → col group within tile
row_lo   = groupID                (rows 0..7)
row_hi   = groupID + 8            (rows 8..15)
col_lo   = in_group * 2           (cols 0,2,4,6 within tile)
col_hi   = col_lo + 8             (cols 8,10,12,14 within tile)

x[0,1] → (row_lo, col_lo, col_lo+1)
x[2,3] → (row_hi, col_lo, col_lo+1)
x[4,5] → (row_lo, col_hi, col_hi+1)
x[6,7] → (row_hi, col_hi, col_hi+1)
```

Key consequence: each lane "owns" exactly 2 rows (row_lo, row_hi). Rows are
distributed across the 8 row groups; within each group, 4 lanes split the 16
cols into 4 chunks.

## Reduction Template

For an N-tile output (e.g. TILES_Bc tiles in Flash Attention's QK^T = 4 tiles
× 16 cols = 64 cols total per row), the per-row reduction proceeds:

### Step 1: per-lane partial over fragment elements

Each lane has 4 "row_lo" elements per tile (`x[0], x[1], x[4], x[5]`) and 4
"row_hi" elements per tile (`x[2], x[3], x[6], x[7]`). Reduce across all
TILES_Bc tiles locally:

```c++
float partial_lo = INIT_VALUE;  // -inf for max, 0 for sum
float partial_hi = INIT_VALUE;

#pragma unroll
for (int n = 0; n < TILES_Bc; n++) {
    partial_lo = REDUCE(partial_lo, score_frag[n].x[0]);
    partial_lo = REDUCE(partial_lo, score_frag[n].x[1]);
    partial_lo = REDUCE(partial_lo, score_frag[n].x[4]);
    partial_lo = REDUCE(partial_lo, score_frag[n].x[5]);
    partial_hi = REDUCE(partial_hi, score_frag[n].x[2]);
    partial_hi = REDUCE(partial_hi, score_frag[n].x[3]);
    partial_hi = REDUCE(partial_hi, score_frag[n].x[6]);
    partial_hi = REDUCE(partial_hi, score_frag[n].x[7]);
}
```

After this, `partial_lo` and `partial_hi` each represent the 16 col values
this lane has across all tiles for its 2 owned rows. Each lane in a row group
holds a different 1/4 of the row.

### Step 2: intra-group shfl (4 lanes per row group)

Reduce within the 4 lanes that share the same row group. Two `shfl_xor`
operations (offsets 1 and 2) cover all 4 lanes:

```c++
partial_lo = REDUCE(partial_lo, __shfl_xor_sync(0xFFFFFFFF, partial_lo, 1));
partial_lo = REDUCE(partial_lo, __shfl_xor_sync(0xFFFFFFFF, partial_lo, 2));
partial_hi = REDUCE(partial_hi, __shfl_xor_sync(0xFFFFFFFF, partial_hi, 1));
partial_hi = REDUCE(partial_hi, __shfl_xor_sync(0xFFFFFFFF, partial_hi, 2));
```

After step 2, all 4 lanes within each row group hold the same full-row
reduction value. This is the equivalent of having read the 16-element row
from smem and reduced it.

### Step 3: write derived results to smem (if needed)

If the next phase consumes the reduction result via WMMA (e.g. softmax
weights → matrix_a fragment), write FP16 values directly to smem at
WMMA-row-major positions:

```c++
__half *row_lo_ptr = warp_smem + row_lo * STRIDE + tile_col_base;
__half *row_hi_ptr = warp_smem + row_hi * STRIDE + tile_col_base;
row_lo_ptr[col_lo]     = derive(score_frag[n].x[0], partial_lo);
row_lo_ptr[col_lo + 1] = derive(score_frag[n].x[1], partial_lo);
row_lo_ptr[col_hi]     = derive(score_frag[n].x[4], partial_lo);
row_lo_ptr[col_hi + 1] = derive(score_frag[n].x[5], partial_lo);
row_hi_ptr[col_lo]     = derive(score_frag[n].x[2], partial_hi);
row_hi_ptr[col_lo + 1] = derive(score_frag[n].x[3], partial_hi);
row_hi_ptr[col_hi]     = derive(score_frag[n].x[6], partial_hi);
row_hi_ptr[col_hi + 1] = derive(score_frag[n].x[7], partial_hi);
```

Each lane writes 8 values per tile, covering exactly its owned (row, col)
positions. All 32 lanes × N tiles collectively cover the full 16×N matrix
exactly once with no overlapping writes.

### Step 4: direct-to-global output (if final stage)

For the final pv-style accumulator output, scale and write per-lane elements
direct to global memory using the same fragment-layout addressing:

```c++
float rcp_lo = __frcp_rn(running_sum_lo);
float rcp_hi = __frcp_rn(running_sum_hi);

#pragma unroll
for (int n = 0; n < TILES_D; n++) {
    int gc_lo = n * WMMA_N + col_lo;
    int gc_hi = gc_lo + 8;
    int g_row_lo = warp_q_base + row_lo;
    int g_row_hi = warp_q_base + row_hi;

    if (g_row_lo < seq_len) {
        O[(size_t)g_row_lo * D_HEAD + gc_lo]     = pv_accum[n].x[0] * rcp_lo;
        O[(size_t)g_row_lo * D_HEAD + gc_lo + 1] = pv_accum[n].x[1] * rcp_lo;
        O[(size_t)g_row_lo * D_HEAD + gc_hi]     = pv_accum[n].x[4] * rcp_lo;
        O[(size_t)g_row_lo * D_HEAD + gc_hi + 1] = pv_accum[n].x[5] * rcp_lo;
    }
    if (g_row_hi < seq_len) {
        O[(size_t)g_row_hi * D_HEAD + gc_lo]     = pv_accum[n].x[2] * rcp_hi;
        O[(size_t)g_row_hi * D_HEAD + gc_lo + 1] = pv_accum[n].x[3] * rcp_hi;
        O[(size_t)g_row_hi * D_HEAD + gc_hi]     = pv_accum[n].x[6] * rcp_hi;
        O[(size_t)g_row_hi * D_HEAD + gc_hi + 1] = pv_accum[n].x[7] * rcp_hi;
    }
}
```

## Companion: Per-Lane Owned-Row State

When the kernel maintains running statistics (max, sum, etc.) across an outer
loop, the natural representation is a per-row array `running_max[Br_WARP]`.
Because warp-wide reductions broadcast results to all 32 lanes, this array
holds 32 redundant copies of each value — wasting up to ~30 regs/thread.

**Lean alternative**: each lane stores only its 2 owned rows (`my_max_lo`,
`my_max_hi`). When the inner loop over rows needs row r's running stat,
fetch via `__shfl_sync(_, my_owned, src_lane)` where:

```c++
int src_lane = (row < 8) ? (row << 2) : ((row - 8) << 2);
float my_owned = (row < 8) ? my_max_lo : my_max_hi;
float row_running = __shfl_sync(0xFFFFFFFF, my_owned, src_lane);
```

Updates only the owning lane (groupID matches the row's group) write to the
local register — other lanes' copies remain unchanged but unused.

## Reference Implementation

`phase3/flash_attention/flash_attn_br16_v2.cu` is the canonical reference
implementation. Inspect Phase B → Phase C → Phase D for the full pattern.

## Measured Impact (Flash Attention, RTX 3070 Ti)

| metric | baseline | nosmem | delta |
|---|---|---|---|
| LDS+STS in cubin | 238 | 30 | -87% |
| smem allocation | 32 KB | 24 KB | -25% |
| seq=1024 b=8 h=8 ms | 2.45 | 1.75 | **-29%** |
| GFLOPS (effective) | 7150 | 10000 | **+40%** |
| % of FP16 TC peak | 4.1% | 5.7% | +1.6 pp |

## Generalization Targets (Issues #78, #79)

- `flash_attn_br16_pipeline.cu` — currently 64 KB / 1 block per SM, has
  largest occupancy headroom from this pattern
- `flash_attn_persistent.cu` — combines with persistent grid for steady-state
  occupancy gain
- `flash_attn_br16_bc128.cu` — Bc=128 currently crosses 50 KB cliff; nosmem
  brings it back under
- `phase4/cross_attention/*.cu` — CLIP-77 cross-attention has the same
  smem_work pattern, expect 15-25% gain (smaller because fewer KV iters)

## Why the Pattern Works

Three concurrent effects:

1. **smem traffic reduction**: FP32 store + FP32 load + FP16 store collapses
   to 0 store + 0 load + FP16 store. The eliminated FP32 traffic is the bulk
   of LDS+STS in the original kernel.

2. **Pipeline parallelism**: shfl_xor_sync executes in the warp scheduler in
   parallel with HMMA pipeline drain. smem round-trips serialize.

3. **Smem footprint reduction**: the eliminated FP32 region (16 KB → 0 KB,
   replaced by 8 KB FP16 weight overlay) frees occupancy headroom for other
   optimizations (Q register cache, K/V padding without occupancy regression).

## Pitfalls

- Verify the fragment layout on your target arch. The mapping above is
  sm_86 (RTX 30-series). sm_80, sm_89, sm_90 may differ.
- Intra-group shfl reduces over 4 lanes (offsets 1, 2). Do NOT use
  `WARP_SIZE/2` as the start offset; that's full-warp reduction and would
  incorrectly mix data from different row groups.
- The pattern requires per-lane partial computation BEFORE the shfl. If you
  start with full-warp shfl assuming all lanes hold the same value, you'll
  get garbage. Each lane begins with its own partial.
- If you also adopt the "lean per-lane owned-row state" pattern, remember
  that updates are conditional on `groupID == row` (or `row - 8` for row_hi).
  Unconditional update overwrites other rows' running stats.

## See Also

- `docs/gpu_reflections.md` Observation O (negative padding result) and P
  (positive nosmem result)
- `tests/flash_attention/verify_wmma_layout.cu` (fragment-layout probe)
- `scripts/ldmatrix_conflicts.R` (smem budget + occupancy calculator)
