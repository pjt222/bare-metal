# Chapter 05 — Flash Attention

> The longest chapter in this series, because Flash Attention is the kernel
> where every other chapter's lessons compound. Read chapter 02 (GEMM) first.

Flash Attention is the de-facto attention algorithm for transformer
inference and training since its publication (Dao et al., 2022). The idea
is simple — avoid materializing the full N×N score matrix in DRAM by
recomputing softmax in-place across tiled passes — but the implementation
touches every concern in this book: Tensor Core fragment layouts, smem
cliffs, register-resident accumulators, online reductions, and pipelining.

This chapter walks through nine versions of the kernel. Each version
exposes a specific bottleneck that motivates the next. By the end, the
implementation reaches **11.5 TFLOPS at seq_len=1024 on GA104** — about 6.6%
of FP16 Tensor Core peak. That sounds low, and it is; but it represents a
1.86× speedup over a from-scratch HMMA baseline and a ~50× speedup over a
scalar implementation.

## The problem

Given query, key, value matrices `Q, K, V ∈ ℝ^(N × d)` (per head),
attention computes:

```
scores  = Q · K^T  / sqrt(d)        (N × N)
weights = softmax(scores)            (N × N, row-wise)
output  = weights · V                (N × d)
```

For a single head with `N = 1024, d = 64`, the score matrix is 4 MB. With
16 heads in batch=8, that is 512 MB of scratch — nearly half the entire
GA104 DRAM bandwidth budget per pass — to compute something we discard.
Materializing the score matrix is the bottleneck, not the FLOPs.

## Version 1 — Scalar (`flash_attn.cu`)

**Setup**: one thread block processes one query row at a time. Each row
streams over all KV positions, maintaining a running `(max, sum, output)`
state via the online softmax recurrence:

```
new_max = max(old_max, current_score)
rescale = exp(old_max - new_max)
new_sum = old_sum * rescale + exp(current_score - new_max)
output  = output * rescale + exp(current_score - new_max) * V_row
```

This recurrence is mathematically identical to the standard softmax: the
per-row max is incrementally updated, and previously accumulated outputs
are rescaled when the max changes.

**Result**: ~50 ms at seq=1024 b=8 h=8. Functional, slow. Every score
multiply-add is one FFMA. No Tensor Cores. About 0.3% of FP16 TC peak.

**Lesson**: the algorithm is correct, but per-row work is serialized inside
a warp. Need to process Q tiles in parallel, with Tensor Cores doing the
QK^T and PV matmuls.

## Version 2 — Br=16 HMMA (`flash_attn_br16.cu`)

**Setup**: each warp owns 16 query rows (`Br = 16`). Each block has 4 warps
covering 64 query rows. The kernel iterates over KV tiles of `Bc = 64`
columns, doing:

- Phase A: cooperative load of `K_tile, V_tile` from DRAM into smem
- Phase B: `score_tile = Q · K_tile^T` via WMMA m16n16k16 HMMA
- Phase C: row-wise online softmax using SHFL.BFLY warp reductions
- Phase D: accumulate `output += weight_tile · V_tile` via WMMA HMMA

The key fragment-layout fact: in `wmma::fragment<accumulator, 16, 16, 16, float>`
on sm_86, lane `L` owns 8 fp32 elements covering exactly **2 rows** of the
16x16 accumulator. With `groupID = L >> 2` (0..7), lane L holds rows
`groupID` (the "lo" row) and `groupID + 8` (the "hi" row).

**Smem layout** (48 KB):
- Q_tile [Br_block × d]    = 64 × 64 × 2 = 8 KB
- K_tile [Bc × d]          = 64 × 64 × 2 = 8 KB
- V_tile [Bc × d]          = 64 × 64 × 2 = 8 KB
- score_smem [Br × Bc]     = 64 × 64 × 4 = 16 KB (FP32)
- weight_smem [Br × Bc]    = 64 × 64 × 2 = 8 KB

Total 48 KB → 2 blocks/SM (8 warps).

**Result**: 2.81 ms, 6112 GFLOPS at seq=1024 b=8 h=8. About 18× the scalar
version. About 3.5% of TC peak.

**Lesson**: the score round-trip through smem (Phase B writes 16 KB, Phase
C reads it back) and the weight round-trip (Phase C writes 8 KB, Phase D
reads it) become the dominant smem traffic. The compute is right but the
data motion is heavy.

## Version 3 — Register-resident PV accumulator (`flash_attn_br16_regpv.cu`)

**Setup**: keep `pv_accum` (the running output) as a WMMA accumulator
fragment in registers across all KV iterations, instead of staging through
smem. Per-row rescaling at each KV tile boundary is done via fragment
element addressing — lane `L` rescales its owned `(row_lo, row_hi)`
elements directly.

**Smem now** (32 KB, no Q tile in smem since Q can be loaded on-the-fly,
no PV smem since pv_accum is register-resident):
- K_tile [Bc × d]      = 8 KB
- V_tile [Bc × d]      = 8 KB
- smem_work [Br × Bc]  = 16 KB (FP32 score round-trip, kept for now)

3 blocks/SM (12 warps) — crossed the favorable side of the 50 KB cliff
upward.

**Result**: 2.45 ms, 7150 GFLOPS at seq=1024 b=8 h=8. **+17%** over v2.

**Lesson**: every round-trip through smem that could be avoided is worth
avoiding. The 16 KB PV smem we removed was costing more than its bookkeeping
suggested, because removing it freed enough space to enable 3 blocks/SM
(was 2). One change improved both data motion AND occupancy.

This kernel was the canonical "good" Flash Attention for some time, and
became the baseline against which all subsequent work was measured.

## Version 4 (failed) — Padded smem (`flash_attn_br16_regpv_pad.cu`)

**Setup**: WMMA `ldmatrix.x4.trans` loads from K_tile (col_major in WMMA's
view) hit 8-way bank conflicts when `stride_bytes mod 32 == 0`. The
classical fix: pad each row by +8 halfs, breaking the stride alignment.

**Smem** grows from 32 KB to 35 KB. Crucial: 35 KB × 3 = 105 KB > 100 KB
cap, so 3 blocks/SM no longer fit. Drops to 2 blocks/SM (8 warps).

**Result**: -20% to -32% across all sizes. Lost everywhere.

**Lesson**: this is **Observation O** in the project notes. At 12 warps/SM,
the warp scheduler hides 8-way LDSM replays via overlapping warps. The
1 KB padding "fix" addresses a non-bottleneck and pays for itself with an
occupancy drop. The right tool was not padding — it was eliminating the
smem allocation entirely (next version).

## Versions 5-7 — Three-stage refactor (Issue #29)

The defining work of the project's optimization pass. Three distinct
techniques applied in sequence, each enabling the next.

### Stage 1 — Lean per-thread softmax state (`flash_attn_br16_regpv_lean.cu`)

**Setup**: in v3, every thread held `running_max[16]` and `running_sum[16]`
arrays — full broadcast-identical copies across all 32 lanes. That's 32
redundant FP32 registers per thread.

Per the WMMA fragment layout (lane L owns 2 rows: `row_lo = L >> 2`,
`row_hi = (L >> 2) + 8`), each lane only *needs* 4 of those 32 floats:
its own `(max_lo, max_hi, sum_lo, sum_hi)`. The 4 lanes within each row
group already hold the same row's data; intra-group `__shfl_sync` can
broadcast on demand.

**Result**: registers drop 156 → 132. Performance neutral standalone.
But this enables...

### Stage 2 — Q register cache (`flash_attn_br16_regpv_lean_qcache.cu`)

**Setup**: examining the SASS revealed Q was being reloaded from L2 every
KV iteration despite being loop-invariant. The compiler's auto-CSE wasn't
sufficient. Explicit `q_frag[TILES_D]` declared outside the KV loop, loaded
once.

Per-thread regs grow 132 → 144 (the q_frag fragments). Still fits at
3 blocks/SM (144 × 128 × 3 = 55296 < 65536).

**Result**: +4-13% across sizes. LDG count inside loop drops 30 → 14.

**Lesson**: SASS-level inspection beats high-level expectations. The C++
compiler does many things, but it does not always hoist invariants you
assume it will hoist.

### Stage 3 — smem_work elimination (`flash_attn_br16_v2.cu`, the winner)

**Setup**: this is the biggest single optimization in the entire project.

The 16 KB FP32 `smem_work` buffer round-tripped scores between Phase B
(QK^T HMMA writes) and Phase C (row-wise softmax reads). In the cubin,
LDS+STS instructions totalled 238 — most of them this round-trip.

The fix is a structural one. In the WMMA m16n16k16 fragment layout, the
4 lanes within each row group collectively hold all 16 columns of one
row of the score tile (lane 0 holds cols 0,1,8,9; lane 1 holds 2,3,10,11;
lane 2 holds 4,5,12,13; lane 3 holds 6,7,14,15). A row-wise reduction over
all 16 cols is achievable by:

1. Each lane computes its 4-col partial max/sum within its owned elements
2. `__shfl_xor_sync(_, _, 1)` combines pairs of lanes (cols 0-3 with 4-7)
3. `__shfl_xor_sync(_, _, 2)` combines pairs of pairs (cols 0-7 with 8-15)

After two shuffles, all 4 lanes in a row group hold the full per-row max
or sum. No smem store, no smem load.

The output FP16 weights are written direct-to-smem at WMMA-row-major
positions for Phase D. The final pv_accum is scaled by 1/sum and written
direct-to-global, no smem staging.

**Smem** drops 32 KB → 24 KB. Still 3 blocks/SM, but with 8 KB margin.

**Cubin**: LDS+STS drops **238 → 30**. 87% reduction in smem traffic.

**Result**: 1.75 ms, 9998 GFLOPS at seq=1024 b=8 h=8. **+40%** over the
v3 regpv baseline. **1.40×** cumulative speedup over the start of the
refactor.

**Pattern documented** in `docs/fragment_shfl_reductions.md`. The technique
generalizes to any softmax-like, layer-norm-like, or group-norm-like
reduction in a Tensor Core kernel.

**Lesson**: when smem is the bottleneck, eliminating an allocation beats
optimizing its layout. The structural fix is always more durable than the
local fix.

## Version 8 — Pipelined cp.async (`flash_attn_br16_v2_pipeline.cu`)

**Setup**: with v2's 24 KB smem footprint, doubling K and V buffers for
cp.async double-buffering brings smem to 40 KB. Just under the 50 KB cliff
→ 2 blocks/SM (8 warps).

The pipeline:
- Outer iteration `i` issues cp.async for K[i+1], V[i+1] tiles
- Compute on K[i], V[i] (already in smem)
- `cp.async.wait_group 1` before next iteration

**Result**: 1.529 ms, 11453 GFLOPS at seq=1024 b=8 h=8. **+14% over v2**.
**1.60× cumulative** over the regpv baseline (verified post-warmup;
earlier "1.86×" claim was based on an extrapolation, since corrected).

**Crucial context**: the original `flash_attn_br16_pipeline.cu` (built on
v2's predecessor) had **lost 5%**. Same cp.async machinery, but at 1
block/SM (4 warps), there were not enough warps to schedule the in-flight
loads. **Same optimization, opposite result, depending on occupancy.**

This is **Observation Q**. Equivalent observations exist for many other
techniques. A growing rule of thumb: when changing anything that affects
occupancy, re-run your catalog of "established" wins — their status may
have flipped.

**Lesson**: cp.async is not a universally-good optimization. It is an
overlap mechanism, and overlap requires warps available to schedule. Below
8 warps, cp.async machinery costs more than it saves.

## Version 9 (regime-dependent) — Bc=128 with v2's smem savings

**Setup**: with smem reduced 32 → 24 KB at Bc=64, doubling to Bc=128 fits
in 48 KB → 2 blocks/SM. Hypothesis: the bigger inner-K tile means 2× HMMA
per phase, better Tensor Core pipeline depth, and 2× fewer outer iterations
(less prologue overhead).

**Results**:

| seq | b | h | v2 (Bc=64) | v2_bc128 (Bc=128) | Δ |
|---|---|---|---|---|---|
| 512  | 16 | 16 | 1.747 ms | 1.888 ms | **-7.4%** |
| 1024 | 8  | 8  | 1.757 ms | 1.875 ms | **-6.3%** |
| 2048 | 4  | 8  | 3.455 ms | 3.669 ms | **-5.8%** |
| 4096 | 2  | 8  | 7.370 ms | **7.255 ms** | **+1.6%** |

**Lesson**: this is **Observation S**. The result is *regime-dependent*, not
uniformly negative. At small/medium seq, the occupancy hit (12 → 8 warps)
dominates: the warp scheduler runs out of slack to hide K/V load latency,
and the bigger tile cannot recover. At seq=4096, the iteration count is
high enough (32 outer iters at Bc=128 vs 64 at Bc=64) that halving the
outer-iter count amortizes prologue and sync overhead enough to overcome
the occupancy hit. Crossover lies between seq=2048 and seq=4096.

At the *same* 2 blocks/SM occupancy, v2_pipeline beats Bc=128 at all sizes
(+19-23%). The right tool when crossing an occupancy boundary is still
cp.async (overlap), not a bigger tile. But for very-large-seq workloads
the Bc=128 variant is not a strictly losing option — it is a small-win
dispatch alternative.

This is the same regime-dependent pattern as cross-attention's CLIP-77
split: an optimization that wins in one regime and loses in another. The
lesson is to *measure across all expected workload sizes* before declaring
an optimization a win or a loss.

## Performance summary

| variant | smem | regs | blocks/SM | warps/SM | ms (seq=1024 b=8 h=8) | GFLOPS | speedup |
|---|---|---|---|---|---|---|---|
| flash_attn (scalar) | — | — | — | — | ~50 | ~350 | 1.0× |
| flash_attn_br16 | 48 KB | 124 | 2 | 8 | 2.81 | 6112 | ~14× |
| flash_attn_br16_regpv | 32 KB | 156 | 3 | 12 | 2.45 | 7150 | 18× |
| flash_attn_br16_regpv_pad | 35 KB | 156 | 2 | 8 | 3.04 | 5768 | 14× (regression) |
| flash_attn_br16_regpv_lean | 32 KB | 132 | 3 | 12 | 2.52 | 6940 | 17.5× |
| flash_attn_br16_regpv_lean_qcache | 32 KB | 144 | 3 | 12 | 2.37 | 7400 | 19× |
| **flash_attn_br16_v2** (smem_work eliminated) | 24 KB | 138 | 3 | 12 | **1.75** | **9998** | **25×** |
| flash_attn_br16_v2_bc128 | 48 KB | 187 | 2 | 8 | 1.875 / **7.255** at seq=4096 | 9341 / 9657 | regime-dependent |
| **flash_attn_br16_v2_pipeline** (cp.async) | 40 KB | 140 | 2 | 8 | **1.529** | **11453** | **29×** |

The two bold rows are the canonical kernels: v2 for the structural
optimization, v2_pipeline for the additional cp.async overlap on top.

## What this chapter teaches

If you are writing a Tensor Core kernel that round-trips data through
smem for a per-row reduction, **the round-trip is probably eliminable**.
The fragment-shfl pattern (`docs/fragment_shfl_reductions.md`) is a
direct, mechanical replacement.

If you are tempted to add smem padding to break bank conflicts, **measure
the occupancy impact first**. At 12 warps/SM, the conflicts are likely
already hidden. A 1 KB padding may cost you 4 warps in exchange for fixing
a non-problem.

If you are tempted to enlarge a tile because "bigger is better",
**measure the occupancy impact first** (same lesson, different
manifestation). Bigger tiles only win within a fixed occupancy regime.
Crossing an occupancy boundary with a bigger tile usually loses.

If you have an optimization that lost in version N, **re-test it in
version N+1**. cp.async lost in the original FA pipeline, won in
v2_pipeline. Persistent grids won in some kernels at 8 warps, broke even
at 12 warps. Optimizations are occupancy-dependent — re-run the catalog.

## How to run it yourself

```bash
cd /mnt/d/dev/p/bare-metal/phase3/flash_attention

# Build all variants
nvcc --cubin -arch=sm_86 -O2 -o flash_br16_v2.sm_86.cubin flash_attn_br16_v2.cu
nvcc --cubin -arch=sm_86 -O2 -o flash_br16_v2_pipeline.sm_86.cubin flash_attn_br16_v2_pipeline.cu
nvcc --cubin -arch=sm_86 -O2 -o flash_v2_persistent.sm_86.cubin flash_attn_v2_persistent.cu
nvcc --cubin -arch=sm_86 -O2 -o flash_br16_v2_bc128.sm_86.cubin flash_attn_br16_v2_bc128.cu

# Build bench
nvcc -arch=sm_86 -O2 -o bench_v2_variants bench_v2_variants.cu -lcuda -I../../phase2/common

# Run twice (first run cold, second run is the steady-state number)
./bench_v2_variants 1024 8 8
./bench_v2_variants 1024 8 8
```

Expected steady-state (RTX 3070 Ti):
- v2 baseline: ~1.75 ms, 10000 GFLOPS
- v2 pipeline: ~1.53 ms, 11500 GFLOPS
- v2 persistent: ~1.90 ms, 9200 GFLOPS (loses at 12 warps)
- v2 Bc=128: ~1.88 ms, 9300 GFLOPS at seq=1024 (loses); slight win at seq=4096

## Source files

- `phase3/flash_attention/flash_attn.cu` (scalar, version 1)
- `phase3/flash_attention/flash_attn_br16.cu` (Br=16 HMMA, version 2)
- `phase3/flash_attention/flash_attn_br16_regpv.cu` (regpv, version 3)
- `phase3/flash_attention/flash_attn_br16_regpv_pad.cu` (failed padding)
- `phase3/flash_attention/flash_attn_br16_regpv_lean.cu` (stage 1)
- `phase3/flash_attention/flash_attn_br16_regpv_lean_qcache.cu` (stage 2)
- `phase3/flash_attention/flash_attn_br16_v2.cu` (**canonical**, smem elim)
- `phase3/flash_attention/flash_attn_br16_v2_pipeline.cu` (**best**, +cp.async)
- `phase3/flash_attention/flash_attn_br16_v2_bc128.cu` (failed bigger tile)

## Cross-references

- Pattern: `docs/fragment_shfl_reductions.md`
- Postmortems: `docs/gpu_reflections.md` Insights 1, 6, 14; Observations O, P, Q, S
- Issues: #29, #78, #79, #80 (closed), #81, #82
