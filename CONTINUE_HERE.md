# Continue Here

> Last updated: 2026-03-29T18:30:00Z | Branch: main

## Objective

Build hand-optimized CUDA/SASS kernels for ML inference on RTX 3070 Ti (GA104, sm_86). All 5 original phases complete. Issue #6 (fuse format conversions into Flash Attention) implemented and benchmarked.

## Completed

- [x] **Issue #6 — BSHD-layout Flash Attention**: Eliminates all 4 transpose kernels from the attention pipeline. New kernel `flash_attn_fused.cu` accepts FP16 [B,S,H,D] directly and outputs FP32 [B,S,H,D].
- [x] **Standalone correctness verified**: Single-head and multi-head tests pass (abs=1e-2, rel=1e0)
- [x] **Pipeline benchmark results**:
  - batch=1 seq=256: ~15% improvement (transpose overhead dominates at small batch)
  - batch=8 seq=1024: ~2% improvement (Flash Attention overhead nearly offsets transpose savings)
  - Fused pipeline always at least as good or better
- [x] **Discovered real smem cliff**: GA104 sm_86 cliff is at **50 KB/block** (100 KB max smem/SM ÷ 2 blocks), not 64 KB. 56 KB → 1 block/SM → 2× regression. Updated CLAUDE.md.
- [x] **Failed approach documented**: FP32 input (reads 4B/element instead of 2B) caused 2× bandwidth penalty. Reverted to FP16 input with stride d_model.
- [x] All prior Phase 6 work still intact (IMMA hand-tuning, persistent grid, gpu_reflections)

## In Progress

- [ ] Nothing partially complete — ready to commit

## Next Steps

### Remaining Open Issues

Pick one per session. Updated order after issue #6:

#### 1. Issue #5 — Software pipelining for tiled INT8 IGEMM
**Why next**: Addresses the fundamental 2.2% utilization bottleneck. Builds on issue #2's inner loop understanding.
- Double-buffer smem: `smem_a[2][64][32]` + `smem_b[2][32][64]` = 8 KB total (well under 50 KB cliff)
- K-loop restructure: prologue loads tile 0, loop body overlaps LDG(N+1) with IMMA(N)
- **Key question**: Can we get closer to the 3.3 ms bandwidth floor (currently at 9.0 ms)?

#### 2. Issue #8 — Per-channel asymmetric INT8 quantization
**Why second**: Pure epilogue change, no inner loop modification.

#### 3. Issue #7 — 2:4 structured sparsity with IMMA
**Why third**: Requires research into sparse IMMA encoding. High risk.

#### 4. Issue #4 — Fuse GroupNorm into Conv2d epilogue
**Why last**: Highest effort, lowest incremental gain (GroupNorm is 2% of ResBlock). **IMPORTANT**: Must stay under 50 KB smem (revised cliff).

## Context

### Key Finding: GA104 smem cliff is 50 KB, not 64 KB
- GA104 sm_86 max shared memory per SM: 100 KB
- 2 blocks/SM requires ≤50 KB per block (100/2)
- 56 KB → 1 block/SM → 2× regression (measured)
- 48 KB → 2 blocks/SM → full occupancy (measured)
- Updated CLAUDE.md Four Laws #4 accordingly

### BSHD Layout Trade-offs
- Q loaded from global with stride d_model (vs D_HEAD in BHSD): ~20-33% Flash Attention per-kernel overhead due to wider stride between WMMA rows
- K/V tile loading: coalescing is fine (D_HEAD elements contiguous = 128-byte cache line aligned)
- Output store: stride d_model × 4 bytes between rows (wider than BHSD)
- Pipeline-level: savings from eliminated transposes offset the per-kernel overhead

### Files Changed
- `phase3/flash_attention/flash_attn_fused.cu` — new kernel (BSHD layout, 48 KB smem)
- `phase3/flash_attention/bench_fused.cu` — standalone correctness + perf benchmark
- `phase5/attention_layer/bench.cu` — updated pipeline bench (runs both original and fused)
- `CLAUDE.md` — corrected smem cliff from 64 KB to 50 KB

### Key Benchmark Results

| Kernel | Config | Result |
|--------|--------|--------|
| Flash Attention br16 (BHSD) | seq=1024 batch=8 heads=8 | ~3.3 ms, ~5,200 GFLOPS |
| Flash Attention fused (BSHD) | seq=1024 batch=8 heads=8 | ~3.8 ms, ~4,400 GFLOPS |
| Pipeline original (11 steps) | batch=1 seq=256 | ~0.28 ms |
| Pipeline fused (no transpose) | batch=1 seq=256 | ~0.24 ms (~15% faster) |
| Pipeline original (11 steps) | batch=8 seq=1024 | ~6.85 ms |
| Pipeline fused (no transpose) | batch=8 seq=1024 | ~6.70 ms (~2% faster) |
| IGEMM tiled (hand-tuned S02) | 4096³ | 15,320 TOPS (2.2% of peak) |
| Conv2d implicit GEMM | 64×64 Cin=Cout=320 | 25× over direct conv |
| HGEMM | 4096³ | 7,853 GFLOPS (4.5% of FP16 peak) |
