# Continue Here

> Last updated: 2026-03-29T22:00:00Z | Branch: main

## Objective

Build hand-optimized CUDA/SASS kernels for ML inference on RTX 3070 Ti (GA104, sm_86). The project progresses from basic primitives through full pipeline integration. The immediate next task is **CuAssembler hand-tuning of the tiled INT8 IGEMM inner loop** (GitHub issue #2).

## Completed

- [x] **Git repo initialized**, pushed to https://github.com/pjt222/bare-metal
- [x] **CLAUDE.md** created with build commands, architecture, conventions, Four Laws
- [x] **running_sum bug fix** — `running_sum[row] *= rescale_factor` was missing in 4 kernels (br16, bc128, pipeline, cross_attn). 1000× correctness improvement (max_abs 1.88e-02 → 2.19e-05). The correct online softmax recurrence is `l = l * exp(m_old - m_new) + sum(exp(s - m_new))`
- [x] **Split-Q Flash Attention** — `phase3/flash_attention/flash_attn_split_q.cu` + `bench_split_q.cu`. Two-kernel pipeline (main + reduce). Roughly tied with br16 at splits=4 (1.05×). Partial buffer overhead prevents larger gains
- [x] **INT8 IMMA GEMM** — 3 kernels in `phase2/igemm/`:
  - `igemm.cu` (naive): 10,897 TOPS at 4096³, IMMA.16816.S8.S8 confirmed
  - `igemm_tiled.cu` (64×64 smem): **15,145 TOPS** — best, 1.39× over naive
  - `igemm_register_blocked.cu` (128×128): 12,760 TOPS — 0.84× of tiled (longer inner loop hurts at 8 warps/SM)
- [x] **End-to-end attention layer pipeline** — `phase5/attention_layer/bench.cu` + `utils.cu`. Chains LayerNorm → 3×HGEMM → transpose → Flash Attention → transpose → HGEMM → residual add. **6.75 ms total at batch=8 seq=1024**. Flash Attention dominates at large batch (42%), format conversions dominate at small batch (46%). Issue #1 closed
- [x] **.gitignore fix** — `bench_*` was excluding `.cu` source files; added `!bench_*.cu` negation. Removed accidentally tracked `phase1/host` binary

## In Progress

- [ ] Nothing partially complete — all work committed and pushed

## Next Steps

1. ~~**CuAssembler hand-tuning of tiled IGEMM** (GitHub issue #2)~~ — **DONE.** IMMA is NOT S08-constrained like HMMA. Compiler S04→S02 gives +1.6%. S01 is correct but too aggressive at scale. See `phase2/igemm/README.md`.
2. **Persistent kernel grid** (GitHub issue #3) — for small-batch Flash Attention where many SMs are idle
3. Consider updating `docs/gpu_reflections.md` with the Phase 6 findings (running_sum bug, split-Q postmortem, IGEMM register-blocking lesson, IMMA vs HMMA pipeline finding)

## Context

### The GA104 Optimization Pattern (confirmed across 4 experiments)
On GA104 with 8 warps/SM, **shorter inner loops beat higher per-warp compute density**:
- cp.async: 4-5% slower (warp interleaving already hides latency)
- Bc=128 Flash Attention: 17-20% slower (occupancy cliff at >64 KB smem)
- Split-Q: partial buffer I/O overhead wipes out KV DRAM savings
- WMMA register-blocking 128×128: 16-mma_sync inner loop too long for 8 warps

This is the project's central finding. The 64 KB smem cliff and 8-warp minimum are the load-bearing constraints on GA104.

### Pipeline Correctness Note
The end-to-end pipeline has 0.4% element failures at abs=0.5 tolerance. This is FP16 precision loss amplified by softmax — small score differences shift attention weights. It's structural (inherent to mixed FP16/FP32), not a bug.

### CuAssembler for Issue #2
- The CuAssembler roundtrip workflow: `build.py compile` → `build.py disasm` → hand-edit `.cuasm` → `build.py assemble`
- Phase 5 Step 4 found HMMA S08 stalls are hardware-constrained on Ampere TC pipeline
- IMMA may behave similarly — this is what issue #2 tests
- Key files: `scripts/build.py`, `tools/CuAssembler/`, `phase2/igemm/igemm_tiled.cu`
- The `.cuasm` files from Phase 5: `phase3/flash_attention/flash_br16.cuasm`, `phase4/conv2d/wmma_gemm.cuasm`

### Reading Order for Next Session
1. This file
2. `CLAUDE.md` — project conventions, build commands, Four Laws
3. `phase2/igemm/igemm_tiled.cu` — the kernel to hand-tune
4. `phase2/igemm/README.md` — current results table
5. GitHub issue #2 — acceptance criteria
6. `docs/ampere_sass_reference.md` — IMMA.16816.S8.S8 instruction reference

### Key Benchmark Results

| Kernel | Config | Result |
|--------|--------|--------|
| Flash Attention br16 | seq=1024 batch=8 heads=8 | 2.81 ms, 6,112 GFLOPS |
| IGEMM tiled (compiler) | 4096³ | 15,078 TOPS (2.2% of 696 TOPS peak) |
| IGEMM tiled (hand-tuned S02) | 4096³ | 15,320 TOPS (+1.6% vs compiler) |
| IGEMM register-blocked | 4096³ | 12,760 TOPS (0.84× tiled — lesson learned) |
| Attention pipeline | batch=8 seq=1024 | 6.75 ms total, Flash Attn 42% |
| Conv2d implicit GEMM | 64×64 Cin=Cout=320 | 25× over direct conv |
| HGEMM | 4096³ | 7,853 GFLOPS (4.5% of FP16 peak) |
