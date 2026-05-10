---
title: "Migrate remaining 23 bench files to BenchDriver"
labels: ["refactoring", "bench-driver"]
---

## Background
PR #53 introduced `kernels/_common/bench_driver.h` and refactored 3 pilot files:
- `kernels/gemm/hgemm/bench_refactored.cu` (348 → 96 lines)
- `kernels/gemm/igemm/bench_refactored.cu` (1093 → 79 lines)
- `kernels/attention/flash_attention/bench_refactored.cu` (298 → 122 lines)

Total reduction: 1,739 → 297 lines (-83%).

## Remaining Files to Migrate

### Phase 2
- [ ] `kernels/gemm/sgemm/bench.cu`
- [ ] `kernels/reductions/softmax/bench.cu`
- [ ] `kernels/reductions/layernorm/bench.cu`
- [ ] `phase2/activations/bench.cu`
- [ ] `kernels/gemm/igemm/bench.cu`
- [ ] `kernels/gemm/igemm/bench_sparse.cu`
- [ ] `kernels/gemm/hgemm_sparse/bench.cu`

### Phase 3
- [ ] `kernels/attention/flash_attention/bench.cu`
- [ ] `kernels/attention/flash_attention/bench_bc128.cu`
- [ ] `kernels/attention/flash_attention/bench_br16.cu`
- [ ] `kernels/attention/flash_attention/bench_br16_regpv.cu`
- [ ] `kernels/attention/flash_attention/bench_fused.cu`
- [ ] `kernels/attention/flash_attention/bench_persistent.cu`
- [ ] `kernels/attention/flash_attention/bench_pipeline.cu`
- [ ] `kernels/attention/flash_attention/bench_split_q.cu`
- [ ] `kernels/attention/flash_attention/bench_wmma.cu`

### Phase 4
- [ ] `phase4/conv2d/bench.cu`
- [ ] `phase4/conv2d/bench_im2col.cu`
- [ ] `phase4/conv2d/bench_implicit_gemm.cu`
- [ ] `kernels/attention/cross_attention/bench.cu`
- [ ] `kernels/attention/cross_attention/bench_pipelined.cu`
- [ ] `kernels/reductions/groupnorm/bench.cu`
- [ ] `phase4/resblock/bench.cu`
- [ ] `phase4/timestep_emb/bench.cu`

### Phase 5
- [ ] `phase5/attention_layer/bench.cu`

## Approach
1. Start with simplest (softmax, activations, timestep_emb)
2. Then GEMM variants (sgemm, hgemm_sparse bench)
3. Then attention/cross-attention
4. Then conv/groupnorm/resblock (most complex host setup)

## Acceptance Criteria
- [ ] All migrated files compile and run correctly
- [ ] Performance numbers match or exceed pre-migration baselines
- [ ] Original bench files kept as `.bak` during migration, removed after verification
- [ ] `bench_driver.h` extended if any kernel needs new driver features

## Notes
- Don't delete original files until new ones are verified
- Some bench files load multiple kernels (e.g., hgemm bench loads 6 variants) — driver variant loop handles this
