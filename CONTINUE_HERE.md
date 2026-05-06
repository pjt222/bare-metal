# Continue Here

> Last updated: 2026-05-06  
> Session: Bench migration sprint — 9 files migrated, 13 total  
> Branch: main | Clean (all changes pushed)

---

## What This Session Did

### Bench Migrations to BenchDriver (13 total)

Original 5: activations, softmax, layernorm, timestep_emb, sgemm.

**This session (9 new migrations):**

| File | Lines Saved | Correctness | Performance |
|------|-------------|-------------|-------------|
| phase2/hgemm_sparse/bench.cu | 440→230 (-48%) | PASS fixed + arbitrary | 41,303 dense-equiv GFLOPS |
| phase2/igemm/bench_igemm_sparse.cu | 206→87 (-58%) | PASS (max_abs=0.00) | 39,214 dense-equiv @ 2048³ |
| phase4/groupnorm/bench.cu | ~370→155 (-58%) | PASS NHWC+NCHW | 26.6/28.3 GB/s |
| phase4/cross_attention/bench.cu | ~257→145 (-44%) | PASS sq=256,skv=77,h=8 | 705 GFLOPS |
| phase4/resblock/bench.cu | ~385→180 (-53%) | PASS default config | 142.8 GFLOPS |
| phase3/flash_attention/bench.cu | ~298→94 (-68%) | PASS seq=512,b=1,h=1 | 194.3 GFLOPS |
| phase3/flash_attention/bench_wmma.cu | ~240→115 (-52%) | PASS v1+v2 | 18.9ms vs 53.1ms |
| phase4/cross_attention/bench_pipelined.cu | ~276→120 (-57%) | PASS all 5 configs | +63% sq=256, -29% sq=4096 |
| phase3/flash_attention/bench_br16.cu | ~249→145 (-42%) | PASS v1 FP32 + v2 HMMA | 2.84ms br16 vs 19.34ms 4-warp |

### Critical Bug Fixes (Session 1 + This Session)

1. **bench.h `CHECK_CU`** — missing `}` broke ALL compilations
2. **bench_driver.h** — `cuCtxCreate` broken on CUDA 12.8 → `cuDevicePrimaryCtxRetain`
3. **18 bench files** — batch context-fixed via `scripts/fix_cuda_context.py`

### Failed & Documented

- **phase4/conv2d/bench.cu** — segfault when migrated to BenchDriver. Reverted to original. Kernel grid dims vs DeviceBuffer pointer arithmetic mismatch. Conv2d kernel may have stricter pointer layout assumptions than BenchDriver provides.

---

## Remaining Open Issues

### #67 — Remaining Bench Migrations

| Category | Count | Files |
|----------|-------|-------|
| **Migrated to BenchDriver** | 13 | activations, softmax, layernorm, timestep_emb, sgemm, hgemm_sparse, igemm_sparse, groupnorm, cross_attention, resblock, flash_attention bench, bench_wmma, bench_br16 |
| **Context-fixed only** | 12 | igemm/bench, flash_attention bench_br16_regpv/bench_fused/bench_persistent/bench_pipeline/bench_split_q/bench_bc128, conv2d bench_im2col/bench_implicit_gemm, attention_layer |
| **Unmigrated / complex** | 10 | hgemm (original, 7 variants), igemm/bench (1094 lines, 20+ kernels), conv2d bench (segfault with BenchDriver), phase5/attention_layer/bench.cu (698 lines, many cubins) |

### #29 — Tiled HGEMM Techniques → Flash Attention

Gap analyzed but not implemented: hgemm_16warp uses 16-warps, cp.async, 128×128 tiles, bank-pad. flash_attn_br16_regpv uses 4-warps, no cp.async, Br=16. Kernel-level rewrite needed — not bench-level work.

### #66 — LDSM for Sparse INT8 GEMM B-Fragment

Analyzed: `mma.sp.m16n8k32` B operand needs col-major K-rows, but `ldmatrix.m8n8.x2` distributes row-major N-cols. Fix requires smem transpose or N-major layout. Risk: may push smem over 50 KB cliff. +24.5% already recovered via metadata preload.

---

## Verified Correctness Baselines (RTX 3070 Ti Laptop)

All PASS on last run at tolerance below:

| Kernel | Size | abs_tol | rel_tol |
|--------|------|---------|---------|
| activations | 16M | 1e-4 | 1e-3 |
| softmax | 65536×32 | 1e-4 | 1e-4 |
| layernorm | 65536×32 | 1e-4 | 1e-3 |
| timestep_emb | 512×1024 | 5e-4 | 5e-4 |
| sgemm | 512³ | 1e-3 | 1e-3 |
| hgemm_sparse | 4096³ | 1e-1 | 1e-1 |
| igemm_sparse_tiled | 256³,2048³,4096³ | 0.5 | 0.1 |
| groupnorm | 4×32×64×64, G=8 | 1e-5 | 1e-4 |
| cross_attention | sq=256,skv=77,h=8 | 1e-2 | 1e-0 |
| cross_attention pipelined | 5 configs | 1e-2 | 1e-0 |
| resblock | N=1,C=64,H=32,W=32 | 1e-2×√C | 0.1 |
| flash_attention | seq=512,b=1,h=1 | 1e-3 | 1e-3 |
| flash_attention wmma | seq=1024,b=8,h=8 | 1e-3 | 1e-1 |
| flash_attention br16 | seq=1024,b=8,h=8 | 1e-2 | 1e-0 |

---

## Recommended Next Session

1. **Warmup**: `python3 scripts/verify_setup.py`
2. **Continue #67**: flash_attention/bench_br16_regpv.cu (264 lines, best result kernel, similar to bench_br16)
3. **Or #29**: Analyze tile size gap in flash_attn_br16_regpv vs hgemm_16warp — produce gap analysis document, not kernel rewrite.
4. **Use `/breathe`, `/rest`, `/meditate` between tasks**.
