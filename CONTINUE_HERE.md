# Continue Here

> Last updated: 2026-05-06  
> Session: Bench migration to BenchDriver (12 files this session)  
> Branch: main | Clean (all changes pushed)

---

## What This Session Did

### Bench Migrations to BenchDriver (12 total)

Original 5 migrated in prior sessions: activations, softmax, layernorm, timestep_emb, sgemm.

**This session (8 new migrations):**

| File | Lines Saved | Correctness | Performance Validated |
|------|-------------|-------------|----------------------|
| phase2/hgemm_sparse/bench.cu | 440→230 (-48%) | PASS (fixed + arbitrary) | 41,303 dense-equiv GFLOPS at 4096³ |
| phase2/igemm/bench_igemm_sparse.cu | 206→87 (-58%) | PASS (max_abs=0.00) | 39,214 dense-equiv at 2048³, 30,986 at 4096³ |
| phase4/groupnorm/bench.cu | ~370→155 (-58%) | PASS (NHWC + NCHW) | 26.6 GB/s (NHWC), 28.3 GB/s (NCHW) |
| phase4/cross_attention/bench.cu | ~257→145 (-44%) | PASS (seq=256,skv=77,heads=8) | 705 GFLOPS (0.4% peak) |
| phase4/resblock/bench.cu | ~385→180 (-53%) | PASS (default config) | 142.8 GFLOPS, ~4 GB/s |
| phase3/flash_attention/bench.cu | ~298→94 (-68%) | PASS (seq=512,batch=1,heads=1) | 194.3 GFLOPS |
| phase3/flash_attention/bench_wmma.cu | ~240→115 (-52%) | PASS (v1 scalar + v2 4-warp) | 18.9ms vs 53.1ms at batch=8,heads=8,seq=1024 |
| phase4/cross_attention/bench_pipelined.cu | ~276→120 (-57%) | PASS (all 5 configs) | +63% at sq=256, -29% at sq=4096 (smem pressure) |

### Bug Fixes During Migration

1. **bench.h `CHECK_CU` macro**: missing `}` for `if`-block broke all compilations. Fixed.
2. **bench_driver.h CUDA 12.8**: `cuCtxCreate` broken (redirects to v4). Replaced with `cuDevicePrimaryCtxRetain` + `cuCtxSetCurrent`.
3. **18 bench files**: batch-fixed context API via `scripts/fix_cuda_context.py` for CUDA 12.8 compatibility.

### Failed & Reverted

- **phase4/conv2d/bench.cu**: segfault when migrated to BenchDriver. Reverted to original (already context-fixed). Root cause: kernel grid dims mismatch with DeviceBuffer pointer arithmetic. Needs investigation — different 3D grid layout may violate kernel assumptions.

---

## Remaining Open Issues

### Ready to Tackle (Medium Effort)

**#67 — Migrate remaining bench files to BenchDriver**
- Easiest remaining: flash_attention/bench_br16, bench_br16_regpv (similar to done work)
- Complex: flash_attention bench_split_q, bench_pipeline (multiple kernels, unique args)
- Skip for now: igemm/bench.cu (1094 lines, 20+ kernels), conv2d/bench.cu (segfault with BenchDriver)

**#29 — Apply tiled HGEMM techniques to Flash Attention QK^T and PV**
- Gap identified: hgemm_16warp uses 16-warps, cp.async double-buffer, 128×128 tiles, bank-pad
- flash_attn_br16_regpv uses 4-warps (128 threads), no cp.async, Br=16 small tiles
- Key insight: 4× fewer warps → 4× less instruction-level parallelism. Bc=64 means fewer KV tiles reused.
- Would require kernel rewrite: not a bench-level optimization.

### Needs Planning (High Effort)

**#66 — Replace scalar B-pack with LDSM for sparse INT8 GEMM**
- Analyzed but not implemented. Current: 160 PRMT per block. Target: ≤112 (≥30% reduction).
- Root cause: `mma.sp` B operand needs col-major K-rows. `ldmatrix.m8n8.x2` distributes row-major N-cols.
- Fix requires smem transpose or N-major layout. Buffers: smem would need 31→~50 KB (risk of smem cliff).

### Exploration / Low Priority

- **#32** — Polyhedral spring networks
- **#18** — 128×256 tiles for online-quant (register pressure)
- **#17** — smem padding for ldmatrix bank conflicts
- **#14** — Tutorial series
- **#7** — 2:4 sparsity with IMMA (done via #65, #66 is fine-tuning)
- **#4** — Fuse GroupNorm into Conv2d epilogue

---

## Bench Migration Status

| Category | Count | Files |
|----------|-------|-------|
| **Migrated to BenchDriver** | 13 | activations, softmax, layernorm, timestep_emb, sgemm, hgemm_sparse, igemm_sparse, groupnorm, cross_attention, resblock, flash_attention bench, flash_attention bench_wmma, cross_attention bench_pipelined |
| **Context-fixed only** | 12 | igemm/bench, flash_attention bench_br16/br16_regpv/fused/persistent/pipeline/split_q/bc128, conv2d bench_im2col/bench_implicit_gemm, attention_layer |
| **Unmigrated / complex** | 10 | hgemm (original, 7 variants), igemm/bench (1094 lines), conv2d bench (segfault with BenchDriver), phase5/attention_layer/bench.cu (698 lines, many cubins) |

---

## Verified Correctness Baselines (RTX 3070 Ti Laptop)

All PASS at tolerance below on last run:

| Kernel | Size | abs_tol | rel_tol | Status |
|--------|------|---------|---------|--------|
| activations | 16M | 1e-4 | 1e-3 | PASS |
| softmax | 65536×32 | 1e-4 | 1e-4 | PASS |
| layernorm | 65536×32 | 1e-4 | 1e-3 | PASS |
| timestep_emb | 512×1024 | 5e-4 | 5e-4 | PASS |
| sgemm | 512³ | 1e-3 | 1e-3 | PASS |
| hgemm_sparse (fixed + arbitrary) | 4096³ | 1e-1 | 1e-1 | PASS |
| igemm_sparse_tiled (after #65) | 256³, 2048³, 4096³ | 0.5 | 0.1 | PASS (max_abs=0.00) |
| groupnorm | 4×32×64×64, G=8 | 1e-5 | 1e-4 | PASS |
| cross_attention | seq=256,skv=77,heads=8 | 1e-2 | 1.0 | PASS |
| cross_attention pipelined | 5 configs (SD 8×8 to long ctx) | 1e-2 | 1.0 | PASS |
| resblock | N=1,C=64,H=32,W=32,G=8 | 1e-2×√C | 0.1 | PASS |
| flash_attention | seq=512,batch=1,heads=1 | 1e-3 | 1e-3 | PASS |
| flash_attention wmma | seq=1024,batch=8,heads=8 | 1e-3 | 1e-1 | PASS |

---

## Recommended Next Session

1. **Warmup**: run `python3 scripts/verify_setup.py`
2. **Pick one medium issue**:
   - **#67**: migrate `flash_attention/bench_br16.cu` to BenchDriver (similar to bench_wmma)
   - **#67**: migrate `flash_attention/bench_br16_regpv.cu` (best result kernel, high value)
3. **Use `/breathe`, `/rest`, `/meditate` between tasks** as instructed
