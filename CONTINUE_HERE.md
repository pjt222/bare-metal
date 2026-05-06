# Continue Here

> Last updated: 2026-05-05  
> Session: GitHub issues cleanup & kernel optimization  
> Branch: main | Clean (all changes pushed)

---

## What This Session Did

### Issues Closed (8)

| Issue | Title | Key Result |
|-------|-------|------------|
| #55-64 | 10 duplicate issues | Marked duplicate of already-closed #35-44 |
| #68 | Track hand-tuned binary | `git add igemm_tiled_handtuned.sm_86.cubin` |
| #33 | Optimize sparse HGEMM >60% dense | Verified 41,721 dense-equiv GFLOPS at 4096³ (131% of dense) |
| #69 | Benchmark CI pre-push hook | `.githooks/pre-push` + `scripts/install-hooks.sh` |
| #65 | Metadata preload to smem (INT8) | `smem_meta[2][128]` double-buffered, +24.5% at 4096³ |
| #44 | README consolidation | Updated all 3 READMEs with validated numbers |

### Bugs Fixed Along the Way

1. **`phase2/common/bench.h`** — `CHECK_CU` macro was missing closing `}` for `if`-block. Broke ALL compilations. Fixed.
2. **`phase2/common/bench_driver.h`** — `cuCtxCreate` broken on CUDA 12.8 (redirects to `cuCtxCreate_v4`). Replaced with `cuDevicePrimaryCtxRetain` + `cuCtxSetCurrent`.
3. **`Makefile`** — `make test` tried to build `bench_refactored.cu` files that lack build rules. Fixed dependency.
4. **18 bench files** — Batch-fixed CUDA 12.8 context API via `scripts/fix_cuda_context.py`. Files in phase2-5.

### Bench Migration Progress (#67)

| Status | Files |
|--------|-------|
| Migrated to BenchDriver (5) | activations, softmax, layernorm, timestep_emb, sgemm |
| Context-fixed only (18) | igemm/bench, flash_attention/*, conv2d/*, cross_attention/*, groupnorm, resblock, attention_layer |
| Still broken/unmigrated (19) | hgemm (kept original, 7 variants), hgemm_sparse (complex pattern), igemm/bench_igemm_sparse, flash_attention bench variants with unique args, phase4/5 complex multi-kernel benches |

### Performance State (Validated)

| Kernel | Size | Result |
|--------|------|--------|
| HGEMM dense 16-warp | 4096³ | 31,910 GFLOPS |
| HGEMM sparse tiled | 4096³ | 41,721 dense-equiv GFLOPS |
| IGEMM dense 128×256 | 4096³ | 27,591 TOPS |
| IGEMM sparse tiled (after #65) | 2048³ | 39,674 dense-equiv TOPS |
| IGEMM sparse tiled (after #65) | 4096³ | 31,835 dense-equiv TOPS (-19.8% vs 2048³) |

---

## Remaining Open Issues

### Ready to Tackle (Medium Effort)

**#67 — Migrate remaining bench files to BenchDriver**
- Easiest next targets (simple element-wise/bandwidth kernels): hgemm_sparse/bench, igemm/bench_igemm_sparse
- Complex ones to skip for now: flash_attention variants (unique arg parsing), phase4 multi-kernel benches
- `scripts/fix_cuda_context.py` already fixes the context bug on any new file

**#29 — Apply tiled HGEMM techniques to Flash Attention QK^T and PV**
- Well-scoped: fuse the HGEMM tiling architecture into existing Flash Attention kernels
- phase3/flash_attention kernels already have FP16 HMMA; the gap is tiling efficiency

### Needs Planning (High Effort)

**#66 — Replace scalar B-pack with LDSM for sparse INT8 GEMM**
- ANALYZED but NOT IMPLEMENTED. Correctness verified, SASS inspected.
- **Root cause:** `mma.sp.m16n8k32` B operand needs same N-column across K-rows (col-major). `ldmatrix.m8n8.x2` distributes adjacent N-cols per thread (row-major). Layouts are orthogonal.
- **Fixes explored:**
  1. cp.async scatter to N-major smem (per-thread scatter during global→smem)
  2. Post-load cooperative transpose (512 threads, 8 KB, per-tile)
  3. Custom swizzle layout for smem_B
- all require restructuring `phase2/igemm/igemm_sparse_tiled.cu` beyond quick fix
- Current workaround: metadata preload (#65) recovered +24.5%. Remaining 19.8% gap = B-fragment overhead.

### Exploration / Low Priority

- **#32** — Polyhedral spring networks (research, no deadline)
- **#18** — 128×256 tiles for online-quant (register pressure investigation)
- **#17** — smem padding for ldmatrix bank conflicts (no observed conflicts)
- **#14** — Tutorial series (write last)
- **#7** — 2:4 sparsity with IMMA (substantially done via #65, #66 is fine-tuning)
- **#4** — Fuse GroupNorm into Conv2d epilogue

---

## Critical Files & Commands

### Rebuild any kernel
```bash
cd phaseN/<kernel_dir>
nvcc --cubin -arch=sm_86 -O2 -o kernel.sm_86.cubin kernel.cu
nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../common  # phase2
nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../../phase2/common  # phase3-5
```

### Fix CUDA 12.8 context on new bench file
```bash
python3 scripts/fix_cuda_context.py   # auto-detects and patches all bench*.cu
```

### Pre-push hook
```bash
bash scripts/install-hooks.sh   # one-time install
git push --no-verify            # bypass when needed
```

### Key docs
- `docs/gpu_reflections.md` — 24 empirical hardware insights, contains all benchmark numbers
- `docs/int8_sparse_4096_regression_analysis.md` — #65/#66 context, L2 thrashing root cause
- `CLAUDE.md` — project constraints (never modify system CUDA, 50 KB smem cliff, etc.)

---

## Known Gotchas

1. **`make test`**: only builds PHASE2_BENCH. Phase3-5 benches have unique build rules that `make` may not catch. Build individually.
2. **`bench_refactored.cu` files**: exist as pilots but DON'T swap them in blindly — `hgemm/bench_refactored.cu` had `hgemm_16warp` FAIL bug (grid config mismatch). Migrate carefully.
3. **`cuCtxCreate` on CUDA 12.8**: silently breaks. Always use `cuDevicePrimaryCtxRetain` + `cuCtxSetCurrent`.
4. **SMEM cliff**: GA104 limit is 50 KB/block for 2 blocks/SM. `igemm_sparse_tiled` uses 31 KB (12+18+1). Any increase needs careful accounting.
5. **S08 stalls**: between consecutive HMMA/HMMA.SP from same warp are HARDWARE-FIXED (8 cycles). Not optimizable via CuAssembler reordering.

---

## Recommended Next Session

1. **Warmup**: run `python3 scripts/verify_setup.py` (checks CUDA, nvcc, GPU, CuAssembler)
2. **Pick one medium issue**:
   - **#67**: migrate hgemm_sparse/bench.cu or igemm/bench_igemm_sparse.cu to BenchDriver
   - **#29**: read `phase3/flash_attention/flash_attn_br16_regpv.cu`, identify tiling gaps vs hgemm_16warp
3. **Use `/breathe`, `/rest`, `/meditate` between tasks** as instructed

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
| hgemm (all 7 variants) | 512³ | 1e-1 | 1e-1 | PASS |
| hgemm_sparse (fixed + arbitrary) | 4096³ | 1e-1 | 1e-1 | PASS |
| igemm_sparse_tiled (after #65) | 256³, 2048³, 4096³ | 0.5 | 0.1 | PASS (max_abs=0.00) |
