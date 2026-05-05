# Continue Here

> Last updated: 2026-04-16T21:15:00Z | Branch: main

## Objective

Hand-optimized CUDA/SASS sparse GEMM on RTX 3070 Ti Laptop GPU (GA104, sm_86).
All primary targets complete and committed. Repo is in a clean, benchmarked state.
Remaining items are exploratory / low priority.

## Completed

- [x] **#33 B-fragment ldmatrix.trans** — FP16 sparse HGEMM: 41,930 dense-equiv GFLOPS (+73%). `2a2efbe`.
- [x] **#34 Dynamic 2:4 metadata** — `sparse_meta.h` (FP16). Arbitrary patterns validated.
- [x] **#7 INT8 sparse IMMA** — `igemm_sparse_tiled.cu` + `bench_igemm_sparse.cu`. 39,745 GFLOPS at
      2048³. `82b9a57`. All sizes PASS zero error. Key design decisions:
      - STRIDE_B=144 (not 136 — 136 mod 16 = 8, breaks cp.async 16B alignment)
      - ldmatrix.trans rejected for B (b16 granularity → N-column pairs, not K-rows)
      - ldmatrix.m8n8.x2 for A works identically to FP16 (both have 16-byte rows)
      - Full 32-bit metadata, 8 nibbles, no upper-16 duplication (unlike FP16)
- [x] **#29 Flash Attn ldmatrix** — RESOLVED (no code change). Compiler already emits
      `LDSM.16.MT88.4` for K loads in `flash_attn_br16_regpv.cu`. PRMT instructions are
      from Q global loads, not smem. Caching Q in smem: +8 KB → 40 KB → 2 blocks/SM (−33%).
- [x] **Laptop dense baseline** — Measured 31,910 GFLOPS (hgemm_16warp, 2048³ and 4096³).
      Updated `phase2/hgemm_sparse/bench.cu` from hardcoded 32,197 (desktop) → 31,910. `26d14e2`.

## In Progress

(none — clean state, all commits pushed)

## Next Steps

1. **#17 smem B padding** — deprioritized; no observed bank conflicts
2. **#18 128×256 tiles** — deprioritized
3. **#4 GroupNorm fusion** — low priority; phase4/ already has groupnorm/
4. **#32 polyhedral research** — exploration only
5. **#14 tutorial series** — write last, after all kernels finalized

## Context

### Performance table (laptop GPU — RTX 3070 Ti Laptop, 31,910 dense baseline)

| Kernel | Size | Dense-equiv GFLOPS | vs Dense |
|--------|------|-------------------|----------|
| hgemm_sparse_tiled (FP16) | 2048³ | ~40,605 | 127% |
| hgemm_sparse_tiled (FP16) | 4096³ | ~40,787 | 128% |
| igemm_sparse_tiled (INT8) | 2048³ | 39,745  | 125% |
| igemm_sparse_tiled (INT8) | 4096³ | 21,593  | 68%  |
| flash_attn regpv           | seq=1024,b=8,h=8 | 6,805 | — |

### INT8 4096³ underperforms vs 2048³
21,593 at 4096³ vs 39,745 at 2048³ — register spill suspected. Not investigated.
Worth profiling with `ncu --metrics l1tex__t_sectors_pipe_lsu_mem_local` if needed.

### igemm bench (phase2/igemm/bench.cu)
Reports all dense IGEMM variants + HGEMM; no hardcoded baseline (uses live measurements).
Only `phase2/hgemm_sparse/bench.cu` had the hardcoded value.
