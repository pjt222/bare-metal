---
title: "Key Insights from Session 2026-05-05"
labels: ["documentation", "insights"]
---

> Session date: 2026-05-05  
> GPU: RTX 3070 Ti Laptop (GA104, sm_86, Ampere, 4 MB L2)

## Infrastructure

1. **GitHub issues live in database, not repo.** Local `.md` files don't sync automatically. Need `gh issue create` or API call.
2. **`.gitignore` `bench_*` pattern is aggressive.** Caught `bench_driver.h`, `bench_regress.py`, `bench_flash_all.py`. Need explicit `!` exceptions.
3. **Hand-tuned `.cubin` binary in `igemm/` not tracked.** `igemm_tiled_handtuned.sm_86.cubin` exists but is neither committed nor fully ignored. Should decide: commit as artifact or `.gitignore` + document rebuild path.

## GPU Architecture

4. **L2 cache size is a hard limit for metadata-heavy kernels.** Sparse 2:4 GEMM metadata working set at 4096³ = 4.1 MB vs 4.0 MB L2 = exact thrashing threshold. The crossover is sharp, not gradual.
5. **Instruction mix reveals more than top-line performance.** 160 PRMT vs 64 PRMT tells the real story — manual INT8 packing costs 2.5× vs hardware LDSM path. But the *regression* cause (L2) was only visible via size-scaling analysis, not instruction count.
6. **ncu blocked on GeForce GPUs (ERR_NVGPUCTRPERM).** `cuobjdump` + `nvcc --cubin -res-usage` + benchmarks are the accessible profiling stack.
7. **Register count alone is misleading.** Sparse kernel uses 64 regs (good) vs dense 126, but the real bottleneck is memory hierarchy, not occupancy.

## Project Hygiene

8. **Bench boilerplate is the biggest codebase drag.** 26 bench files × ~100 duplicate lines = 2,600 lines. `bench_driver.h` reduced 3 refactored files from 1739 → 297 lines (-83%). Remaining 23 files still need migration.
9. **Makefile rules need per-phase specificity.** The top-level Makefile works, but `bench*` wildcard is too broad for Flash Attention (10+ bench variants). Need explicit variant discovery.
10. **`CHECK_CU` macro had a syntax bug.** Missing `}` in `do { ... } while(0)` caused `cuCtxCreate` compilation failure. Fixed in PR #54.

## Open Questions for Next Session

- How much does metadata preloading to smem actually help? (needs kernel modification + bench)
- Can `ldmatrix` replace scalar B-pack in sparse INT8? (needs fragment layout verification)
- Full benchmark driver migration: remaining 23 bench files
- Automated CI for GPU benchmarks (needs self-hosted runner or manual trigger)
