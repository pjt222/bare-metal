---
title: "Move WIP and test files out of production source paths"
labels: ["cleanup", "refactoring"]
---

## Problem
Development/debug artifacts live alongside production kernels. They clutter the source tree and confuse newcomers about which files are "real."

## Files to Relocate

### Phase 2 sparse HGEMM
| Current | Suggested | Type |
|---------|-----------|------|
| `kernels/gemm/hgemm_sparse/test_dense_manual.cu` | `tests/hgemm_sparse/test_dense_manual.cu` | Layout verification |
| `kernels/gemm/hgemm_sparse/test_mma_sp.cu` | `tests/hgemm_sparse/test_mma_sp.cu` | Sparse mma.sp test |
| `kernels/gemm/hgemm_sparse/verify_wmma_ab_layout.cu` | `tests/hgemm_sparse/verify_wmma_ab_layout.cu` | Fragment layout verify |

### Phase 2 IGEMM
| Current | Suggested | Type |
|---------|-----------|------|
| `kernels/gemm/igemm/test_inplace_race.cu` | `tests/igemm/test_inplace_race.cu` | WAR hazard reproduction |

### Phase 3 Flash Attention
| Current | Suggested | Type |
|---------|-----------|------|
| `phase3/flash_attention/verify_wmma_layout.cu` | `tests/flash_attention/verify_wmma_layout.cu` | WMMA layout verify |

## Proposed Directory Structure
```
 tests/
 ├── CMakeLists.txt (or Makefile)
 ├── hgemm_sparse/
 │   ├── test_dense_manual.cu
 │   ├── test_mma_sp.cu
 │   └── verify_wmma_ab_layout.cu
 ├── igemm/
 │   └── test_inplace_race.cu
 └── flash_attention/
     └── verify_wmma_layout.cu
```

## Rules
- Keep production paths clean: only kernels + benches + READMEs
- Tests verify assumptions (fragment layouts, hardware behavior). They are not benchmarks.
- Each test has its own small README explaining what it tests and expected output.

## Acceptance Criteria
- [ ] `tests/` directory created with subdirs mirroring phase2/3 structure
- [ ] All listed files moved, preserving git history (use `git mv`)
- [ ] Build scripts updated to compile tests separately
- [ ] Top-level README points to `tests/` for "how do I verify my understanding"

## Effort
Low — pure file moves + path fixes.
