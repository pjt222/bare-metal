---
title: "Consolidate READMEs and update top-level README with current state"
labels: ["documentation"]
---

## Problem
The top-level `README.md` shows Phase 4 as complete but does not mention Phase 5 (attention layer, INT8 sparse, etc.). Phase-specific READMEs have stale numbers. A newcomer cannot get an accurate project overview from the top-level README.

## Gaps

### Top-level README
- ❌ No mention of Phase 5 (`phase5/attention_layer/`)
- ❌ Phase 4 table only shows UNet primitives, missing Conv2d implicit GEMM results
- ❌ No mention of sparse GEMM (Phase 2 expansion)
- ❌ No mention of INT8 IGEMM (15,320+ TOPS)
- ❌ Performance hierarchy from `gpu_reflections.md` not surfaced
- ❌ Missing "how to get started" section beyond phases table

### Phase READMEs
- `kernels/gemm/hgemm/README.md`: Shows 7,853 GFLOPS as best, but `hgemm_16warp` achieves **31,910 GFLOPS**
- `kernels/gemm/igemm/README.md`: Good but missing online-quant variants (16,646+ GFLOPS)
- `phase3/flash_attention/README.md`: Probably stale vs `gpu_reflections.md` findings
- `phase4/README.md`: May not include implicit GEMM results

## Proposed Changes

### Top-level README — Add sections
```markdown
## Current Best Results

### GEMM
| Kernel | Size | Performance | % Peak |
|--------|------|-------------|--------|
| HGEMM 16-warp | 4096³ | 31,910 GFLOPS | 18.3% |
| HGEMM sparse 2:4 | 2048³ | 41,930 dense-equiv GFLOPS | — |
| IGEMM pipelined cp.async | 4096³ | 20,688 TOPS | 3.0% |
| IGEMM 128×256 (1 blk/SM) | 4096³ | 27,591 TOPS | 4.0% |
| Online FP16→INT8 quant | 4096³ | 16,646 GFLOPS | 9.6% |

### Flash Attention
| Kernel | Config | Time | GFLOPS |
|--------|--------|------|--------|
| flash_attn_br16_regpv | seq=1024,b=8,h=8 | 2.81 ms | 6,112 |

### Diffusion UNet
| Kernel | Config | Time | Performance |
|--------|--------|------|-------------|
| Implicit GEMM conv2d | 64×64,320ch | 1.13 ms | 6,687 GFLOPS |
```

### Phase READMEs — Audit and update
For each phase README:
1. Verify all numbers match `gpu_reflections.md` or latest bench run
2. Add "best known result" callout box
3. Link to relevant `gpu_reflections.md` insights by number

## Acceptance Criteria
- [ ] Top-level README includes Phase 5 mention
- [ ] Top-level README has "Current Best Results" table
- [ ] All phase READMEs audited for stale numbers
- [ ] All stale numbers updated from `gpu_reflections.md`
- [ ] READMEs cross-link to `docs/gpu_reflections.md` insights

## Effort
Low-Medium — audit + copy-paste from existing docs.
