---
title: "Track untracked hand-tuned binary: igemm_tiled_handtuned.sm_86.cubin"
labels: ["cleanup", "git"]
---

## Problem
`kernels/gemm/igemm/igemm_tiled_handtuned.sm_86.cubin` exists in working tree but is neither committed nor `.gitignore`d. Creates confusion about whether it's volatile or a preserved artifact.

```bash
$ git status
?? kernels/gemm/igemm/igemm_tiled_handtuned.sm_86.cubin
```

## Decision Needed

### Option A: Commit as artifact (preferred for SASS work)
- Rationale: Hand-tuned SASS binaries are the project output. Committing them preserves exact verified state.
- Action: Add to git, document in README
- Risk: Binary bloat (~few KB each, manageable)

### Option B: .gitignore + document rebuild path
- Rationale: Binaries are generated from `.cuasm` + build script. Only sources should be tracked.
- Action: Add `.gitignore` pattern `!*_handtuned.sm_86.cubin` already exists but doesn't match subdirectory path
- Risk: Rebuild requires CuAssembler toolchain, may drift from original

## Recommendation
Commit this specific binary (it's a verified artifact from CuAssembler hand-tuning session) and document rebuild path in `kernels/gemm/igemm/README.md`.

## Files
- `kernels/gemm/igemm/igemm_tiled_handtuned.sm_86.cubin`
