---
title: "Fix .gitignore conflicts and add missing entries"
labels: ["bug", "good-first-issue"]
---

## Problem
`.gitignore` has several issues:

### 1. `CONTINUE_HERE.md` is tracked but ignored
```
# Session handoff (ephemeral, not committed)
CONTINUE_HERE.md
```
But the file IS tracked in git and is the primary session-state document. Either:
- Commit it intentionally (remove from `.gitignore`), OR
- Replace with a different mechanism (e.g., start each session with `git log`)

Recommendation: **Remove from `.gitignore` and commit it.** Session state is valuable project history.

### 2. `.claude/` directory not ignored
The `.claude/` directory (imported context/conversation state) appears in `git status` as untracked. It should be ignored.

### 3. Hand-tuned binary naming inconsistent
```
!*_handtuned.cuasm   ← allow hand-tuned source
```
But compiled hand-tuned binaries (`.cubin`) are not similarly excepted. Should be:
```
!*_handtuned.cuasm
!*_handtuned.sm_86.cubin
```

### 4. Missing IDE/editor patterns
Add common patterns:
```
# Editors
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
```

### 5. `*.sass` ignored but useful for reference
```
*.sass
```
Raw SASS from `cuobjdump -sass` is useful for quick inspection without CuAssembler. Consider keeping `.sass` files for key kernels (or generate them on-demand via Makefile).

## Proposed `.gitignore`
```gitignore
# Compiled CUDA binaries
*.cubin
!*_handtuned.sm_86.cubin

# Generated SASS disassembly (keep hand-tuned variants)
*.cuasm
!*_handtuned.cuasm

# Raw SASS reference (generated on-demand, not committed)
*.sass

# Compiled executables (not source files)
bench
bench_*
!bench_*.cu
test_dense_manual
test_mma_sp
test_inplace_race
verify_wmma_ab_layout
verify_wmma_layout
kernels/tutorial/host

# CuAssembler working artifacts
*.cuasm.orig

# Third-party tools (clone separately)
tools/CuAssembler/

# Python
__pycache__/
*.pyc

# IDE / Editor
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Claude Code / context
.claude/
```

## Acceptance Criteria
- [ ] `CONTINUE_HERE.md` removed from `.gitignore` (or decision documented)
- [ ] `.claude/` added to `.gitignore`
- [ ] Editor/OS patterns added
- [ ] `git status --short` is clean (no untracked noise)

## Effort
Very low.
