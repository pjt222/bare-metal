---
title: "Create top-level build system (Makefile or CMake)"
labels: ["enhancement", "infrastructure"]
---

## Problem
No unified build system exists. Each phase uses ad-hoc `nvcc` commands. Phase 1 has a `Makefile`; phases 2–5 do not.

Building the project requires manually entering 20+ different commands across subdirectories. No way to build all kernels or run a full regression from a single command.

## Current State
```
phase1/Makefile          ← exists (Windows cmd style)
phase2/sgemm/            ← no Makefile
phase2/hgemm/            ← no Makefile
phase3/flash_attention/  ← no Makefile (10+ bench files)
```

## Proposed Solution
Add top-level `Makefile`:
```makefile
.PHONY: all phase1 phase2 phase3 phase4 phase5 test clean

all: phase1 phase2 phase3 phase4 phase5

phase1:
	$(MAKE) -C phase1 all

phase2:
	for dir in phase2/*/; do $(MAKE) -C "$$dir" cubin || true; done

# ... etc

test:
	python scripts/bench_all.py
```

Alternative: `CMakeLists.txt` for cross-platform builds.

## Acceptance Criteria
- [ ] `make all` builds all kernel cubins
- [ ] `make clean` removes all generated artifacts
- [ ] `make test` (or `make bench`) runs all benchmarks
- [ ] Per-phase targets (`make phase3`) work
- [ ] Document build commands in top-level README

## Effort
Low — mostly wiring existing commands into targets.
