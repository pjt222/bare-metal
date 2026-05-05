---
title: "Add CONTRIBUTING.md with coding conventions and PR checklist"
labels: ["documentation", "good-first-issue"]
---

## Problem
No contributor guidelines exist. The project's implicit conventions (from `CLAUDE.md` and existing code) are not documented for external contributors.

## Conventions to Document

### Code Style
- Use `extern "C"` on all kernel entry points
- `__launch_bounds__` required on all kernels
- Prefer `size_t` for large index calculations (avoid 32-bit overflow)
- Use `__restrict__` on all kernel pointer arguments
- Shared memory: use `extern __shared__ char smem_raw[]` for dynamic sizing

### Naming
- Kernels: `snake_case` descriptive name (e.g., `igemm_pipelined_cpasync`)
- Bench files: `bench.cu` for primary, `bench_variant.cu` for alternates
- Constants: `ALL_CAPS` with underscore separators

### Benchmark Requirements
Every new kernel MUST have:
1. `bench.cu` with CUDA Driver API launch (not runtime API)
2. CPU reference computation for correctness verification
3. `WARMUP()` before `BENCH()`
4. `check_fp32()` or equivalent with documented tolerance
5. Results printed in standard format: `label  ms  GFLOPS  (vs_ref)`

### Correctness Tolerances
| Precision | abs_tol | rel_tol | Context |
|-----------|---------|---------|---------|
| FP32 scalar | 1e-3 | 1e-3 | SGEMM, vector ops |
| FP16 Tensor Core | 1e-2 | 1e-2 | HGEMM, Flash Attention |
| INT8 Tensor Core | 0.5 | 0.1 | IGEMM (quantization error) |
| Fast math (sin/cos) | 5e-4 | 5e-4 | Timestep embedding |
| Conv2d (deep accumulation) | 1e-2 | 1e-2 | 9× reaccumulation |

### Documentation Requirements
Every new kernel directory MUST have:
1. `README.md` with: what the kernel does, measured results, build command
2. Key design decisions explained (tile sizes, why those sizes)
3. Link to relevant `gpu_reflections.md` insights

### SASS Hand-Editing Workflow
When modifying SASS via CuAssembler:
1. Run `roundtrip` before any edits
2. Make ONE change at a time
3. Test correctness after each change
4. Document the change and measured effect in commit message

### Before Submitting PR
- [ ] `python scripts/verify_setup.py` passes
- [ ] All new kernels bench correctly against CPU reference
- [ ] Performance meets or exceeds documented baseline (or regression is explained)
- [ ] `make clean && make all` builds successfully (after #01)
- [ ] `git diff` shows only intended changes

## Proposed File: `CONTRIBUTING.md`
```markdown
# Contributing to bare-metal GPU

## Quick Start
1. Fork and clone
2. Run `python3 scripts/verify_setup.py`
3. Make changes
4. Build and test your kernel
5. Submit PR

## Code Conventions
...

## SASS Editing
...
```

## Acceptance Criteria
- [ ] `CONTRIBUTING.md` exists at repo root
- [ ] Includes all conventions listed above
- [ ] Links to `docs/gpu_reflections.md` for architecture context
- [ ] Mentions `CLAUDE.md` as project-specific AI assistant guidance

## Effort
Low — documentation only.
