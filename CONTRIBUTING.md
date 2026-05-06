# Contributing to bare-metal GPU

This project builds ML primitives from hand-optimized SASS on GA104 (RTX 3070 Ti, sm_86). Every kernel is benchmarked and correctness-checked against a CPU reference.

## Quick Start

1. Fork and clone
2. Run `python3 scripts/verify_setup.py` — must pass before any changes
3. Make changes on a feature branch (`git checkout -b feature/my-kernel`)
4. Build and test your kernel
5. Open a Pull Request

## Code Conventions

### Kernel Code

- **`extern "C"`** on all kernel entry points (prevents C++ mangling, required by Driver API)
- **`__launch_bounds__`** required on all kernels (e.g. `__launch_bounds__(128, 3)`)
- Prefer **`size_t`** for large index calculations (avoids 32-bit overflow at seq_len > 65535)
- Use **`__restrict__`** on all kernel pointer arguments
- Shared memory: use `extern __shared__ char smem_raw[]` for dynamic sizing when possible

### Naming

- Kernels: `snake_case` descriptive (e.g. `igemm_pipelined_cpasync`)
- Bench files: `bench.cu` for primary; `bench_variant.cu` for alternates
- Constants: `ALL_CAPS` with underscore separators

### Benchmark Requirements

Every new kernel **must** have a `bench.cu` with:

1. **CUDA Driver API** launch (not Runtime API) — see `phase2/common/bench.h`
2. **CPU reference** computation for correctness verification
3. `WARMUP()` before `BENCH()` — GPU clocks must stabilize
4. `check_fp32()` or equivalent with **documented tolerance**
5. Results printed in format: `label  ms  GFLOPS  (vs_ref)`

### Correctness Tolerances

| Precision | abs_tol | rel_tol | Context |
|-----------|---------|---------|---------|
| FP32 scalar | 1e-3 | 1e-3 | SGEMM, vector ops |
| FP16 Tensor Core | 1e-2 | 1e-2 | HGEMM, Flash Attention |
| INT8 Tensor Core | 0.5 | 0.1 | IGEMM (quantization error) |
| Fast math (sin/cos) | 5e-4 | 5e-4 | Timestep embedding |
| Conv2d (deep accumulation) | 1e-2 | 1e-2 | 9× reaccumulation |

### Documentation Requirements

Every new kernel directory **must** have a `README.md` with:

1. What the kernel does and why it exists
2. Measured results (size, time, GFLOPS/TOPS)
3. Build command (copy-pasteable)
4. Link to relevant `docs/gpu_reflections.md` insights (e.g. "See Insight 14")

## SASS Hand-Editing Workflow

When modifying SASS via CuAssembler:

1. Run `python scripts/build.py roundtrip kernel.cu` before any edits
2. Make **one** change at a time
3. Test correctness after each change
4. Document the change and measured effect in the commit message

The CuAssembler control code format:
```
[B0-----:R-:W0:Y:S04]  FADD R0, R2, R6 ;
```
See `docs/control_codes.md` for field meanings.

### Benchmark Regression Check

Install the pre-push hook to catch performance regressions before they reach CI:

```bash
bash scripts/install-hooks.sh
```

This configures a `pre-push` hook that:
1. Runs `make test` to build and smoke-test benches
2. Runs `scripts/bench_regress.py` to detect performance regressions against `docs/baselines.json`
3. Blocks the push if any kernel regresses beyond tolerance

**Bypass** (for WIP or when you know baseline is stale):
```bash
git push --no-verify
```

- [ ] `python3 scripts/verify_setup.py` passes
- [ ] All new kernels bench correctly against CPU reference
- [ ] Performance meets or exceeds documented baseline (or regression is explained)
- [ ] `git diff --stat` shows only intended changes
- [ ] Commit messages describe *what* and *why* (not just "fix bug")

## Resources

- [`docs/gpu_reflections.md`](docs/gpu_reflections.md) — 24 empirical insights from this hardware
- [`docs/ampere_sass_reference.md`](docs/ampere_sass_reference.md) — instruction quick reference
- [`docs/troubleshooting.md`](docs/troubleshooting.md) — common pitfalls and fixes
- [`CLAUDE.md`](CLAUDE.md) — project-specific guidance for AI assistants
