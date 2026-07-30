# Contributing to bare-metal GPU

This project builds ML primitives from hand-optimized SASS on GA104 (RTX 3070 Ti, sm_86). Every kernel is benchmarked and correctness-checked against a CPU reference.

## Quick Start

1. Fork and clone
2. Run `Rscript scripts/verify_setup.R` — must pass before any changes
3. Make changes on a feature branch (`git checkout -b feature/my-kernel`)
4. Build and test your kernel
5. Open a Pull Request

## Git flow (2026-07-30 onward)

**No direct pushes to `main`.** Every change goes:

    commit -> topic branch -> PR -> adversarial review -> merge

- **Adversarial review before merge.** Have the diff argued against, not just
  read. Agent-assisted work should run the `advocatus-diaboli` reviewer; a human
  reviewer plays the same role. Record what it found on the PR, including
  findings you rejected and why.
- **Merge with a MERGE COMMIT — never squash, never rebase-merge.** Squashing a
  base branch makes a stacked child hit a three-way conflict on its next rebase,
  and it destroys the individual-commit trail that makes a review auditable.
  `gh pr merge --merge`, not `--squash`.
- The pre-push hook runs on every branch push (README link audit, renv sync
  check, `make test-r`, `make test`, `bench_regress.R`). Expect a few minutes of
  benches; do not interrupt it — an interrupted push can leave a spinning bench
  that wedges the WSL CUDA path.

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

1. **CUDA Driver API** launch (not Runtime API) — see `kernels/_common/bench.h`
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

When modifying SASS via the local R package `cuasmR` (see `docs/cuasm_r.md`):

1. Run `Rscript scripts/build.R roundtrip kernel.cu` before any edits
   (verifies cuasmR can byte-identical roundtrip the cubin)
2. Make **one** change at a time
3. Test correctness after each change
4. Document the change and measured effect in the commit message

The edit pattern is byte-level: read instr_hex / ctrl_hex, modify, write
back. Example FADD -> FMUL on Phase 1:
```r
library(cuasmR)
obj <- cuasm_read("kernels/tutorial/vector_add.sm_86.cubin")
obj <- cuasm_set(obj, "vector_add", slot = 13,
                 instr_hex = "0x0000000304097220",   # FMUL opcode
                 ctrl_hex  = "0x004fca0000400000")
cuasm_write(obj, "kernels/tutorial/vector_add.fmul.cubin")
```

New opcode encodings come from disassembling a sibling `.cu`. The cuasmR
`cuasm_save_cuasm()` produces a human-readable text dump showing each
slot's `instr_hex` / `ctrl_hex` for grep-friendly inspection. See
`docs/control_codes.md` for control-word field meanings.

### Benchmark Regression Check

Install the pre-push hook to catch performance regressions before they reach CI:

```bash
bash scripts/install-hooks.sh
```

This configures a `pre-push` hook that runs, cheapest and most deterministic
first:

1. `scripts/audit/check_links.R` — every relative link in every README resolves
2. `scripts/audit/renv_check.R` — `renv.lock` matches the installed library
3. `make test-r` — the GPU-free R test suites (#163)
4. `make test` — build and smoke-test benches (best-effort; does not block)
5. `scripts/bench/bench_regress.R` — performance against `data/baselines.json`

Steps 1, 2, 3 and 5 block the push on failure. Only step 3 needs no GPU, so it
is the one substantial check that still works on a machine with no card
attached.

**Bypass** (for WIP or when you know baseline is stale):
```bash
git push --no-verify
```

- [ ] `Rscript scripts/verify_setup.R` passes
- [ ] All new kernels bench correctly against CPU reference
- [ ] Performance meets or exceeds documented baseline (or regression is explained)
- [ ] `git diff --stat` shows only intended changes
- [ ] Commit messages describe *what* and *why* (not just "fix bug")

## Resources

- [`docs/gpu_reflections.md`](docs/gpu_reflections.md) — 24 empirical insights from this hardware
- [`docs/ampere_sass_reference.md`](docs/ampere_sass_reference.md) — instruction quick reference
- [`docs/troubleshooting.md`](docs/troubleshooting.md) — common pitfalls and fixes
- [`CLAUDE.md`](CLAUDE.md) — project-specific guidance for AI assistants
