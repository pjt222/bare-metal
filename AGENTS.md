# AGENTS.md

Canonical agent-facing reference for this repository. Any AI coding
assistant or human collaborator should read this file before making
non-trivial changes. Tool-specific instruction files
(`CLAUDE.md`, `.github/copilot-instructions.md`) forward here and add
only what their tool needs beyond what is documented below.

## Project

Hand-optimized CUDA / SASS kernels targeting **RTX 3070 Ti
(GA104, sm_86, Ampere)**. No cuBLAS, cuDNN, or PyTorch in the
optimized path; reference libraries are linked only for measured
comparisons under `kernels/reference/`. R is the only first-class
tooling language; there are no Python run-time dependencies.

## Hardware constants

- 46 SMs (laptop bin) / 48 SMs (desktop bin), 128 cores/SM, 64K
  32-bit registers/SM, 100 KB max shared memory/SM.
- FP32 peak 21.7 TFLOPS, FP16 Tensor Core peak 174 TFLOPS, INT8
  Tensor Core peak 348 TOPS dense (2:4-sparse 696), DRAM 608 GB/s,
  L2 4 MB.
- The 50 KB shared-memory cliff is load-bearing: blocks at >50 KB
  drop to 1 block/SM (4 warps), measured 2× regression vs blocks
  at ≤50 KB (2 blocks/SM, 8 warps).

## System policy

**Do not modify host-installed CUDA, drivers, R, or other system
software.** The toolchain is hand-configured and verified. If a
workflow needs adjustment, add scripts or wrappers under
`scripts/`; never change `/usr/local/cuda/`, `/etc/`, or any global
package manager state.

## Toolchain

Current: CUDA 13.2 (`nvcc V13.2.78`), Nsight Compute 2026.1, R 4.6.0.
`/usr/local/cuda` symlinks to the active version. WSL2 hosts
`libcuda.so` under `/usr/lib/wsl/lib/`; R subprocesses strip this
from `LD_LIBRARY_PATH` unless re-added, which existing
`scripts/*.R` entry points do automatically.

## Build and verification

Single entry points, all defined in the root `Makefile`:

```
make reproduce   # setup + verify + all + bench + figures
make setup       # renv::restore() + install local cuasmR + renv sync check
make verify      # CUDA, GPU, cuasmR, renv health check
make renv-check  # verify renv.lock matches the installed library
make all         # compile every .cu to .cubin and every bench
make bench       # run benches vs data/baselines.json
make bench-all   # full-corpus run, records to results/bench_all/
make test        # smoke-test compiled GEMM/reductions/elementwise
make test-r      # GPU-free R test suites (needs no CUDA, no GPU)
make figures     # regenerate docs/figures via R scripts
make clean       # remove cubins, sass dumps, bench executables
make disasm      # disassemble all cubins via scripts/build.R
```

Family-narrow targets exist: `make tutorial gemm reductions
attention convolution elementwise memory_layout composition
reference`.

The pre-push gate is `.githooks/pre-push` (installed by
`scripts/install-hooks.sh`). It runs five steps, ordered cheapest and most
deterministic first so a doomed push is rejected fast and legibly:

| Step | Runs | Blocks | Needs a GPU |
|------|------|--------|-------------|
| `scripts/audit/check_links.R` | ~1 s | yes | no |
| `scripts/audit/renv_check.R` | ~2–10 s | yes | no |
| `make test-r` | ~2 min | yes | no |
| `make test` | minutes | no (best-effort) | yes |
| `scripts/bench/bench_regress.R` | minutes | on a measured regression | yes |

`renv_check` precedes `test-r` because every R suite needs `cuasmR` loadable, so
an out-of-sync library should produce renv's diagnosis rather than a confusing
testthat error. `make test` precedes `bench_regress.R` because it builds the
executables that get measured.

**A run that measured nothing warns; it does not block (#176).**
`bench_regress.R` has three exit codes — `0` PASSED, `1` FAILED (at least one
measured regression), `2` INCONCLUSIVE (nothing was measured, so nothing is
certified). The hook renders `2` as a yellow warning and allows the push.

This is a deliberate decision, recorded here because the alternative is
defensible and someone will propose it. Until #176 the third case did not
exist: an all-skip run exited 0 and the hook printed "All benchmarks within
tolerance. Push allowed." having measured zero kernels — which it did on five
consecutive real pushes, at 7 of 7 skipped. Making that state *blocking* was
rejected for one reason: the routine case on this laptop is already mostly
skipped (three configs sit behind a host-side clock lock this hook never
applies, per #156; throttle takes more whenever the machine is warm), so a
blocking INCONCLUSIVE would reject ordinary pushes, and its escape hatch —
`git push --no-verify` — switches off all five steps, including the GPU-free R
suites that #163 exists to enforce. A warning that gets read beats a block that
gets bypassed. The honesty requirement is met by the summary line, which now
always names the fraction measured:

```
Total: 7 | Measured: 3 | Regressions: 0 | Improvements: 0 | Skipped: 4
RESULT: PASSED -- 3 of 7 config(s) measured, all within tolerance (4 skipped)
```

Three callers, one policy each, all reading the same exit code: the hook warns,
`make bench` warns, and `scripts/probe/run_locked_eval.ps1` propagates it —
right for a deliberate locked evaluation, where measuring nothing must not read
as success. If a binding signal is ever wanted here, that is #179, and it should
be decided for the whole gate rather than by tightening one exit code.

To actually measure the locked configs, lock the clock host-side first (elevated
Windows shell: `nvidia-smi.exe -lgc 1605,1605`), run
`Rscript scripts/bench/bench_regress.R --clock-locked 1605`, then release it
with `nvidia-smi.exe -rgc`.

**`make test-r` runtime.** 117 s of tests on AC, measured 2026-07-31 on the
RTX 3070 Ti laptop across four suites: `tests/bench_all/test_bench_all.R` 83 s,
`tests/bench_regress/test_meta.R` 9 s, `tests/bench_regress/test_verdict.R` 9 s
(added by #176), the `cuasmR` package suite 16 s. The 2026-07-30 three-suite run
was 115 s of tests / 124 s wall on AC and 133 s / 140 s on battery; the wall
figure predates the plumbing canary, which adds one R startup without changing
the per-suite total, since the canary is deliberately excluded from it. Both are
n=1. Note the totals barely moved while a whole suite was added and
`test_bench_all.R` swung 91 s → 83 s: run-to-run variance on this mount is
larger than a 9 s suite, so do not read a single total as a trend.

That battery penalty is only **1.16×**, which is much smaller than it looks like
it should be: a no-op `Rscript` on this box is **2.1×** slower on battery
(2.56 s AC, 5.34 s battery), tracking the CPU downclock. The suites do not track
it, because they are dominated by filesystem work rather than compute — see the
next paragraph. Do not extrapolate one from the other; measure the thing you
want to quote. (This note exists because the first draft predicted "roughly half
on AC" from the startup ratio, and the measurement came back 115 s, not 70 s.)

Both power figures are n=1. The battery runs came first in the session and the
AC run later, so any warm-page-cache benefit accrued to the AC number — meaning
1.16× if anything *overstates* the battery penalty rather than understating it.
The 2.0 s AC startup recorded under "Startup cost" below is a different day's
measurement of the same thing; read both as "about 2–2.5 s on AC" rather than
trying to reconcile them.

**Most of that runtime is the 9p mount, not the tests.** Controlled measurement,
2026-07-30 — the same repo copied to ext4 inside the WSL VM and run on the same
box, same R, same library, so the filesystem is the only variable:

| | `make test-r` | `test_bench_all.R` |
|---|---|---|
| `/mnt/d` (9p) | 115 s | 91 s |
| `~` (ext4) | **14 s** | **6 s** |

**8.2× overall, 15× on the dominant suite.** So roughly seven eighths of the
local hook cost is the mount, not the assertions. Bear that in mind before
"optimising" the tests — there is little there to win, and the same work is
cheap the moment it runs anywhere else.

(A GitHub runner does similar work in ~6 s for the whole target, but do not use
that as the comparison: it differs in CPU, disk and R build as well as
filesystem, and it runs four fewer assertions — see "CI limitations" below. An
earlier draft quoted 45× from exactly that confounded pairing. The
ext4-on-the-same-box row above is the one that isolates the variable. Note the
ext4 `test_bench_all.R` figure and that whole-run CI figure are both 6 s by
coincidence; they are different quantities.)

The suites were invoked by nothing at all before #163 — not the hook, not the
Makefile, not CI. One of them had been dead for 58 days without anyone noticing
(#171), which is the argument for the gate in one sentence.

**CI limitations.** GitHub-hosted runners have no Ampere GPU. Cubin builds,
benchmark runs, and anything requiring `nvcc -arch=sm_86` cannot run in CI.
Two workflows cover the GPU-free surface: `.github/workflows/docs.yml` (markdown
link validation, version-string consistency, Quarto doc rendering) and
`.github/workflows/tests.yml` (`make test-r`, against an renv library cached on
`renv.lock`). Local `make reproduce` remains the only path for GPU verification.

`tests.yml` exists because CI is the only half of the gate a `git push
--no-verify` cannot skip. **It is not authoritative for everything the local run
covers**, and runs strictly fewer assertions: four tests skip on a runner — the
three `cuasmR` roundtrip tests (they need a built cubin, which is gitignored,
and `nvdisasm` on `PATH`) and `test_meta.R`'s live-capture test (needs
`nvidia-smi`). `make test-r` prints the skip count for exactly this reason: a
skip is coverage that did not happen, and reporting it as a clean pass is how
the roundtrip check would quietly stop running. CI *asserts* that count (4);
locally it is reported but not enforced, because a dev box with a built cubin
legitimately skips nothing.

One divergence is not a skip but a structural blind spot: the runner's
source-vs-installed `cuasmR` check cannot fire in CI at all, since CI reinstalls
the package from the working tree every run. A stale local install is catchable
only by the local half of the gate.

## Publishing the corpus to Hugging Face

`make publish-hf` re-syncs the kernel corpus to the Hugging Face
dataset repo `pjt222/ga104-cuda-kernels` (audience: SASS /
optimization researchers). It is the single command behind WS4 of
issue #109.

The target runs `scripts/publish_hf.R`, which: verifies the toolchain
and GPU, loads `HF_TOKEN`, rebuilds the corpus
(`make clean && make all && make disasm` — **an Ampere GPU is
required**), asserts every expected cubin/sass exists and is current
and cross-checks coverage against `data/baselines.json`, writes
`SHA256SUMS`, renders the dataset card from `hf/README.md`, and runs
`hf repo create` + `hf upload`.

`HF_TOKEN` (a write-scoped token) is read from the repo-root `.env`
file, or from the environment if already set. Copy `.env.example` to
`.env` and paste the token; `.env` is gitignored and must never be
committed.

Inspect the resolved upload manifest without building or uploading:

```
make publish-hf ARGS=--dry-run
```

## R environment

`renv.lock` pins R 4.6.0 and every script dependency. The
`.Rprofile` at the repo root auto-activates the project library
on `Rscript` startup. First-time setup is `make setup` (or
`Rscript -e 'renv::restore()'` for renv only).

**Startup cost — do not re-enable the autoloader's sync check (#162).**
`.Rprofile` sets `options(renv.config.synchronized.check = FALSE)`
before sourcing `renv/activate.R`. Without it,
renv's autoloader runs a full project dependency scan on *every* R
startup: Rprof attributes 92 % of `renv::load()` to that path, and its
cost is filesystem stats over the 9p `/mnt/d` mount. Measured on AC, a
no-op `Rscript -e 'invisible(1)'` went **33.2 s → 2.0 s**, and the
GPU-free test suite went from ~1 h to ~74 s. The check is not lost — it
runs deliberately via `scripts/audit/renv_check.R`, wired into the
pre-push hook, `make setup` and `make renv-check`. Dependency discovery
itself is deliberately left unweakened: there is no `.renvignore` and
`snapshot.type` stays `implicit`, so a new R script anywhere in the tree
still registers.

**Install cuasmR with `Rscript scripts/install_cuasmR.R`, never bare
`install.packages()`.** Only `renv::install()` writes the
`RemoteType: local` stamp that renv matches against the lockfile; a base
install leaves the project permanently `[Local != unknown]` out of sync.
This regressed once already (fixed in #133, reverted by the installer
script, re-fixed in #162) — the script now asserts the stamp landed.

Required packages: `jsonlite`, `ggplot2`, `scales`, `patchwork`,
`dplyr`, `tidyr`, `tibble`, `rmarkdown`, `yaml`, `testthat`, and
the local `cuasmR` package installed via
`Rscript scripts/install_cuasmR.R`.

## Repository layout

```
kernels/                  primary product surface, grouped by family
  _common/                shared bench.h, check.h, bench_driver.h
  tutorial/               vector_add: SASS hello world (FADD→FMUL)
  gemm/                   sgemm / hgemm / hgemm_sparse / igemm
  reductions/             softmax / layernorm / groupnorm
  elementwise/            activations / timestep_emb
  attention/              flash_attention / cross_attention
  convolution/            conv2d / resblock
  memory_layout/          cymatic (Chladni-pattern gather)
  composition/            attention_layer (multi-kernel layer)
  reference/              cublas / cudnn / cusparselt local references
R/cuasmR/                 local R package: byte-level cubin patcher
scripts/                  R tooling: build, audit, bench, profile,
                          model, cymatic subdirs
tests/                    development and verification tests (not perf)
experiments/              front-end sandbox (cuda-oxide spike)
data/                     regenerable CSV/JSON (baselines, audits)
results/                  captured benchmark + NCU output
docs/                     documentation and analyses
viz/                      interactive visualizations
```

## SASS hand-edit workflow

The pipeline is `.cu → nvcc → PTX → ptxas → SASS (.cubin) →
cuasmR → patched .cubin`. The R package `cuasmR` reads cubins via
nvdisasm, indexes instructions by file offset in the `.text`
section, and patches at the byte level. No re-encoding from SASS
text is performed; new opcodes come from disassembling a sibling
`.cu`.

```bash
Rscript scripts/build.R compile   kernel.cu              # .cu -> .cubin
Rscript scripts/build.R disasm    kernel.sm_86.cubin     # .cubin -> .cuasm
Rscript scripts/build.R roundtrip kernel.cu              # byte-identical check

Rscript -e '
  library(cuasmR)
  obj <- cuasm_read("path/to/kernel.sm_86.cubin")
  obj <- cuasm_set(obj, kernel = "my_kernel", slot = 13,
                   instr_hex = "0x...", ctrl_hex = "0x...")
  cuasm_write(obj, "path/to/kernel.patched.cubin")
'
```

Full design in `docs/cuasm_r.md`.

## Code conventions

- Kernel entry points: `extern "C"`, `__launch_bounds__(threads, blocks)`,
  `__restrict__` on pointer arguments, `size_t` for indices that may
  exceed 32 bits.
- Dynamic shared memory declared as `extern __shared__ char smem_raw[]`.
- Bench harnesses use the CUDA Driver API, not the Runtime API.
  Module load uses a relative cubin filename, so benches must run
  from their own directory.
- Output format `label  ms  GFLOPS/TOPS` so
  `scripts/bench/bench_regress.R` can parse it.
- Per-kernel README required: purpose, measured results,
  copy-pasteable build commands, references to relevant docs and
  postmortems.

## Correctness

Every kernel has a `bench.cu` with a CPU reference. `check_fp32()`
uses AND-logic: an element fails only when both absolute and
relative error exceed tolerance. Per-precision defaults:

| Precision / class                  | abs    | rel    |
|------------------------------------|-------:|-------:|
| FP32 scalar                        | 1e-3   | 1e-3   |
| FP16 Tensor Core (HMMA)            | 1e-2   | 1e-2   |
| `--use_fast_math` (sin/cos)        | 5e-4   | 5e-4   |
| Conv2d (9× re-accumulation)        | 1e-2   | 1e-2   |
| INT8 Tensor Core (IMMA, sym quant) | 0.5    | 0.1    |

## Performance measurement

- GFLOPS: `(2 * M * N * K) / (time_ms / 1000) / 1e9`.
- Bandwidth: `total_bytes / (time_ms / 1000) / 1e9`.
- Timing via `BenchTimer` (CUDA events), not wall-clock.
- Warmup precedes measured runs; default is 5 warmup + 11 timed runs
  reported as median.
- Regression gate: `data/baselines.json` plus
  `scripts/bench/bench_regress.R`. Tolerance defaults to 10% per
  kernel; per-config overrides allowed.

## The four laws of GA104

1. Feed Tensor Cores continuously. Overlap loads with HMMA/IMMA.
   At ≥8 warps, cp.async benefit depends on compute/load ratio
   (helpful when compute is short, harmful when compute is long).
2. Read each byte of DRAM at most once per kernel. im2col converts
   9× re-reads to 1×; implicit GEMM eliminates the col buffer.
3. Fill the warp schedulers. 32 warps/SM is ideal, 8 sufficient;
   below 8 indicates a structural problem.
4. Never cross the 50 KB shared-memory cliff per block.

These laws are derived empirically; see `docs/gpu_reflections.md`
for the observations behind each.

## Key SASS instructions

- `HMMA.16816.F32` — FP16→FP32 Tensor Core, 16×8×16 per warp. S08
  stall between consecutive HMMAs is a hardware constraint.
- `IMMA.16816.S8.S8` — INT8 Tensor Core, 16×8×16, S04 default,
  S02 sustainable when operands are ready.
- `HMMA.16816.SP` — sparse 2:4 variant of HMMA.
- `SHFL.BFLY` — warp butterfly reduction (5 instructions for a
  32-lane reduce).
- `MUFU.EX2/RCP/RSQ/SIN/COS` — special function unit, ≈16 cycle
  latency. `exp(x) = 2^(x · log2(e))` requires FMUL + MUFU.EX2.
- `FFMA` — FP32 fused multiply-add.
- `LDGSTS` (cp.async) — async global → shared copy, bypasses the
  register file.

## Documentation entry points

- `docs/index.md` — full document map.
- `docs/inventory.md` — kernel inventory grouped by family.
- `docs/gpu_reflections.md` — observation catalogue (first-person
  format is a deliberate stylistic experiment; see preamble).
- `docs/CONTINUE_HERE.md` — session handoff scratchpad. **Gitignored** (local,
  per-author, intra-session); absent from a fresh clone.
- `docs/tutorial/` — six-chapter prose walkthrough.
- `CHANGELOG.md` — structural reorganizations and audit history.

## Current state

All five development phases are complete (vector add, GEMM family,
Flash Attention, diffusion primitives, sparse / INT8 / epilogue
optimization). Active optimization queue is near-empty; remaining
items are research-grade scope, tracked in the open issue list — currently
the #152 benchmark-convergence epic and its follow-ups (#158–#168). Do not
read the issue numbers here as a live queue; check the tracker.
See `docs/CONTINUE_HERE.md` for the working handoff, if present — it is
gitignored and local to whoever last worked in the tree.
