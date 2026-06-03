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
make reproduce   # setup + verify + all + bench
make setup       # renv::restore() + install local cuasmR
make verify      # CUDA, GPU, cuasmR, renv health check
make all         # compile every .cu to .cubin and every bench
make bench       # run benches vs data/baselines.json
make test        # smoke-test compiled GEMM/reductions/elementwise
make clean       # remove cubins, sass dumps, bench executables
make disasm      # disassemble all cubins via scripts/build.R
```

Family-narrow targets exist: `make tutorial gemm reductions
attention convolution elementwise memory_layout composition
reference`.

The pre-push gate is `scripts/install-hooks.sh`, which runs
`make test`, a README link audit, and `scripts/bench/bench_regress.R`.

**CI limitations.** GitHub-hosted runners have no Ampere GPU. Cubin builds,
benchmark runs, and anything requiring `nvcc -arch=sm_86` cannot run in CI.
The `.github/workflows/docs.yml` workflow covers only GPU-free checks: markdown
link validation, version-string consistency, and Quarto doc rendering. Local
`make reproduce` remains the only path for GPU verification.

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
- `docs/CONTINUE_HERE.md` — session handoff scratchpad.
- `docs/tutorial/` — six-chapter prose walkthrough.
- `CHANGELOG.md` — structural reorganizations and audit history.

## Current state

All five development phases are complete (vector add, GEMM family,
Flash Attention, diffusion primitives, sparse / INT8 / epilogue
optimization). Active optimization queue is near-empty; remaining
items are research-grade scope and tracked under GitHub issues
#103–#105. See `docs/CONTINUE_HERE.md` for the working handoff.
