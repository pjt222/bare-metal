# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hand-optimized CUDA/SASS assembly kernels targeting **RTX 3070 Ti (GA104, sm_86, Ampere)**. No cuBLAS, cuDNN, or PyTorch — ML primitives built from scratch in SASS. Five phases complete: vector add, GEMM/softmax/layernorm, Flash Attention, diffusion UNet primitives, and optimization experiments.

## Hardware Constraints (GA104)

- 48 SMs, 128 cores/SM, 64K registers/SM, 128 KB shared memory/SM
- FP32 peak: 21.7 TFLOPS | FP16 Tensor peak: 174 TFLOPS | DRAM BW: 608 GB/s
- **The 50 KB smem cliff is load-bearing**: ≤50 KB = 2 blocks/SM = 8 warps (good); >50 KB = 1 block/SM = 4 warps (occupancy collapse, 2× regression measured). GA104 sm_86 max smem/SM is 100 KB; the cliff is at 100/2=50 KB/block, not 64 KB. (Confirmed: 48 KB → 2 blocks ✓, 56 KB → 1 block ✗)
- L2 cache: 4 MB — implicit GEMM speedup scales with col buffer size relative to this

## R Environment (renv)

Project uses [renv](https://rstudio.github.io/renv/) for R package management.
Lockfile `renv.lock` pins R 4.6.0 + all R script dependencies. The `.Rprofile`
at the repo root auto-activates the project library on `Rscript` startup.

First-time setup after clone:
```bash
Rscript -e 'renv::restore()'
```

Required packages:
- `jsonlite` — JSON parse (bench_regress.R)
- `ggplot2`, `scales`, `patchwork` — figures (sass_histogram.R, dashboards)
- `dplyr`, `tidyr`, `tibble`, `rmarkdown`, `yaml` — used by other R helpers
- `testthat` — unit tests for the local cuasmR package
- `cuasmR` — local R package (`R/cuasmR/`) for SASS hand-edits, install
  via `Rscript scripts/install_cuasmR.R`

**Tooling language policy**: R is the primary language for everything
under `scripts/*` and `R/*`. There are no Python dependencies at
run-time; the previous reliance on `cloudcores/CuAssembler` (Python) was
replaced by the local `cuasmR` R package (#102, see `docs/cuasm_r.md`).

## CAUTION: Do Not Modify System Installations

**Never modify, reinstall, update, or reconfigure system-level tools** — this includes CUDA toolkit (`/usr/local/cuda/`), nvcc, cuobjdump, nvdisasm, nvidia drivers, Python system packages, or any other host-installed software. These are manually configured and verified for this hardware.

If a build step or script needs adjustment, create or modify scripts **inside this repository** (e.g., `scripts/`, `tools/`). Wrapper scripts, PATH overrides, and local virtualenvs are acceptable — system-wide changes are not.

## Build Commands

All commands run in WSL with CUDA on PATH (`export PATH=/usr/local/cuda/bin:$PATH`).
Current toolchain: CUDA 13.2 (`nvcc V13.2.78`), Nsight Compute 2026.1.1.0.
`/usr/local/cuda` symlinks to `/etc/alternatives/cuda` -> `/usr/local/cuda-13.2`.

**WSL2 GPU library path**: `libcuda.so` lives in `/usr/lib/wsl/lib/`. Subprocess
launchers (notably Rscript) prepend their own libdir to `LD_LIBRARY_PATH`,
masking the WSL passthrough. R-based tooling under `scripts/*.R` must add
`/usr/lib/wsl/lib` back to `LD_LIBRARY_PATH` before spawning CUDA processes
or the child sees "no CUDA-capable device is detected".

```bash
# Compile kernel to cubin
nvcc --cubin -arch=sm_86 -O2 -o kernel.sm_86.cubin kernel.cu

# Compile benchmark executable (most phases use CUDA Driver API)
nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../../phase2/common

# Run benchmark (args vary per kernel — typically M N K or seq_len batch heads)
./bench 4096 4096 4096

# SASS inspection
cuobjdump -sass kernel.sm_86.cubin | grep -E 'HMMA|SHFL|MUFU|FFMA'
cuobjdump -sass kernel.sm_86.cubin | grep HMMA | wc -l
```

### SASS Hand-Edit Workflow (cuasmR)

```bash
# CLI driver
Rscript scripts/build.R compile   kernel.cu             # .cu -> .cubin
Rscript scripts/build.R disasm    kernel.sm_86.cubin    # .cubin -> .cuasm dump
Rscript scripts/build.R roundtrip kernel.cu             # compile + read+write byte-identical check

# Programmatic edits (typical flow):
Rscript -e '
  library(cuasmR)
  obj <- cuasm_read("path/to/kernel.sm_86.cubin")
  obj <- cuasm_set(obj, kernel = "my_kernel", slot = 13,
                   instr_hex = "0x...", ctrl_hex = "0x...")
  cuasm_write(obj, "path/to/kernel.patched.cubin")
'
```

New opcode encodings come from disassembling a sibling `.cu` (compile,
`cuasm_read`, copy `instr_hex` for the desired instruction). Full
documentation in `docs/cuasm_r.md`.

### Environment Verification

```bash
Rscript scripts/verify_setup.R   # checks CUDA, nvcc, GPU, cuasmR
```

## Architecture

```
CUDA C++ (.cu) -> nvcc -> PTX -> ptxas -> SASS (.cubin) -> cuasmR (R) -> patched .cubin
```

### Phase Structure

Each phase directory contains kernel `.cu` files, `bench_*.cu` benchmarks, a `Makefile`, and a `README.md` with algorithm walkthrough and results.

- **phase1/** — Vector add "Hello World" (FADD→FMUL hand-edit proof of concept)
- **phase2/** — ML primitives: `sgemm/`, `hgemm/`, `softmax/`, `layernorm/`, `activations/`
- **phase3/** — Flash Attention variants: scalar → 4-warp → Br=16 HMMA (19x speedup)
- **phase4/** — Diffusion UNet: `timestep_emb/`, `groupnorm/`, `conv2d/`, `resblock/`, `cross_attention/`
- **phase2/common/** — Shared headers (`bench.h`, `check.h`) included by all benchmarks
- **docs/** — Deep technical references (SASS instruction set, control codes, memory hierarchy, optimization postmortems)
- **scripts/** — `build.R` (compile / disasm / roundtrip), `verify_setup.R`,
  `install_cuasmR.R`, NCU + bench harnesses
- **R/cuasmR/** — local R package replacing upstream CuAssembler (#102);
  reads cubins via nvdisasm, byte-patches `.text` slots

### Shared Headers

- `phase2/common/bench.h` — `BenchTimer` (CUDA events), `CHECK_CU` macro, GFLOPS calculation
- `phase2/common/check.h` — `check_fp32()` correctness verification against CPU reference

## Correctness Verification

Every kernel has a `bench.cu` that runs a CPU reference, launches the GPU kernel, and compares with `check_fp32()`. The check uses **AND logic**: an element fails only if BOTH absolute AND relative error exceed tolerance.

Tolerance conventions by precision:
- FP32 scalar kernels: `abs=1e-3, rel=1e-3`
- FP16 Tensor Core (HMMA): `abs=1e-2, rel=1e-2`
- `--use_fast_math` (sin/cos): `abs=5e-4, rel=5e-4`
- Conv2d (9x reaccumulation): `abs=1e-2, rel=1e-2`
- INT8 Tensor Core (IMMA, symmetric quant): `abs=0.5, rel=0.1`

## Performance Measurement

- **GEMM GFLOPS**: `(2 * M * N * K) / (time_ms / 1000) / 1e9`
- **Bandwidth**: `total_bytes / (time_ms / 1000) / 1e9`
- All timing uses CUDA events (`BenchTimer`), not wall-clock
- Warmup runs precede measured runs

## The Four Laws of Making GA104 Happy

1. **Feed Tensor Cores continuously** — overlap loads with HMMA/IMMA. At 8+ warps, cp.async benefit depends on compute/load ratio: helpful when short (8 IMMA/tile, +35%), harmful when long (64 HMMA/tile, -5%). See `gpu_reflections.md` Insights 2, 14.
2. **Read each byte of DRAM exactly once** — im2col converts 9x re-reads to 1x; implicit GEMM eliminates the col buffer entirely.
3. **Fill the warp schedulers** — 32 warps/SM ideal, 8 sufficient. Below 8 = structural problem.
4. **Never cross the 50 KB smem cliff** — >50 KB/block → 1 block/SM → exposed DRAM stalls. (GA104 has 100 KB max smem/SM; cliff = 100/2 = 50 KB/block.)

## Key SASS Instructions

- **HMMA.16816.F32** — Tensor Core (16x8x16, FP16→FP32, warp-wide). S08 stall between consecutive HMMAs is a hardware pipeline constraint, not reducible.
- **SHFL.BFLY** — Warp butterfly reduction (5 instructions for 32-lane reduce)
- **MUFU.EX2/RCP/RSQ/SIN/COS** — Special function unit (~16 cycle latency). `exp(x) = 2^(x * log2(e))` requires FMUL + MUFU.EX2.
- **FFMA** — FP32 fused multiply-add (core of non-tensor GEMM)
- **IMMA.16816.S8.S8** — INT8 Tensor Core (16x8x16 sub-tile, INT8→INT32). Two per PTX mma.sync, four per WMMA mma_sync call. Dequantized via I2FP.F32.S32 + FMUL. Unlike HMMA's S08, IMMA sustains S01 throughput when operands are ready — S04 stalls from compiler are conservative (S02 is optimal).
- **LDGSTS** (cp.async) — Async global→shared copy, bypasses register file

## Current State and Next Steps

All 5 phases complete. See `docs/CONTINUE_HERE.md` for session state, benchmark results table, and prioritized next steps (split-Q Flash Attention, INT8 IMMA, persistent kernel grid). See `docs/gpu_reflections.md` for optimization postmortems.

## Team Activation (REQUIRED when user requests a team)

When the user asks to activate or use a team: (1) call `ToolSearch("select:TeamCreate")` to load the TeamCreate tool, (2) read the team definition from `/mnt/d/dev/p/agent-almanac/teams/<team-name>.md`, (3) call `TeamCreate` with the team configuration. Do NOT fall back to spawning individual agents via the Agent tool — always use TeamCreate for team requests.

Available teams are listed in `/mnt/d/dev/p/agent-almanac/teams/_registry.yml`.
