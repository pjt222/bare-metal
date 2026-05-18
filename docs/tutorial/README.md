# Bare-Metal GPU Programming — Tutorial Series

A pedagogical walkthrough of writing high-performance GPU kernels for the
RTX 3070 Ti (GA104, sm_86, Ampere) without using cuBLAS, cuDNN, or PyTorch.
Built from the empirical work in this repository: 5 phases of increasingly
sophisticated kernels, 17+ optimization postmortems, and several
instructive failures.

All six chapters are full prose, ~80 KB total. Each chapter is
self-contained and cross-references the actual kernels and benchmark
numbers in this repository. The series fits a 1-2 day reading.
Chapters can be read in any order; 02 → 03/04 → 05 is the suggested
dependency chain, with 06 as a synthesis chapter that can be read
first or last.

## Chapters

| # | Title | Topic |
|---|---|---|
| 01 | [SASS Hello World](01-sass-hello-world.md) | FADD→FMUL hand-edit, CuAssembler roundtrip, why hand-editing matters |
| 02 | [GEMM from Scratch](02-gemm-from-scratch.md) | naive → tiled → register-blocked → HMMA → 16-warp HGEMM (31910 GFLOPS) |
| 03 | [INT8 Tensor Cores](03-int8-tensor-cores.md) | IMMA, online quant, 2:4 sparse (35509 dense-equiv TOPS) |
| 04 | [Software Pipelining](04-software-pipelining.md) | cp.async, regime-dependent wins, before/after FA case study |
| 05 | [Flash Attention](05-flash-attention.md) | online softmax, fragment-shfl reductions, occupancy tradeoffs, 3 instructive failures |
| 06 | [The Four Laws](06-the-four-laws.md) | Empirical principles distilled from all experiments |

## Reading Order

The chapters can be read in two ways:

- **Top-down** (start with 06): if you want the principles first and then the
  experiments that produced them, read chapter 06 first. It is self-contained
  and references the others as evidence.
- **Bottom-up** (start with 01): if you want to follow the chronological
  development, start with chapter 01 (SASS hand-editing) and work forward.
  Each chapter assumes the previous.

## Prerequisites

- Working CUDA 12.x install with `nvcc`, `cuobjdump`, `nvdisasm` on PATH
- An sm_86-class GPU (RTX 30-series, RTX A-series, etc.)
- Familiarity with C++ and basic CUDA programming model (thread, block, warp, smem)
- Linux or WSL2 environment

## Source Material

Each tutorial cross-references actual files in this repository:

- `kernels/tutorial/` — the SASS hello world (vector_add)
- `kernels/gemm/{sgemm,hgemm,hgemm_sparse,igemm}/` — GEMM variants
- `kernels/attention/flash_attention/` — Flash Attention implementations
- `kernels/{convolution,reductions,elementwise}/` — diffusion primitives
- `kernels/composition/` — multi-kernel layer composition
- `kernels/memory_layout/cymatic/` — Chladni-pattern memory layout study
- `docs/gpu_reflections.md` — empirical observations (17+ postmortems)
- `docs/{ampere_sass_reference,control_codes,memory_hierarchy}.md` — hardware references
- `docs/fragment_shfl_reductions.md` — reusable Tensor Core reduction pattern

## Conventions

- All performance numbers measured on RTX 3070 Ti Laptop GPU (8 GB)
- All builds use `nvcc -arch=sm_86 -O2`
- All correctness checks use `kernels/_common/check.h` AND-logic tolerances:
  fail only if BOTH absolute AND relative error exceed threshold
- Code blocks are tested; if a snippet doesn't compile, file an issue

## Series structure

Chapter | Topic | Length
---|---|---
01 SASS Hello World | toolchain, FADD→FMUL hand-edit | 12 KB
02 GEMM from Scratch | naive → 16-warp HGEMM (31910 GFLOPS) | 14 KB
03 INT8 Tensor Cores | IMMA, online quant, 2:4 sparse | 14 KB
04 Software Pipelining | cp.async regime analysis | 14 KB
05 Flash Attention | 9 versions, smem_work elim, regime splits | 15 KB
06 The Four Laws | synthesis chapter | 13 KB

Total ~80 KB / ~20,000 words. Each chapter has "Try it yourself" build
commands that run against the actual repo files.

## Conventions across chapters

- Performance numbers are post-warmup, 3-run mean unless otherwise noted
- Failed experiments are documented as case studies, not omitted
- Each chapter cross-references `docs/gpu_reflections.md` observations
- Code blocks compile against the repo as-is
