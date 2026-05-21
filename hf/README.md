---
license: mit
language:
  - en
tags:
  - cuda
  - sass
  - gpu-kernels
  - ampere
  - ga104
  - sm_86
  - optimization
  - benchmarks
pretty_name: GA104 Hand-Optimized CUDA Kernel Corpus
configs:
  - config_name: sass_histogram
    data_files: data/sass_histogram.csv
---

# GA104 Hand-Optimized CUDA Kernel Corpus

A measurement corpus of hand-optimized CUDA / SASS kernels targeting the
**RTX 3070 Ti (GA104, sm_86, Ampere)**. Every kernel is written without
cuBLAS, cuDNN, or PyTorch in the optimized path; vendor libraries are
linked only for measured comparison under `kernels/reference/`. This
dataset is for SASS and GPU-optimization researchers — it pairs each
`.cu` source with its compiled machine code and its disassembly, so the
exact instruction stream a kernel produced on this toolchain can be
studied without owning the hardware.

## The four laws of GA104

The corpus is organized around four empirically derived constraints.
The observations behind each are in `docs/gpu_reflections.md`.

1. **Feed Tensor Cores continuously.** Overlap loads with HMMA / IMMA.
   At ≥8 warps, `cp.async` benefit depends on the compute/load ratio —
   helpful when compute is short, harmful when compute is long.
2. **Read each byte of DRAM at most once per kernel.** im2col converts
   9× re-reads to 1×; implicit GEMM eliminates the column buffer.
3. **Fill the warp schedulers.** 32 warps/SM is ideal, 8 sufficient;
   below 8 indicates a structural problem.
4. **Never cross the 50 KB shared-memory cliff per block.** Blocks at
   >50 KB drop to 1 block/SM (4 warps), a measured 2× regression.

## sm_86 only

Every artifact in this corpus is compiled for **compute capability 8.6
(sm_86) only**. The cubins are not portable to other architectures and
will not load on a non-Ampere or non-GA10x device. This is deliberate:
the corpus is a single-target measurement record, not a portable
kernel library. There are no fat binaries and no PTX-only fallbacks.

## Provenance

This corpus was built and published by `scripts/publish_hf.R` in the
source repository. The stamp below is filled in at publish time.

- **Source commit:** `{{COMMIT_SHA}}`
- **Build date:** `{{BUILD_DATE}}`
- **Compiler:** CUDA 13.2, `nvcc V13.2.78`
- **Cubin build line:** `nvcc -arch=sm_86 -O2 --cubin`
- **Disassembly:** `cuobjdump -sass` (via `scripts/build.R disasm`)
- **Measurement hardware:** RTX 3070 Ti Laptop (GA104, 46-SM bin)

## What the `.cubin` and `.sass` files are

`*.sm_86.cubin` files are **GPU machine code** — ELF containers of
sm_86 SASS, loaded through the CUDA Driver API (`cuModuleLoad`). They
are *not* host executables; running one on a CPU does nothing. The
matching `*.sm_86.sass` files are `cuobjdump -sass` disassembly of
those cubins, provided so the instruction stream is greppable without
a CUDA install.

The repository's local R package `cuasmR` reads a cubin via
`nvdisasm`, indexes instructions by `.text` byte offset, and patches at
the byte level — a **byte-identical roundtrip** (`compile → disasm →
reassemble`) is part of its test surface. See `docs/cuasm_r.md`.

## Headline performance

Measured on RTX 3070 Ti Laptop (GA104, sm_86, 46-SM bin), CUDA 13.2 /
nvcc V13.2.78, driver 595.97. Sparse 2:4 figures are dense-equivalent
(the multiply count the sparse pattern would do as dense work).

| Kernel                   | Size             | GFLOPS / TOPS   | % peak |
|--------------------------|------------------|-----------------|--------|
| Sparse HGEMM 2:4         | 2048³            | **41,721** (eq) | 24.0%  |
| Sparse INT8 mma.sp       | 2048³            | 39,674 (eq)     | 11.4%  |
| HGEMM 16-warp            | 4096³            | 31,910          | 18.3%  |
| IGEMM 128×256            | 4096³            | 27,591          | 7.9%   |
| Flash Attention v2       | seq=1024 b=8 h=8 | 11,453          | 6.6%   |
| Conv2d implicit GEMM     | 64×64, 320ch     | 6,687           | 3.8%   |

The headline figure is **41,721 GFLOPS dense-equivalent** for sparse
2:4 HGEMM at 2048³. Full per-kernel tables and the phase progression
(naive SGEMM → sparse INT8, 90.5× cumulative) are in
`docs/inventory.md`.

## File layout

The corpus distinguishes **tracked sources** from the **generated
supplement** rebuilt at publish time.

| Path             | Contents                                                       |
|------------------|----------------------------------------------------------------|
| `kernels/`       | Tracked `.cu` / `.cuh` kernel sources, grouped by family       |
| `kernels/**/*_handtuned.sm_86.cubin` | Tracked hand-patched cubins (SASS hand-edits) |
| `generated/`     | Rebuilt full `.sm_86.cubin` + `.sm_86.sass` set (build output) |
| `data/`          | Regenerable CSV/JSON: baselines, register audit, SASS histogram|
| `docs/`          | Researcher-facing analyses and references                      |
| `AGENTS.md`      | Hardware constants, toolchain, conventions, the four laws      |
| `SHA256SUMS`     | SHA-256 of every cubin in the corpus                           |
| `LICENSE`        | MIT                                                            |

`generated/` is build output — it is rebuilt deterministically from
the tracked `kernels/` sources with the build line above. Treat
`kernels/` as canonical and `generated/` as a convenience supplement.

## Reproduction

The single source of truth and the only supported reproduction path is
the GitHub repository:

**https://github.com/pjt222/bare-metal**

From a fresh clone on an sm_86 host: `make reproduce` (setup → verify →
build → bench). This dataset is a published snapshot; it is not the
build system.

## License

MIT. See `LICENSE`. Vendor libraries linked under
`kernels/reference/` for comparison are *not* redistributed here — only
the project's own benchmark wrappers are included.
