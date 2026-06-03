# bare-metal GPU

> **Hand-optimized SASS assembly kernels targeting RTX 3070 Ti (GA104, sm_86, Ampere).**
> No cuBLAS, no cuDNN, no PyTorch. Just nvcc, cuobjdump, and our R-native cubin patcher (cuasmR).

This repository builds a library of ML kernels — GEMM, Flash
Attention, convolution, normalization, sparse 2:4 and INT8 variants —
from `nvcc` down to native SASS, with no vendor library in the
optimized path. Every kernel is compiled to a `cubin`, disassembled
with `cuobjdump`, optionally hand-edited at the byte level via the
local R package [cuasmR](docs/cuasm_r.md), reassembled, and run on a
real RTX 3070 Ti Laptop. Performance numbers are measured (median of
11 runs after 5 warmup iterations), not extrapolated.

The accessible stack ends at SASS — below it the driver and firmware
are cryptographically sealed:

```
CUDA C/C++          ← you write this
     │ nvcc
PTX (Virtual ISA)   ← documented, portable, stable ABI
     │ ptxas (driver JIT)
SASS (Native ISA)   ← sm_86, undocumented, reverse-engineered     ← WE WORK HERE
     │ [SIGNATURE WALL — cryptographic, cannot cross]
Driver / Firmware   ← locked
```

The headline result: a **2:4 structured-sparse HGEMM at 41,721
dense-equivalent GFLOPS** (24% of the GA104 FP16 Tensor Core peak),
reached through a chain of single-mechanism optimizations that
compound to ~90× over a textbook GEMM. The full per-kernel breakdown,
phase progression, and Flash Attention waterfall live in
[`docs/inventory.md`](docs/inventory.md).

The optimization work is distilled into the **four laws of GA104** —
feed the Tensor Cores, read each DRAM byte once, fill the warp
schedulers, never cross the 50 KB shared-memory cliff. They are
derived empirically and explained in
[`docs/tutorial/06-the-four-laws.md`](docs/tutorial/06-the-four-laws.md).

## Documentation

The repository is organized like a paper: this README is the
abstract; the detailed sections live under `docs/`. Start at the
documentation map and follow the dependency order from there.

- [`docs/index.md`](docs/index.md) — full documentation map, read this first.
- [`docs/inventory.md`](docs/inventory.md) — kernel inventory by
  family, with headline performance, phase progression, and the
  Flash Attention waterfall.
- [`docs/comparison_to_sota.md`](docs/comparison_to_sota.md) —
  measured gap to local cuBLAS / cuDNN / cuSPARSELt.
- [`docs/roofline_measured.md`](docs/roofline_measured.md) —
  NCU-measured roofline per profiled kernel.
- [`docs/tutorial/`](docs/tutorial/) — six-chapter prose walkthrough
  (~20K words); suggested order 02 → 03/04 → 05 with 06 as synthesis.
- [`docs/gpu_reflections.md`](docs/gpu_reflections.md) — observation
  catalogue. The first-person voice is a deliberate stylistic
  experiment; see its preamble.
- [`docs/cymatic_memory_mapping.md`](docs/cymatic_memory_mapping.md) —
  Chladni-pattern memory layout study; a conditional layout win on
  mode-aligned access, loss on nodal-line access.
- [`AGENTS.md`](AGENTS.md) — canonical agent-facing reference:
  hardware constants, toolchain, build entry points, code
  conventions, and the SASS hand-edit workflow.

## Setup and reproduction

[`SETUP.md`](SETUP.md) covers the environment install. After CUDA + R
are installed system-wide, the project reproduces in two commands:

```bash
git clone https://github.com/pjt222/bare-metal.git && cd bare-metal
make reproduce          # setup + verify + build + bench-vs-baselines
```

`make reproduce` chains `setup` (renv restore + cuasmR install),
`verify` (env check), `all` (compile every cubin + bench), `bench`
(run benches and compare against `data/baselines.json`; this stage
prints `RESULT: PASSED -- all benchmarks within tolerance`), and
`figures` (regenerate `docs/figures/`). The run finishes with the
`Full reproduction complete` banner. The pre-push hook calls the same
regression check. Build entry points are documented in
[`AGENTS.md`](AGENTS.md).

## License

MIT. See [LICENSE](LICENSE).
