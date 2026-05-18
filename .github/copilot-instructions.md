# bare-metal Copilot instructions

## Build and test commands

This repository is developed in **WSL2/Linux** and targets **GA104 / `sm_86`**. The standard top-level entry points are:

```bash
make setup       # restore renv packages and install the local cuasmR package
make verify      # verify CUDA, GPU visibility, renv, and cuasmR
make all         # build all kernel cubins and benchmark executables
make bench       # run benchmark regression checks against docs/baselines.json
make test        # smoke-test compiled GEMM/reduction/elementwise benches
make reproduce   # setup -> verify -> all -> bench
make clean       # remove generated cubins, sass dumps, and bench executables
make disasm      # disassemble built cubins through scripts/build.R
```

Useful narrower build/test commands:

```bash
make tutorial
make gemm
make reductions
make elementwise
make attention
make convolution
make memory_layout
make composition

Rscript scripts/build.R compile   kernels/tutorial/vector_add.cu
Rscript scripts/build.R disasm    kernels/tutorial/vector_add.sm_86.cubin
Rscript scripts/build.R roundtrip kernels/tutorial/vector_add.cu

Rscript scripts/bench/bench_regress.R --kernel kernels/gemm/hgemm/hgemm_16warp.cu
Rscript scripts/bench/bench_regress.R --list

Rscript tests/bench_regress/test_parser.R
Rscript tests/bench_regress/test_meta.R
Rscript -e 'library(testthat); library(cuasmR); test_dir("R/cuasmR/tests/testthat")'
```

There is **no dedicated lint target**. The repo’s automated gate is the pre-push workflow installed by `bash scripts/install-hooks.sh`, which runs `make test`, a README link audit, and `scripts/bench/bench_regress.R`.

## High-level architecture

The core workflow is:

```text
CUDA .cu -> nvcc -> .cubin -> cuobjdump/nvdisasm inspection -> optional cuasmR byte patch -> benchmark executable -> docs/baselines.json regression check
```

- `kernels/` is the main product surface. Kernels are organized by **family/content** (`gemm`, `reductions`, `attention`, `convolution`, `elementwise`, `memory_layout`, `composition`, `tutorial`) rather than by phase. Each kernel directory usually contains the kernel `.cu`, one or more `bench*.cu` harnesses, and a README with measured results.
- `kernels/_common/` holds the shared harness pieces used across benches: `bench.h` for Driver API timing/warmup/GFLOPS helpers and `check.h` for CPU-reference correctness checks.
- Bench executables load cubins through the **CUDA Driver API**, not the Runtime API. The bench process is expected to run from the kernel’s own directory because modules are usually loaded by relative cubin filename.
- `R/cuasmR/` is a repo-local R package that replaced the older Python CuAssembler flow. It does **byte-level patching of existing instruction/control words inside `.text` sections**; it does not assemble edited SASS text back into a cubin.
- `scripts/` is mostly R tooling around that kernel pipeline: `build.R` for compile/disasm/roundtrip, `bench/bench_regress.R` for baseline comparison, `profile/` and `audit/` for measurement/report generation, and model scripts for occupancy/shared-memory analysis.
- `docs/baselines.json` is part of the executable architecture, not just documentation: it maps kernel source paths to the correct bench executable, output-matching rules, tolerances, and fair-run metadata requirements.
- `experiments/` has its own build conventions and is intentionally excluded from the default `Makefile` sweep.

## Key conventions

- The repository is **hardware-specific**: assume **RTX 3070 Ti / GA104 / `sm_86`** unless a file explicitly says otherwise. The “50 KB shared-memory cliff” is a load-bearing constraint across many kernels and tile-size decisions.
- Tooling under `scripts/` and `R/` is **R-first**. Do not reintroduce Python-based workflow assumptions for cubin editing or benchmark automation unless the repo already uses them in that area.
- Do **not** modify system CUDA / driver installations from repository work. If a workflow needs adjustment, prefer repo-local scripts, wrappers, or environment fixes.
- `.Rprofile` auto-activates `renv`, and R subprocesses in WSL must keep `/usr/lib/wsl/lib` visible in `LD_LIBRARY_PATH` so `libcuda.so` is found. Existing R scripts already handle this for common entry points.
- Kernel entry points are expected to use `extern "C"`, `__launch_bounds__`, `__restrict__`, and `size_t` for large index math. Dynamic shared memory is typically declared as `extern __shared__ char smem_raw[]`.
- Every performance kernel should have a `bench.cu` or `bench_*.cu` harness with:
  - CUDA Driver API launches
  - a CPU reference correctness check
  - warmup before timed runs
  - documented tolerances
  - output in the parser-friendly `label  ms  GFLOPS/TOPS` style used by `bench_regress.R`
- Correctness checks use the repo’s `check_fp32()` convention: a value is considered wrong only when **both** absolute and relative error exceed tolerance.
- New kernel directories are expected to carry a README with the kernel purpose, measured results, copy-pasteable build commands, and links back to relevant docs/postmortems.
