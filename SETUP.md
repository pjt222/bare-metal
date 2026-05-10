# Environment Setup

> **Development happens in WSL2.** The Windows NVIDIA driver exposes
> the GPU to WSL via `/usr/lib/wsl/lib/`. All build scripts,
> benchmarks, and SASS tooling assume a Linux environment. There is
> no native-Windows build path; the Windows section of an earlier
> revision was removed when the project moved off Python tooling.

## TL;DR (after CUDA + R are installed system-wide)

```bash
git clone https://github.com/pjt222/bare-metal.git
cd bare-metal
make reproduce        # setup + verify + build + bench-vs-baselines
```

The `make reproduce` target runs:

1. `make setup`   — `renv::restore()` + install local cuasmR R package
2. `make verify`  — env check (CUDA, GPU, cuasmR)
3. `make all`     — compile every kernel `.cu` to `.cubin` + every bench `.cu` to executable
4. `make bench`   — run all benches, compare against `docs/baselines.json`

A clean run on a healthy GPU ends with `RESULT: PASSED -- all
benchmarks within tolerance`. The pre-push git hook calls the same
regression check, so what passes locally also passes the hook.

## System prerequisites (one-time)

These are OS-level dependencies and live outside the project; the
project's `renv` library doesn't manage them.

### 1. CUDA 13.2 toolkit (in WSL Ubuntu 24.04)

```bash
sudo apt update
sudo apt install nvidia-cuda-toolkit
```

Verify:

```bash
nvcc --version            # expect: Cuda compilation tools, release 13.2
cuobjdump --version
nvdisasm --version
nvidia-smi                # expect: NVIDIA RTX 30-series Ampere or compatible
```

The Windows-side NVIDIA driver supplies `libcuda.so` via
`/usr/lib/wsl/lib/`. R subprocesses don't see this path by default;
project scripts (`bench_regress.R`, `bench_meta.R`, `verify_setup.R`,
`build.R`) prepend it to `LD_LIBRARY_PATH` automatically.

> The project compiles to `sm_86` (GA104, RTX 3070 Ti). Other
> sm-versions need a one-line edit to the top-level `Makefile`
> (`SM_ARCH := sm_XX`) and a re-baseline pass.

### 2. R 4.6.0

```bash
# Ubuntu 24.04 — R 4.6 from CRAN
wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
sudo add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
sudo apt install r-base r-base-dev
```

Verify:

```bash
R --version               # expect: R version 4.6.x
```

The `.Rprofile` at the repo root auto-activates the project's renv
library on every `Rscript` invocation. No system-wide R packages are
required; everything is pinned in `renv.lock`.

### 3. (optional, GPU profiling) Nsight Compute 2026.1+

Required only for `scripts/profile/ncu_profile_all.sh` and friends.
Comes with CUDA 13.2 toolkit; no separate install.

## Project setup (per clone)

```bash
git clone https://github.com/pjt222/bare-metal.git
cd bare-metal
make setup                # renv::restore() + cuasmR install
make verify               # confirms toolchain
```

`make setup` does:

- `Rscript -e 'if (!requireNamespace("renv", ...)) install.packages("renv"); renv::restore()'` — installs all R deps from `renv.lock`. First time per machine: ~3-5 minutes; subsequent: instant.
- `Rscript scripts/install_cuasmR.R` — installs the local `R/cuasmR/` package into the renv library.

`make verify` runs `scripts/verify_setup.R` which checks:

- `nvcc`, `cuobjdump`, `nvdisasm` on PATH
- `nvidia-smi` reaches the GPU
- `libcuda.so` is loadable from R subprocesses (the WSL passthrough)
- `cuasmR` package loads
- renv library is on the search path

Any FAIL in `make verify` blocks the rest. The script's output names
the exact remediation (e.g. "add CUDA bin to PATH", "rerun renv
restore").

## Building

```bash
make all                  # all cubins + all benches
make phase2               # one phase only
make clean                # remove generated artifacts
make disasm               # cubins -> SASS dumps
```

`scripts/` and `experiments/` are excluded from the default sweep
(they have their own build conventions).

## Running benches

```bash
make bench                # all kernels vs docs/baselines.json
```

Equivalent to `Rscript scripts/bench/bench_regress.R`. Each kernel
runs once, GPU + host state is captured before and after, and the
result is compared against the recorded baseline. Three verdicts:

- `OK` — within tolerance (default 10%, per-config override
  available)
- `IMPROVED` / `REGRESSION` — outside tolerance, real change
- `SKIPPED` — measurement-time GPU state was unfair (thermal
  throttle, sw power cap, etc.); the run is dropped, not failed.
  See [Tier 10 doc in baselines.json schema](docs/baselines.json).

To list recorded baselines without running:

```bash
Rscript scripts/bench/bench_regress.R --list
```

## SASS hand-edit workflow (cuasmR)

```bash
# CLI driver
Rscript scripts/build.R compile   kernel.cu             # .cu -> .cubin
Rscript scripts/build.R disasm    kernel.sm_86.cubin    # .cubin -> .cuasm dump
Rscript scripts/build.R roundtrip kernel.cu             # compile + read+write byte-identical check
```

Programmatic edits via the `cuasmR` package — see
[`docs/cuasm_r.md`](docs/cuasm_r.md).

## Hardware constraint

Project targets **GA104 (sm_86, RTX 3070 Ti)** specifically. The "50
KB shared-memory cliff" in the Four Laws is hardware-specific (cliff
= max-smem-per-SM ÷ 2). On other Ampere parts the cliff sits
elsewhere; the kernels still run but tile-size decisions may be
suboptimal.

Re-baselining on different hardware is mandatory:

```bash
# Edit Makefile: SM_ARCH := sm_XX
make clean && make all
# Manually verify each kernel runs, then capture new numbers:
Rscript scripts/bench/bench_regress.R --list  # see what to measure
# Run each bench, record into docs/baselines.json by hand
```

## Troubleshooting

**`nvcc: command not found`**:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
```

**`make verify` says GPU not detected** (but `nvidia-smi` works in the shell):

R subprocesses strip `/usr/lib/wsl/lib/` from `LD_LIBRARY_PATH`. The
project scripts re-add it automatically; if you're running benches
manually:

```bash
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
./phase2/hgemm/bench 2048 2048 2048
```

**`make bench` reports SKIPPED for kernels you wanted measured**:

The GPU was being throttled (power cap, thermal). Wait a minute,
re-run. On a laptop on battery, plug into AC. The schema
`require_ac` lets per-kernel runs require AC.

**`renv::restore()` complains about missing system libraries**:

R packages with C++ bindings (e.g. `cli`, `xml2`) need development
headers. Install:

```bash
sudo apt install libxml2-dev libssl-dev libcurl4-openssl-dev libfontconfig1-dev libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev
```

**Pre-push hook fails with REGRESSION**:

Either the change you're pushing is a real perf regression, or the
GPU is in a weird state. Re-run `make bench` to see the verdict; if
SKIPPED items were the cause, the hook will accept on retry. To
override (e.g. for WIP commits):

```bash
git push --no-verify
```

**`make all` fails on a single kernel**:

Build is best-effort (`make -k all`) so other kernels keep
compiling. Inspect the failed file's directory for kernel-specific
notes; many `phase4/conv2d/debug*.cu` files are intentionally
unbuildable scratchpads (gitignored).
