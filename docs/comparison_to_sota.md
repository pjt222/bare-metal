# Local reference comparison pipeline

> Legacy filename retained for stable links. This document no longer uses
> extrapolated "SOTA estimate" tables. It now tracks only **locally measured
> reference implementations** on this exact machine.

## Why this changed

The repo previously mixed:

- **measured local project baselines** in `data/baselines.json`
- **estimated external reference values** in README/SOTA docs

That was not robust enough. The current policy is:

1. **Project numbers** must be measured locally.
2. **Reference numbers** must also be measured locally.
3. If a local reference stack is unavailable, the result is reported as
   **not measured locally**, not estimated.

## Current local reference coverage

Available on this machine:

- **cuBLAS / cuBLASLt**
- **cuSPARSELt**
- **cuDNN**

Still missing as direct local reference paths:

- **cuDNN graph-SDPA frontend support in the installed headers**
- **a direct GroupNorm harness**

So the reproducible local-reference pipeline now covers dense GEMM,
sparse INT8 GEMM, and the main 3x3 conv2d anchor.

## Commands

```bash
make reference          # build local reference-library benches
make bench-reference    # validate data/reference_baselines.json
make compare-reference  # join data/baselines.json to data/reference_baselines.json
```

Low-level entry points:

```bash
Rscript scripts/bench/bench_reference.R
Rscript scripts/bench/compare_reference.R
```

## Data files

- `data/baselines.json` — measured local project baselines
- `data/reference_baselines.json` — measured local reference-library baselines

Both files use the same fair-run policy:

- reject throttled / unfair runs
- keep exact config tuples
- record values on this exact GA104 / RTX 3070 Ti Laptop setup

## Measured local comparison

Current measured local reference rows:

| Workload | Ours | Local reference | % of reference | Gap |
|---|---:|---:|---:|---:|
| HGEMM 16-warp 2048³ | 31,875 GFLOPS | 28,631 GFLOPS (cuBLAS) | **111.3%** | 0.90x |
| HGEMM 16-warp 4096³ | 31,765 GFLOPS | 29,708 GFLOPS (cuBLAS) | **106.9%** | 0.94x |
| IGEMM pipelined cp.async 4096³ | 20.23 TOPS | 29.44 TOPS (cuBLAS) | **68.7%** | 1.46x |
| Sparse IGEMM tiled 2048³ | 31.59 TOPS | 124.28 TOPS (cuSPARSELt) | **25.4%** | 3.93x |
| Sparse IGEMM tiled 4096³ | 30.89 TOPS | 170.11 TOPS (cuSPARSELt) | **18.2%** | 5.51x |
| Conv2d implicit GEMM 1×64×64×320×320 | 7,150 GFLOPS | 16,910 GFLOPS (cuDNN) | **42.3%** | 2.37x |

These are the only rows that are currently both:

1. implemented as local reference-library harnesses
2. recorded in `data/reference_baselines.json`
3. matched to project baselines in `data/baselines.json`

## Explicitly unsupported today

These workloads are intentionally **not** filled with guesses:

| Workload | Missing local stack | Status |
|---|---|---|
| Flash Attention seq=1024 b=8 h=8 d=64 | cuDNN SDPA | not measured locally |
| GroupNorm SD 320ch | cuDNN | not measured locally |

## Interpretation

The important result is not a single headline gap factor anymore; it is
that the comparison is now **reproducible and machine-local**.

On this machine and under this pipeline:

- the current HGEMM anchor outperforms the recorded local cuBLAS path for
  the two square GEMM anchors we measure here
- the current dense INT8 anchor still trails local cuBLAS
- the current sparse INT8 anchor trails local cuSPARSELt by 3.9x at 2048³
  and 5.5x at 4096³
- the current conv2d implicit GEMM anchor reaches 42.3% of the measured
  local cuDNN reference
- unsupported stacks remain blank until a real local harness exists

That is a much stronger basis for engineering decisions than the earlier
estimate-based table.

## Extending coverage

To add a new local reference row:

1. add a new reference bench executable under `kernels/reference/`
2. add its measured local baseline to `data/reference_baselines.json`
3. map it to a project baseline via `project_kernel` + `project_config`
4. rerun `make bench-reference` and `make compare-reference`

The next obvious additions are:

1. direct local attention via a cuDNN SDPA-capable frontend on this machine
2. a direct GroupNorm reference harness
3. more cuDNN conv2d shapes beyond the current 64×64×320 anchor
