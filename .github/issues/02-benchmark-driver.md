---
title: "Extract common benchmark driver to eliminate boilerplate duplication"
labels: ["enhancement", "refactoring"]
---

## Problem
Every `bench.cu` duplicates ~100 lines of CUDA context setup, memory allocation, fill, CPU reference, timing loops, and correctness checks. At 26 bench files, this is ~2,600 lines of duplicated logic.

## Current Pattern (repeated 26×)
```cpp
// Context init
CUdevice dev; CUcontext ctx;
cuInit(0); cuDeviceGet(&dev, 0); cuCtxCreate(&ctx, 0, dev);

// Allocation
cuMemAlloc(&d_A, size_A); cuMemAlloc(&d_B, size_B); // ...

// Fill
cpu_ref = ...; fill_random(h_A, ...);  // memcpy to device

// CPU reference
for (int i = ...) { /* triple-nested SGEMM */ }

// Warmup + timing
WARMUP(3, cuLaunchKernel(...));
float ms = BENCH(10, cuLaunchKernel(...));

// Correctness
cuMemcpyDtoH(h_C, d_C, ...);
CheckResult r = check_fp32(h_C, cpu_ref, ...);
```

## Proposed Solution
Create `kernels/_common/bench_driver.h` with template/struct-based driver:

```cpp
// Usage in bench.cu:
int main(int argc, char** argv) {
    KernelParams p = parse_args(argc, argv);
    BenchDriver driver(p.M, p.N, p.K);
    
    // Register kernel and reference
    driver.set_kernel(launch_igemm,   "igemm_pipelined");
    driver.set_reference(cpu_sgemm);
    
    // Run
    driver.warmup(3);
    driver.run(10);
    driver.check(/*abs_tol=*/1e-2f, /*rel_tol=*/1e-2f);
    driver.print_results();
}
```

## Key Design Decisions
- Driver must remain header-only (no link step complications)
- Support both cuBLAS-less mode (CPU reference) and optional cuBLAS comparison
- Templated on input/output types (FP32, FP16, INT8)
- RAII for context/memory/events to avoid segfault-on-exit (see troubleshooting.md)

## Files to Refactor (priority order)
1. `kernels/gemm/igemm/bench.cu` — most complex, tests multiple variants
2. `kernels/attention/flash_attention/bench.cu` — many variants, good test case
3. `kernels/gemm/hgemm/bench.cu` — simplest, good reference
4. Remaining 23 bench files

## Acceptance Criteria
- [ ] `bench_driver.h` exists in `kernels/_common/`
- [ ] At least 3 bench files refactored to use it
- [ ] No functionality lost (same timing, same correctness checks)
- [ ] PR includes before/after line count comparison

## Effort
Medium — 1–2 days for design, 2–3 days for refactor of priority files.
