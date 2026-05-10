/*
 * bench.cu — Activations benchmark: GELU, SiLU, Fast-GELU, ReLU (BenchDriver refactor)
 *
 * Build:
 *   nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../common
 *
 * Usage:
 *   ./bench              # default: 16M elements
 *   ./bench 33554432     # 32M elements
 *
 * Expected SASS:
 *   cuobjdump -sass activations.sm_86.cubin | grep MUFU
 *   → MUFU.TANH  (gelu_kernel)
 *   → MUFU.EX2   (silu_kernel, gelu_fast)
 *   → MUFU.RCP   (silu_kernel, gelu_fast)
 */

#include <cuda.h>
#include <cstdio>
#include <cmath>
#include "../../kernels/_common/bench_driver.h"

// -----------------------------------------------------------------------
// CPU references
// -----------------------------------------------------------------------
static float cpu_gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    return x * 0.5f * (1.0f + tanhf(inner));
}

static float cpu_silu(float x) {
    return x / (1.0f + expf(-x));
}

static float cpu_gelu_fast(float x) {
    float sigmoid = 1.0f / (1.0f + expf(-1.702f * x));
    return x * sigmoid;
}

static float cpu_relu(float x) {
    return fmaxf(x, 0.0f);
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------
int main(int argc, char **argv) {
    int num_elements = (argc > 1) ? atoi(argv[1]) : 16 * 1024 * 1024;

    printf("=== Activation Functions — MUFU.TANH / MUFU.EX2 / MUFU.RCP ===\n");
    printf("Elements: %d (%.1f M)\n\n", num_elements, num_elements / 1.0e6f);

    BenchDriver driver;
    driver.init_context();

    size_t total_bytes = (size_t)num_elements * sizeof(float);
    auto d_input  = driver.device_alloc<float>(num_elements);
    auto d_output = driver.device_alloc<float>(num_elements);
    auto h_input  = driver.host_alloc<float>(num_elements);
    auto h_ref    = driver.host_alloc<float>(num_elements);
    auto h_out    = driver.host_alloc<float>(num_elements);

    fill_random(h_input.get(), num_elements, 42);
    for (int i = 0; i < num_elements; i++) h_input[i] *= 3.0f;
    driver.copy_h2d(d_input, h_input, total_bytes);

    const int block_size = 256;
    const int elements_per_thread = 4;
    dim3 grid((num_elements + block_size * elements_per_thread - 1)
              / (block_size * elements_per_thread), 1, 1);
    dim3 block(block_size, 1, 1);

    struct V {
        const char *name, *sym;
        float (*cpu_ref)(float);
        float abs_tol, rel_tol;
        const char *note;
    };
    std::vector<V> variants = {
        {"relu_kernel",  "relu_kernel",      cpu_relu,      1e-6f, 1e-5f,
         "FMNMX only — memory bound ceiling"},
        {"gelu_fast",    "gelu_fast",        cpu_gelu_fast, 1e-4f, 1e-3f,
         "MUFU.EX2 + MUFU.RCP"},
        {"silu_kernel",  "silu_kernel",      cpu_silu,      1e-4f, 1e-3f,
         "MUFU.EX2 + MUFU.RCP"},
        {"gelu_kernel",  "gelu_kernel",      cpu_gelu,      1e-4f, 1e-3f,
         "MUFU.TANH + FFMA"},
    };

    double gbytes = 2.0 * total_bytes / 1e9;

    // Load cubin once
    CUfunction first = driver.load_kernel("activations.sm_86.cubin", variants[0].sym);
    (void)first;  // module loaded, functions cached

    printf("Correctness:\n");
    for (auto &v : variants) {
        CUfunction fn = driver.load_kernel("activations.sm_86.cubin", v.sym);
        void *args[] = { &d_input, &d_output, &num_elements };

        // CPU reference
        for (int i = 0; i < num_elements; i++) h_ref[i] = v.cpu_ref(h_input[i]);

        // GPU correctness
        CHECK_CU(cuMemsetD32((CUdeviceptr)d_output.get(), 0, num_elements));
        CHECK_CU(cuLaunchKernel(fn, grid.x, grid.y, grid.z,
                                block.x, block.y, block.z,
                                0, NULL, args, NULL));
        CHECK_CU(cuCtxSynchronize());
        driver.copy_d2h(h_out, d_output, total_bytes);
        driver.check(h_out.get(), h_ref.get(), num_elements,
                     v.abs_tol, v.rel_tol, v.name, false);
    }

    printf("\nPerformance (avg of 100 runs, 5 warmup):\n");
    printf("  %-20s  %8s  %8s  %s\n", "Kernel", "ms", "GB/s", "Note");
    printf("  %-20s  %8s  %8s  %s\n", "------", "--", "----", "----");

    for (auto &v : variants) {
        CUfunction fn = driver.load_kernel("activations.sm_86.cubin", v.sym);
        void *args[] = { &d_input, &d_output, &num_elements };
        float ms = driver.benchmark_kernel(fn, grid, block, 0, args, 5, 100);
        double bw = gbytes / (ms / 1000.0);
        printf("  %-20s  %7.3f ms  %7.1f GB/s  %s\n", v.name, ms, bw, v.note);
    }

    printf("\nTo inspect the key SASS instructions:\n");
    printf("  cuobjdump -sass activations.sm_86.cubin | grep MUFU\n");
    printf("  → MUFU.TANH  (gelu_kernel)\n");
    printf("  → MUFU.EX2   (silu_kernel, gelu_fast)\n");
    printf("  → MUFU.RCP   (silu_kernel, gelu_fast)\n");

    return 0;
}
