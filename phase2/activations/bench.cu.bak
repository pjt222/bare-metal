/*
 * bench.cu — Activations benchmark: GELU, SiLU, Fast-GELU, ReLU
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
 *   → MUFU.TANH  (gelu_kernel — hardware tanh)
 *   → MUFU.EX2   (silu_kernel, gelu_fast — hardware 2^x)
 *   → MUFU.RCP   (silu_kernel, gelu_fast — hardware reciprocal)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda.h>

#include "../common/bench.h"
#include "../common/check.h"

// -----------------------------------------------------------------------
// CPU reference implementations
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
// Benchmark helper — launches kernel, measures throughput in GB/s
// -----------------------------------------------------------------------
static float bench_kernel(
    CUfunction func,
    CUdeviceptr dev_input, CUdeviceptr dev_output,
    int num_elements,
    int block_size, int elements_per_thread,
    int warmup_iters, int bench_iters
) {
    int grid_size = (num_elements + (block_size * elements_per_thread) - 1)
                    / (block_size * elements_per_thread);
    void *args[] = { &dev_input, &dev_output, &num_elements };

    for (int i = 0; i < warmup_iters; i++) {
        CHECK_CU(cuLaunchKernel(func, grid_size, 1, 1, block_size, 1, 1, 0, NULL, args, NULL));
    }
    CHECK_CU(cuCtxSynchronize());

    float avg_ms;
    {
        BenchTimer timer;
        timer.start();
        for (int i = 0; i < bench_iters; i++) {
            CHECK_CU(cuLaunchKernel(func, grid_size, 1, 1, block_size, 1, 1, 0, NULL, args, NULL));
        }
        avg_ms = timer.stop_ms() / bench_iters;
    }
    return avg_ms;
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------
int main(int argc, char **argv) {
    int num_elements = (argc > 1) ? atoi(argv[1]) : 16 * 1024 * 1024;  // 16M default

    printf("=== Activation Functions — MUFU.TANH / MUFU.EX2 / MUFU.RCP ===\n");
    printf("Elements: %d (%.1f M)\n\n", num_elements, num_elements / 1.0e6f);

    CHECK_CU(cuInit(0));
    CUdevice cu_device;
    CHECK_CU(cuDeviceGet(&cu_device, 0));

    char device_name[256];
    CHECK_CU(cuDeviceGetName(device_name, sizeof(device_name), cu_device));
    printf("Device: %s\n\n", device_name);

    CUcontext cu_context;
    CHECK_CU(cuCtxCreate(&cu_context, 0, cu_device));

    // --- Load kernels ---
    CUmodule   act_module;
    CUfunction gelu_func, silu_func, gelu_fast_func, relu_func;
    CUresult load_result = cuModuleLoad(&act_module, "activations.sm_86.cubin");
    if (load_result != CUDA_SUCCESS) {
        const char *err_str = nullptr;
        cuGetErrorString(load_result, &err_str);
        fprintf(stderr, "Cannot load activations.sm_86.cubin: %s\n", err_str);
        fprintf(stderr, "Build with: nvcc --cubin -arch=sm_86 -O2 -o activations.sm_86.cubin activations.cu\n");
        return EXIT_FAILURE;
    }
    CHECK_CU(cuModuleGetFunction(&gelu_func,      act_module, "gelu_kernel"));
    CHECK_CU(cuModuleGetFunction(&silu_func,      act_module, "silu_kernel"));
    CHECK_CU(cuModuleGetFunction(&gelu_fast_func, act_module, "gelu_fast"));
    CHECK_CU(cuModuleGetFunction(&relu_func,      act_module, "relu_kernel"));
    printf("Activation kernels loaded.\n\n");

    // --- Host memory ---
    size_t total_bytes = (size_t)num_elements * sizeof(float);
    float *host_input  = (float *)malloc(total_bytes);
    float *host_output = (float *)malloc(total_bytes);
    float *host_ref    = (float *)malloc(total_bytes);

    // Use values in a reasonable range to test tanh numerics
    // LCG gives values in [-1, 1]; scale some entries to test wider range
    fill_random(host_input, num_elements, 42);
    for (int i = 0; i < num_elements; i++) {
        host_input[i] *= 3.0f;   // extend to [-3, 3] — typical activation range
    }

    // --- Device memory ---
    CUdeviceptr dev_input, dev_output;
    CHECK_CU(cuMemAlloc(&dev_input,  total_bytes));
    CHECK_CU(cuMemAlloc(&dev_output, total_bytes));
    CHECK_CU(cuMemcpyHtoD(dev_input, host_input, total_bytes));

    const int block_size         = 256;
    const int elements_per_thread = 4;
    const int grid_size = (num_elements + block_size * elements_per_thread - 1)
                          / (block_size * elements_per_thread);

    // --- Correctness check for each kernel ---
    printf("Correctness:\n");

    struct {
        const char    *name;
        CUfunction     func;
        float        (*cpu_ref)(float);
        float          abs_tol;
        float          rel_tol;
    } kernels[] = {
        { "gelu_kernel",  gelu_func,      cpu_gelu,      1e-4f, 1e-3f },
        { "silu_kernel",  silu_func,      cpu_silu,      1e-4f, 1e-3f },
        { "gelu_fast",    gelu_fast_func, cpu_gelu_fast, 1e-4f, 1e-3f },
        { "relu_kernel",  relu_func,      cpu_relu,      1e-6f, 1e-5f },
    };

    for (auto &k : kernels) {
        // CPU reference
        for (int i = 0; i < num_elements; i++) {
            host_ref[i] = k.cpu_ref(host_input[i]);
        }

        // GPU run
        void *args[] = { &dev_input, &dev_output, &num_elements };
        CHECK_CU(cuLaunchKernel(k.func, grid_size, 1, 1, block_size, 1, 1, 0, NULL, args, NULL));
        CHECK_CU(cuCtxSynchronize());
        CHECK_CU(cuMemcpyDtoH(host_output, dev_output, total_bytes));

        auto result = check_fp32(host_output, host_ref, num_elements, k.abs_tol, k.rel_tol, false);
        print_check_result(k.name, result);
    }

    // --- Performance benchmark ---
    int warmup_iters = 5;
    int bench_iters  = 100;
    double gbytes = 2.0 * total_bytes / 1e9;   // read input + write output

    printf("\nPerformance (avg of %d runs, %d warmup):\n", bench_iters, warmup_iters);
    printf("  %-20s  %8s  %8s  %s\n", "Kernel", "ms", "GB/s", "Note");
    printf("  %-20s  %8s  %8s  %s\n", "------", "--", "----", "----");

    struct {
        const char *name;
        CUfunction  func;
        const char *note;
    } perf_kernels[] = {
        { "relu_kernel",  relu_func,      "FMNMX only — memory bound ceiling" },
        { "gelu_fast",    gelu_fast_func, "MUFU.EX2 + MUFU.RCP" },
        { "silu_kernel",  silu_func,      "MUFU.EX2 + MUFU.RCP" },
        { "gelu_kernel",  gelu_func,      "MUFU.TANH + FFMA" },
    };

    for (auto &k : perf_kernels) {
        float avg_ms = bench_kernel(k.func, dev_input, dev_output, num_elements,
                                    block_size, elements_per_thread,
                                    warmup_iters, bench_iters);
        double bw = gbytes / (avg_ms / 1000.0);
        printf("  %-20s  %7.3f ms  %7.1f GB/s  %s\n", k.name, avg_ms, bw, k.note);
    }

    printf("\nTo inspect the key SASS instructions:\n");
    printf("  cuobjdump -sass activations.sm_86.cubin | grep MUFU\n");
    printf("  → MUFU.TANH  (gelu_kernel)\n");
    printf("  → MUFU.EX2   (silu_kernel, gelu_fast)\n");
    printf("  → MUFU.RCP   (silu_kernel, gelu_fast)\n");

    // --- Cleanup ---
    cuMemFree(dev_input);
    cuMemFree(dev_output);
    cuModuleUnload(act_module);
    cuCtxDestroy(cu_context);
    free(host_input); free(host_output); free(host_ref);

    return 0;
}
