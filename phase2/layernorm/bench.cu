/*
 * bench.cu — LayerNorm benchmark: correctness + throughput
 *
 * Build:
 *   nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../common
 *
 * Usage:
 *   ./bench               # default: 65536 rows × 32 cols
 *   ./bench 65536 128     # typical transformer hidden dim / num_heads
 *   ./bench 16384 512     # wider rows
 *
 * Expected SASS:
 *   cuobjdump -sass layernorm.sm_86.cubin | grep -E 'SHFL|MUFU'
 *   → SHFL.BFLY  (3 rounds of Welford mean/M2 reduction)
 *   → MUFU.RSQ   (reciprocal sqrt of variance)
 *   → MUFU.RCP   (used internally by rsqrtf approximation)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda.h>

#include "../common/bench.h"
#include "../common/check.h"

// -----------------------------------------------------------------------
// CPU reference LayerNorm
// -----------------------------------------------------------------------
static void cpu_layernorm(
    const float *input,
    const float *gamma,
    const float *beta,
    float       *output,
    int num_rows,
    int row_width,
    float epsilon
) {
    for (int row = 0; row < num_rows; row++) {
        const float *row_in  = input  + (size_t)row * row_width;
        float       *row_out = output + (size_t)row * row_width;

        // Mean
        double sum = 0.0;
        for (int col = 0; col < row_width; col++) sum += row_in[col];
        float row_mean = (float)(sum / row_width);

        // Variance
        double sum_sq = 0.0;
        for (int col = 0; col < row_width; col++) {
            float deviation = row_in[col] - row_mean;
            sum_sq += (double)(deviation * deviation);
        }
        float row_variance = (float)(sum_sq / row_width);

        // Normalize
        float rsqrt_var = 1.0f / sqrtf(row_variance + epsilon);
        for (int col = 0; col < row_width; col++) {
            float normalized = (row_in[col] - row_mean) * rsqrt_var;
            row_out[col] = gamma[col] * normalized + beta[col];
        }
    }
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------
int main(int argc, char **argv) {
    int num_rows  = (argc > 1) ? atoi(argv[1]) : 65536;
    int row_width = (argc > 2) ? atoi(argv[2]) : 32;
    float epsilon = 1e-5f;

    printf("=== LayerNorm Benchmark — Welford + MUFU.RSQ ===\n");
    printf("Input: %d rows × %d cols  epsilon=%.0e\n\n", num_rows, row_width, (double)epsilon);

    CHECK_CU(cuInit(0));
    CUdevice cu_device;
    CHECK_CU(cuDeviceGet(&cu_device, 0));

    char device_name[256];
    CHECK_CU(cuDeviceGetName(device_name, sizeof(device_name), cu_device));
    printf("Device: %s\n\n", device_name);

    CUcontext cu_context;
    CHECK_CU(cuCtxCreate(&cu_context, 0, cu_device));

    // --- Load kernel ---
    CUmodule   layernorm_module;
    CUfunction warp_func, block_func;
    CUresult load_result = cuModuleLoad(&layernorm_module, "layernorm.sm_86.cubin");
    if (load_result != CUDA_SUCCESS) {
        const char *err_str = nullptr;
        cuGetErrorString(load_result, &err_str);
        fprintf(stderr, "Cannot load layernorm.sm_86.cubin: %s\n", err_str);
        fprintf(stderr, "Build with: nvcc --cubin -arch=sm_86 -O2 -o layernorm.sm_86.cubin layernorm.cu\n");
        return EXIT_FAILURE;
    }
    CHECK_CU(cuModuleGetFunction(&warp_func,  layernorm_module, "layernorm_warp"));
    CHECK_CU(cuModuleGetFunction(&block_func, layernorm_module, "layernorm_block"));
    printf("LayerNorm kernels loaded.\n\n");

    // --- Host memory ---
    size_t total_elements = (size_t)num_rows * row_width;
    size_t total_bytes    = total_elements * sizeof(float);
    size_t param_bytes    = row_width * sizeof(float);

    float *host_input  = (float *)malloc(total_bytes);
    float *host_output = (float *)malloc(total_bytes);
    float *host_ref    = (float *)malloc(total_bytes);
    float *host_gamma  = (float *)malloc(param_bytes);
    float *host_beta   = (float *)malloc(param_bytes);

    fill_random(host_input, total_elements, 42);

    // Initialize gamma=1, beta=0 (standard initial values)
    for (int col = 0; col < row_width; col++) {
        host_gamma[col] = 1.0f;
        host_beta[col]  = 0.0f;
    }

    // CPU reference
    printf("Computing CPU reference...\n");
    cpu_layernorm(host_input, host_gamma, host_beta, host_ref, num_rows, row_width, epsilon);
    printf("CPU reference done.\n\n");

    // --- Device memory ---
    CUdeviceptr dev_input, dev_output, dev_gamma, dev_beta;
    CHECK_CU(cuMemAlloc(&dev_input,  total_bytes));
    CHECK_CU(cuMemAlloc(&dev_output, total_bytes));
    CHECK_CU(cuMemAlloc(&dev_gamma,  param_bytes));
    CHECK_CU(cuMemAlloc(&dev_beta,   param_bytes));
    CHECK_CU(cuMemcpyHtoD(dev_input, host_input,  total_bytes));
    CHECK_CU(cuMemcpyHtoD(dev_gamma, host_gamma,  param_bytes));
    CHECK_CU(cuMemcpyHtoD(dev_beta,  host_beta,   param_bytes));

    // --- Correctness ---
    printf("Correctness:\n");

    bool use_warp_kernel  = (row_width <= 32);
    bool use_block_kernel = (row_width <= 4 * 128);

    if (use_warp_kernel) {
        CHECK_CU(cuMemsetD32(dev_output, 0, total_elements));
        void *args[] = { &dev_input, &dev_gamma, &dev_beta, &dev_output,
                         &num_rows, &row_width, &epsilon };
        CHECK_CU(cuLaunchKernel(warp_func,
            num_rows, 1, 1,
            32, 1, 1,
            0, NULL, args, NULL));
        CHECK_CU(cuCtxSynchronize());
        CHECK_CU(cuMemcpyDtoH(host_output, dev_output, total_bytes));
        auto result = check_fp32(host_output, host_ref, total_elements, 1e-4f, 1e-3f);
        print_check_result("layernorm_warp", result);
    }

    if (use_block_kernel) {
        int block_size = 128;
        CHECK_CU(cuMemsetD32(dev_output, 0, total_elements));
        void *args[] = { &dev_input, &dev_gamma, &dev_beta, &dev_output,
                         &num_rows, &row_width, &epsilon };
        CHECK_CU(cuLaunchKernel(block_func,
            num_rows, 1, 1,
            block_size, 1, 1,
            0, NULL, args, NULL));
        CHECK_CU(cuCtxSynchronize());
        CHECK_CU(cuMemcpyDtoH(host_output, dev_output, total_bytes));
        auto result = check_fp32(host_output, host_ref, total_elements, 1e-4f, 1e-3f);
        print_check_result("layernorm_block", result);
    }

    // --- Performance ---
    int warmup_iters = 5;
    int bench_iters  = 100;

    printf("\nPerformance (avg of %d runs, %d warmup):\n", bench_iters, warmup_iters);

    // Memory bandwidth: read input (once), read gamma, read beta, write output
    // Params (gamma, beta) are small and likely cached
    double gbytes = (total_bytes + param_bytes * 2 + total_bytes) / 1e9;

    if (use_warp_kernel) {
        void *args[] = { &dev_input, &dev_gamma, &dev_beta, &dev_output,
                         &num_rows, &row_width, &epsilon };
        for (int i = 0; i < warmup_iters; i++) {
            CHECK_CU(cuLaunchKernel(warp_func, num_rows, 1, 1, 32, 1, 1, 0, NULL, args, NULL));
        }
        CHECK_CU(cuCtxSynchronize());

        float avg_ms;
        {
            BenchTimer timer;
            timer.start();
            for (int i = 0; i < bench_iters; i++) {
                CHECK_CU(cuLaunchKernel(warp_func, num_rows, 1, 1, 32, 1, 1, 0, NULL, args, NULL));
            }
            avg_ms = timer.stop_ms() / bench_iters;
        }
        printf("  %-30s %7.3f ms   %8.2f GB/s\n", "layernorm_warp", avg_ms,
               gbytes / (avg_ms / 1000.0));
    }

    if (use_block_kernel) {
        int block_size = 128;
        void *args[] = { &dev_input, &dev_gamma, &dev_beta, &dev_output,
                         &num_rows, &row_width, &epsilon };
        for (int i = 0; i < warmup_iters; i++) {
            CHECK_CU(cuLaunchKernel(block_func, num_rows, 1, 1, block_size, 1, 1, 0, NULL, args, NULL));
        }
        CHECK_CU(cuCtxSynchronize());

        float avg_ms;
        {
            BenchTimer timer;
            timer.start();
            for (int i = 0; i < bench_iters; i++) {
                CHECK_CU(cuLaunchKernel(block_func, num_rows, 1, 1, block_size, 1, 1, 0, NULL, args, NULL));
            }
            avg_ms = timer.stop_ms() / bench_iters;
        }
        printf("  %-30s %7.3f ms   %8.2f GB/s\n", "layernorm_block", avg_ms,
               gbytes / (avg_ms / 1000.0));
    }

    printf("\nTo inspect the key SASS instructions:\n");
    printf("  cuobjdump -sass layernorm.sm_86.cubin | grep -E 'SHFL|MUFU'\n");
    printf("  → SHFL.BFLY  (Welford mean/M2 reductions)\n");
    printf("  → MUFU.RSQ   (reciprocal sqrt of variance)\n");

    // --- Cleanup ---
    cuMemFree(dev_input); cuMemFree(dev_output);
    cuMemFree(dev_gamma); cuMemFree(dev_beta);
    cuModuleUnload(layernorm_module);
    cuCtxDestroy(cu_context);
    free(host_input); free(host_output); free(host_ref);
    free(host_gamma); free(host_beta);

    return 0;
}
