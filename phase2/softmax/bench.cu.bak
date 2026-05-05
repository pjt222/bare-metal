/*
 * bench.cu — Softmax benchmark: correctness + throughput
 *
 * Tests both softmax_warp (≤32 elements/row) and softmax_block (≤512 elements/row).
 *
 * Build:
 *   nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../common
 *
 * Usage:
 *   ./bench               # default: 65536 rows × 32 cols (warp kernel)
 *   ./bench 65536 128     # 128-element rows → block kernel
 *   ./bench 65536 512     # wide rows
 *
 * Expected SASS to see:
 *   cuobjdump -sass softmax.sm_86.cubin | grep -E 'SHFL|MUFU|FMAX'
 *   → SHFL.BFLY  (warp reduction)
 *   → MUFU.EX2   (fast exp2 hardware unit)
 *   → MUFU.RCP   (fast reciprocal hardware unit)
 *   → FMAX       (max comparison in reduction)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda.h>

#include "../common/bench.h"
#include "../common/check.h"

// -----------------------------------------------------------------------
// CPU reference softmax — numerically stable (max subtraction)
// -----------------------------------------------------------------------
static void cpu_softmax(
    const float *input,
    float       *output,
    int num_rows,
    int row_width
) {
    for (int row = 0; row < num_rows; row++) {
        const float *row_in  = input  + (size_t)row * row_width;
        float       *row_out = output + (size_t)row * row_width;

        // Find max
        float row_max = row_in[0];
        for (int col = 1; col < row_width; col++) {
            row_max = fmaxf(row_max, row_in[col]);
        }

        // Sum of exp(x - max)
        float exp_sum = 0.0f;
        for (int col = 0; col < row_width; col++) {
            row_out[col] = expf(row_in[col] - row_max);
            exp_sum += row_out[col];
        }

        // Normalize
        float rcp = 1.0f / exp_sum;
        for (int col = 0; col < row_width; col++) {
            row_out[col] *= rcp;
        }
    }
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------
int main(int argc, char **argv) {
    int num_rows  = (argc > 1) ? atoi(argv[1]) : 65536;
    int row_width = (argc > 2) ? atoi(argv[2]) : 32;

    printf("=== Softmax Benchmark — SHFL.BFLY + MUFU.EX2 + MUFU.RCP ===\n");
    printf("Input: %d rows × %d cols\n\n", num_rows, row_width);

    // Validate row_width constraints
    if (row_width > 32 && row_width > 4 * 128) {
        printf("WARNING: row_width=%d exceeds softmax_block capacity (%d).\n",
               row_width, 4 * 128);
        printf("Recompile with larger BLOCK_SIZE or ELEMENTS_PER_THREAD.\n\n");
    }

    CHECK_CU(cuInit(0));
    CUdevice cu_device;
    CHECK_CU(cuDeviceGet(&cu_device, 0));

    char device_name[256];
    CHECK_CU(cuDeviceGetName(device_name, sizeof(device_name), cu_device));
    printf("Device: %s\n\n", device_name);

    CUcontext cu_context;
    CHECK_CU(cuCtxCreate(&cu_context, 0, cu_device));

    // --- Load kernel cubin ---
    CUmodule   softmax_module;
    CUfunction warp_func, block_func;
    CUresult load_result = cuModuleLoad(&softmax_module, "softmax.sm_86.cubin");
    if (load_result != CUDA_SUCCESS) {
        const char *err_str = nullptr;
        cuGetErrorString(load_result, &err_str);
        fprintf(stderr, "Cannot load softmax.sm_86.cubin: %s\n", err_str);
        fprintf(stderr, "Build with: nvcc --cubin -arch=sm_86 -O2 -o softmax.sm_86.cubin softmax.cu\n");
        return EXIT_FAILURE;
    }
    CHECK_CU(cuModuleGetFunction(&warp_func,  softmax_module, "softmax_warp"));
    CHECK_CU(cuModuleGetFunction(&block_func, softmax_module, "softmax_block"));
    printf("Softmax kernels loaded.\n\n");

    // --- Host memory ---
    size_t total_elements = (size_t)num_rows * row_width;
    size_t total_bytes    = total_elements * sizeof(float);

    float *host_input  = (float *)malloc(total_bytes);
    float *host_output = (float *)malloc(total_bytes);
    float *host_ref    = (float *)malloc(total_bytes);

    fill_random(host_input, total_elements, 42);

    // CPU reference
    printf("Computing CPU reference...\n");
    cpu_softmax(host_input, host_ref, num_rows, row_width);
    printf("CPU reference done.\n\n");

    // --- Device memory ---
    CUdeviceptr dev_input, dev_output;
    CHECK_CU(cuMemAlloc(&dev_input,  total_bytes));
    CHECK_CU(cuMemAlloc(&dev_output, total_bytes));
    CHECK_CU(cuMemcpyHtoD(dev_input, host_input, total_bytes));

    // --- Correctness check ---
    printf("Correctness:\n");

    bool use_warp_kernel  = (row_width <= 32);
    bool use_block_kernel = (row_width <= 4 * 128);  // ELEMENTS_PER_THREAD * BLOCK_SIZE

    if (use_warp_kernel) {
        // softmax_warp: grid=(num_rows,1,1), block=(32,1,1)
        CHECK_CU(cuMemsetD32(dev_output, 0, total_elements));
        void *args[] = { &dev_input, &dev_output, &num_rows, &row_width };
        CHECK_CU(cuLaunchKernel(warp_func,
            num_rows, 1, 1,   // grid
            32, 1, 1,          // block: one warp
            0, NULL, args, NULL));
        CHECK_CU(cuCtxSynchronize());
        CHECK_CU(cuMemcpyDtoH(host_output, dev_output, total_bytes));

        auto result = check_fp32(host_output, host_ref, total_elements, 1e-4f, 1e-4f);
        print_check_result("softmax_warp", result);
    }

    if (use_block_kernel) {
        // softmax_block: grid=(num_rows,1,1), block=(BLOCK_SIZE=128,1,1)
        int block_size = 128;
        CHECK_CU(cuMemsetD32(dev_output, 0, total_elements));
        void *args[] = { &dev_input, &dev_output, &num_rows, &row_width };
        CHECK_CU(cuLaunchKernel(block_func,
            num_rows, 1, 1,
            block_size, 1, 1,
            0, NULL, args, NULL));
        CHECK_CU(cuCtxSynchronize());
        CHECK_CU(cuMemcpyDtoH(host_output, dev_output, total_bytes));

        auto result = check_fp32(host_output, host_ref, total_elements, 1e-4f, 1e-4f);
        print_check_result("softmax_block", result);
    }

    // --- Performance benchmark ---
    int warmup_iters = 5;
    int bench_iters  = 100;

    printf("\nPerformance (avg of %d runs, %d warmup):\n", bench_iters, warmup_iters);

    // FLOPs for softmax: 3 ops per element (exp, add for sum, mul for normalize)
    // Plus reduction overhead — approximate as 5 FLOPS/element
    double flops_total = 5.0 * total_elements;
    double gbytes_total = 2.0 * total_bytes / 1e9;  // read + write

    if (use_warp_kernel) {
        void *args[] = { &dev_input, &dev_output, &num_rows, &row_width };
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

        double bandwidth_gb_s = gbytes_total / (avg_ms / 1000.0);
        printf("  %-30s %7.3f ms   %8.2f GB/s\n", "softmax_warp", avg_ms, bandwidth_gb_s);
    }

    if (use_block_kernel) {
        int block_size = 128;
        void *args[] = { &dev_input, &dev_output, &num_rows, &row_width };
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

        double bandwidth_gb_s = gbytes_total / (avg_ms / 1000.0);
        printf("  %-30s %7.3f ms   %8.2f GB/s\n", "softmax_block", avg_ms, bandwidth_gb_s);
    }

    printf("\nTo inspect the key SASS instructions:\n");
    printf("  cuobjdump -sass softmax.sm_86.cubin | grep -E 'SHFL|MUFU|FMAX'\n");
    printf("  → SHFL.BFLY  (warp butterfly shuffle reduction)\n");
    printf("  → MUFU.EX2   (fast 2^x — maps from exp2f(x * log2e))\n");
    printf("  → MUFU.RCP   (fast reciprocal — maps from __frcp_rn)\n");

    // --- Cleanup ---
    cuMemFree(dev_input);
    cuMemFree(dev_output);
    cuModuleUnload(softmax_module);
    cuCtxDestroy(cu_context);
    free(host_input); free(host_output); free(host_ref);

    return 0;
}
