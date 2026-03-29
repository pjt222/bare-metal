/*
 * bench.cu — IGEMM benchmark: INT8 Tensor Core vs FP16 HGEMM vs theoretical peak
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o igemm.sm_86.cubin igemm.cu
 *   nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../common
 *
 * Usage:
 *   ./bench [M] [N] [K]
 *   ./bench 4096 4096 4096   # default — large enough to saturate Tensor Cores
 *
 * Expected: ~4× improvement over FP16 HGEMM in throughput (TOPS).
 * Theoretical: 696 TOPS INT8 / 174 TFLOPS FP16 = 4×
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda.h>

#include "../common/bench.h"
#include "../common/check.h"

// -----------------------------------------------------------------------
// Symmetric per-tensor quantization: FP32 → INT8
// -----------------------------------------------------------------------
static float compute_scale(const float *data, int num_elements) {
    float max_abs = 0.0f;
    for (int i = 0; i < num_elements; i++) {
        float abs_val = fabsf(data[i]);
        if (abs_val > max_abs) max_abs = abs_val;
    }
    if (max_abs == 0.0f) return 1.0f;  // avoid division by zero
    return max_abs / 127.0f;
}

static void quantize_symmetric(
    const float *fp32_data,
    signed char *int8_data,
    int num_elements,
    float scale
) {
    float inv_scale = 1.0f / scale;
    for (int i = 0; i < num_elements; i++) {
        int quantized = (int)roundf(fp32_data[i] * inv_scale);
        if (quantized >  127) quantized =  127;
        if (quantized < -128) quantized = -128;
        int8_data[i] = (signed char)quantized;
    }
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------
int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 4096;
    int N = (argc > 2) ? atoi(argv[2]) : 4096;
    int K = (argc > 3) ? atoi(argv[3]) : 4096;

    // Round to nearest multiple of 16 (WMMA requirement)
    M = (M + 15) / 16 * 16;
    N = (N + 15) / 16 * 16;
    K = (K + 15) / 16 * 16;

    printf("=== IGEMM Benchmark — INT8 Tensor Cores (sm_86) ===\n");
    printf("Matrix: C[%d×%d] = A[%d×%d] * B[%d×%d]  (INT8 in, INT32 accum, FP32 out)\n\n",
           M, N, M, K, K, N);

    CHECK_CU(cuInit(0));
    CUdevice cu_device;
    CHECK_CU(cuDeviceGet(&cu_device, 0));

    char device_name[256];
    CHECK_CU(cuDeviceGetName(device_name, sizeof(device_name), cu_device));
    printf("Device: %s\n\n", device_name);

    CUcontext cu_context;
    CHECK_CU(cuCtxCreate(&cu_context, 0, cu_device));

    // --- Load IGEMM kernel ---
    CUmodule   igemm_module;
    CUfunction igemm_func;
    CUresult load_result = cuModuleLoad(&igemm_module, "igemm.sm_86.cubin");
    if (load_result != CUDA_SUCCESS) {
        const char *err = nullptr;
        cuGetErrorString(load_result, &err);
        fprintf(stderr, "Cannot load igemm.sm_86.cubin: %s\n", err);
        fprintf(stderr, "Build with: nvcc --cubin -arch=sm_86 -O2 -o igemm.sm_86.cubin igemm.cu\n");
        return EXIT_FAILURE;
    }
    CHECK_CU(cuModuleGetFunction(&igemm_func, igemm_module, "igemm_wmma"));

    // --- Load tiled IGEMM kernel ---
    CUmodule   tiled_module = NULL;
    CUfunction tiled_func   = NULL;
    bool have_tiled = (cuModuleLoad(&tiled_module, "igemm_tiled.sm_86.cubin") == CUDA_SUCCESS);
    if (have_tiled) {
        CHECK_CU(cuModuleGetFunction(&tiled_func, tiled_module, "igemm_tiled"));
    }
    printf("IGEMM kernels loaded: naive%s.\n\n", have_tiled ? " + tiled" : " (tiled cubin not found)");

    // --- Allocate host memory ---
    size_t a_elems = (size_t)M * K;
    size_t b_elems = (size_t)K * N;
    size_t c_elems = (size_t)M * N;

    float       *host_a_fp32 = (float *)      malloc(a_elems * sizeof(float));
    float       *host_b_fp32 = (float *)      malloc(b_elems * sizeof(float));
    float       *host_c_fp32 = (float *)      malloc(c_elems * sizeof(float));
    float       *host_ref    = (float *)      malloc(c_elems * sizeof(float));
    signed char *host_a_int8 = (signed char *)malloc(a_elems);
    signed char *host_b_int8 = (signed char *)malloc(b_elems);

    fill_random(host_a_fp32, a_elems, 42);
    fill_random(host_b_fp32, b_elems, 99);
    fill_zeros(host_ref, c_elems);

    // --- Quantize ---
    float scale_a = compute_scale(host_a_fp32, a_elems);
    float scale_b = compute_scale(host_b_fp32, b_elems);
    quantize_symmetric(host_a_fp32, host_a_int8, a_elems, scale_a);
    quantize_symmetric(host_b_fp32, host_b_int8, b_elems, scale_b);
    printf("Quantization: scale_a=%.6f  scale_b=%.6f\n", scale_a, scale_b);

    // CPU FP32 reference (only for small matrices)
    bool run_cpu_ref = (M <= 512);
    if (run_cpu_ref) {
        printf("Computing CPU reference...\n");
        cpu_sgemm(M, N, K, 1.0f, host_a_fp32, K, host_b_fp32, N, 0.0f, host_ref, N);
    } else {
        printf("CPU reference skipped (matrix too large — use M<=512 for correctness check)\n");
    }

    // --- Allocate device memory ---
    CUdeviceptr dev_a_int8, dev_b_int8, dev_c;
    CHECK_CU(cuMemAlloc(&dev_a_int8, a_elems));           // 1 byte per int8
    CHECK_CU(cuMemAlloc(&dev_b_int8, b_elems));
    CHECK_CU(cuMemAlloc(&dev_c,      c_elems * sizeof(float)));

    CHECK_CU(cuMemcpyHtoD(dev_a_int8, host_a_int8, a_elems));
    CHECK_CU(cuMemcpyHtoD(dev_b_int8, host_b_int8, b_elems));

    // --- Launch configs ---
    int grid_naive_x = (N + 31) / 32;  // naive: 32×32 per block
    int grid_naive_y = (M + 31) / 32;
    int grid_tiled_x = (N + 63) / 64;  // tiled: 64×64 per block
    int grid_tiled_y = (M + 63) / 64;

    // --- Correctness check ---
    if (run_cpu_ref) {
        printf("\nCorrectness (INT8 input, INT32 accum, FP32 dequantized output):\n");

        // Naive
        CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
        void *args[] = { &dev_a_int8, &dev_b_int8, &dev_c, &M, &N, &K, &scale_a, &scale_b };
        CHECK_CU(cuLaunchKernel(igemm_func,
            grid_naive_x, grid_naive_y, 1,   64, 2, 1,
            0, NULL, args, NULL));
        CHECK_CU(cuCtxSynchronize());
        CHECK_CU(cuMemcpyDtoH(host_c_fp32, dev_c, c_elems * sizeof(float)));
        auto r1 = check_fp32(host_c_fp32, host_ref, c_elems, 0.5f, 0.1f);
        print_check_result("igemm_wmma  (naive)", r1);

        // Tiled
        if (have_tiled) {
            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            void *args_t[] = { &dev_a_int8, &dev_b_int8, &dev_c, &M, &N, &K, &scale_a, &scale_b };
            CHECK_CU(cuLaunchKernel(tiled_func,
                grid_tiled_x, grid_tiled_y, 1,   64, 2, 1,
                0, NULL, args_t, NULL));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(host_c_fp32, dev_c, c_elems * sizeof(float)));
            auto r2 = check_fp32(host_c_fp32, host_ref, c_elems, 0.5f, 0.1f);
            print_check_result("igemm_tiled (smem)", r2);
        }
        printf("  Note: quantization error expected — symmetric per-tensor, scale=max_abs/127\n");
    }

    // --- Performance benchmark ---
    int warmup_iters = 5;
    int bench_iters  = 50;
    printf("\nPerformance (avg of %d runs, %d warmup):\n", bench_iters, warmup_iters);

    double int8_peak_tops  = 696000.0;  // RTX 3070 Ti INT8 Tensor Core peak
    double fp16_peak_gflops = 174000.0;

    void *bench_args[] = { &dev_a_int8, &dev_b_int8, &dev_c, &M, &N, &K, &scale_a, &scale_b };

    // Helper to benchmark a kernel
    auto run_bench = [&](CUfunction fn, int gx, int gy, const char *label) {
        for (int i = 0; i < warmup_iters; i++) {
            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            CHECK_CU(cuLaunchKernel(fn, gx, gy, 1, 64, 2, 1, 0, NULL, bench_args, NULL));
        }
        CHECK_CU(cuCtxSynchronize());

        float ms;
        {
            BenchTimer timer;
            timer.start();
            for (int i = 0; i < bench_iters; i++) {
                CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
                CHECK_CU(cuLaunchKernel(fn, gx, gy, 1, 64, 2, 1, 0, NULL, bench_args, NULL));
            }
            ms = timer.stop_ms() / bench_iters;
        }
        double t = compute_gflops_gemm(M, N, K, ms);
        printf("  %-35s %7.3f ms   %8.2f TOPS\n", label, ms, t);
        return t;
    };

    double naive_tops = run_bench(igemm_func, grid_naive_x, grid_naive_y, "igemm_wmma  (naive)");
    double tiled_tops = 0.0;
    if (have_tiled) {
        tiled_tops = run_bench(tiled_func, grid_tiled_x, grid_tiled_y, "igemm_tiled (smem)");
    }

    printf("  %-35s %7s      %8.0f TOPS  (theoretical)\n", "INT8 Tensor Core peak", "--", int8_peak_tops);
    printf("  %-35s %7s      %8.0f GFLOPS  (theoretical)\n", "FP16 Tensor Core peak", "--", fp16_peak_gflops);
    printf("\n");
    double best_tops = have_tiled ? fmax(naive_tops, tiled_tops) : naive_tops;
    printf("  Best: %.1f%% of INT8 peak, %.1f× vs FP16 peak\n",
           100.0 * best_tops / int8_peak_tops, best_tops / fp16_peak_gflops);
    if (have_tiled && tiled_tops > 0)
        printf("  Tiled vs naive: %.2f×\n", tiled_tops / naive_tops);

    printf("\nSASS inspection:\n");
    printf("  cuobjdump -sass igemm.sm_86.cubin | grep IMMA         # naive\n");
    printf("  cuobjdump -sass igemm_tiled.sm_86.cubin | grep IMMA   # tiled\n");

    // --- Cleanup ---
    cuMemFree(dev_a_int8);
    cuMemFree(dev_b_int8);
    cuMemFree(dev_c);
    cuModuleUnload(igemm_module);
    if (tiled_module) cuModuleUnload(tiled_module);
    cuCtxDestroy(cu_context);
    free(host_a_fp32); free(host_b_fp32); free(host_c_fp32);
    free(host_ref); free(host_a_int8); free(host_b_int8);

    return 0;
}
