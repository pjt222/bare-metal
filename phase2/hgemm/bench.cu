/*
 * bench.cu — HGEMM benchmark: naive vs tiled Tensor Core vs theoretical peak
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o hgemm.sm_86.cubin hgemm.cu
 *   nvcc --cubin -arch=sm_86 -O2 -o hgemm_tiled.sm_86.cubin hgemm_tiled.cu
 *   nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../common
 *
 * Usage:
 *   ./bench [M] [N] [K]
 *   ./bench 4096 4096 4096   # default — large enough to saturate Tensor Cores
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda.h>

#include "../common/bench.h"
#include "../common/check.h"

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

    printf("=== HGEMM Benchmark — Tensor Cores (sm_86) ===\n");
    printf("Matrix: C[%d×%d] = A[%d×%d] * B[%d×%d]  (FP16 in, FP32 out)\n\n", M, N, M, K, K, N);

    CHECK_CU(cuInit(0));
    CUdevice cu_device;
    CHECK_CU(cuDeviceGet(&cu_device, 0));

    char device_name[256];
    CHECK_CU(cuDeviceGetName(device_name, sizeof(device_name), cu_device));
    printf("Device: %s\n\n", device_name);

    CUcontext cu_context;
    CHECK_CU(cuCtxCreate(&cu_context, 0, cu_device));

    // --- Load naive HGEMM kernel ---
    CUmodule   naive_module = NULL;
    CUfunction naive_func   = NULL;
    bool have_naive = (cuModuleLoad(&naive_module, "hgemm.sm_86.cubin") == CUDA_SUCCESS);
    if (have_naive) {
        CHECK_CU(cuModuleGetFunction(&naive_func, naive_module, "hgemm_wmma"));
    }

    // --- Load tiled HGEMM kernel ---
    CUmodule   tiled_module = NULL;
    CUfunction tiled_func   = NULL;
    bool have_tiled = (cuModuleLoad(&tiled_module, "hgemm_tiled.sm_86.cubin") == CUDA_SUCCESS);
    if (have_tiled) {
        CHECK_CU(cuModuleGetFunction(&tiled_func, tiled_module, "hgemm_tiled"));
    }

    // --- Load direct-store HGEMM kernel ---
    CUmodule   direct_module = NULL;
    CUfunction direct_func   = NULL;
    bool have_direct = (cuModuleLoad(&direct_module, "hgemm_tiled_direct.sm_86.cubin") == CUDA_SUCCESS);
    if (have_direct) {
        CHECK_CU(cuModuleGetFunction(&direct_func, direct_module, "hgemm_tiled_direct"));
    }

    // --- Load 16-warp HGEMM kernel ---
    CUmodule   w16_module = NULL;
    CUfunction w16_func   = NULL;
    bool have_w16 = (cuModuleLoad(&w16_module, "hgemm_16warp.sm_86.cubin") == CUDA_SUCCESS);
    if (have_w16) {
        CHECK_CU(cuModuleGetFunction(&w16_func, w16_module, "hgemm_16warp"));
    }

    // --- Load 16-warp + smem epilogue HGEMM kernel ---
    CUmodule   w16e_module = NULL;
    CUfunction w16e_func   = NULL;
    bool have_w16e = (cuModuleLoad(&w16e_module, "hgemm_16warp_epi.sm_86.cubin") == CUDA_SUCCESS);
    if (have_w16e) {
        CHECK_CU(cuModuleGetFunction(&w16e_func, w16e_module, "hgemm_16warp_epi"));
    }

    if (!have_naive && !have_tiled && !have_direct && !have_w16) {
        fprintf(stderr, "No kernels found. Build with:\n");
        fprintf(stderr, "  nvcc --cubin -arch=sm_86 -O2 -o hgemm.sm_86.cubin hgemm.cu\n");
        fprintf(stderr, "  nvcc --cubin -arch=sm_86 -O2 -o hgemm_tiled.sm_86.cubin hgemm_tiled.cu\n");
        return EXIT_FAILURE;
    }

    printf("Kernels loaded:%s%s%s%s%s\n\n",
           have_naive ? " naive" : "",
           have_tiled ? " + tiled(128x128)" : "",
           have_direct ? " + direct-store" : "",
           have_w16 ? " + 16warp" : "",
           have_w16e ? " + 16warp+epi" : "");

    // --- Allocate host memory ---
    size_t a_elems = (size_t)M * K;
    size_t b_elems = (size_t)K * N;
    size_t c_elems = (size_t)M * N;

    size_t a_bytes_fp16 = a_elems * sizeof(unsigned short);
    size_t b_bytes_fp16 = b_elems * sizeof(unsigned short);
    size_t c_bytes_fp32 = c_elems * sizeof(float);

    float          *host_a_fp32 = (float *)malloc(a_elems * sizeof(float));
    float          *host_b_fp32 = (float *)malloc(b_elems * sizeof(float));
    float          *host_c_fp32 = (float *)malloc(c_bytes_fp32);
    float          *host_ref    = (float *)malloc(c_bytes_fp32);
    unsigned short *host_a_fp16 = (unsigned short *)malloc(a_bytes_fp16);
    unsigned short *host_b_fp16 = (unsigned short *)malloc(b_bytes_fp16);

    fill_random(host_a_fp32, a_elems, 42);
    fill_random(host_b_fp32, b_elems, 99);
    fill_zeros(host_ref, c_elems);

    // Convert FP32 → FP16 on host
    auto fp32_to_fp16_host = [](float f) -> unsigned short {
        unsigned int bits;
        memcpy(&bits, &f, 4);
        unsigned short sign = (bits >> 31) & 0x1;
        int exp = ((bits >> 23) & 0xFF) - 127 + 15;
        unsigned int mant = (bits >> 13) & 0x3FF;
        if (exp <= 0) return sign << 15;
        if (exp >= 31) exp = 31;
        return (unsigned short)((sign << 15) | (exp << 10) | mant);
    };

    for (size_t i = 0; i < a_elems; i++) host_a_fp16[i] = fp32_to_fp16_host(host_a_fp32[i]);
    for (size_t i = 0; i < b_elems; i++) host_b_fp16[i] = fp32_to_fp16_host(host_b_fp32[i]);

    // CPU FP32 reference (only for small matrices — skip for large)
    bool run_cpu_ref = (M <= 512);
    if (run_cpu_ref) {
        printf("Computing CPU reference...\n");
        cpu_sgemm(M, N, K, 1.0f, host_a_fp32, K, host_b_fp32, N, 0.0f, host_ref, N);
    } else {
        printf("CPU reference skipped (matrix too large — use M<=512 for correctness check)\n");
    }

    // --- Allocate device memory ---
    CUdeviceptr dev_a_fp16, dev_b_fp16, dev_c;
    CHECK_CU(cuMemAlloc(&dev_a_fp16, a_bytes_fp16));
    CHECK_CU(cuMemAlloc(&dev_b_fp16, b_bytes_fp16));
    CHECK_CU(cuMemAlloc(&dev_c,      c_bytes_fp32));

    CHECK_CU(cuMemcpyHtoD(dev_a_fp16, host_a_fp16, a_bytes_fp16));
    CHECK_CU(cuMemcpyHtoD(dev_b_fp16, host_b_fp16, b_bytes_fp16));

    // --- Launch configs ---
    int grid_naive_x = (N + 31) / 32;    // naive: 32×32 per block
    int grid_naive_y = (M + 31) / 32;
    int grid_tiled_x = (N + 127) / 128;  // tiled: 128×128 per block
    int grid_tiled_y = (M + 127) / 128;

    // --- Correctness check ---
    if (run_cpu_ref) {
        printf("\nCorrectness (FP16 input, FP32 accumulation):\n");

        if (have_naive) {
            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            void *args[] = { &dev_a_fp16, &dev_b_fp16, &dev_c, &M, &N, &K };
            CHECK_CU(cuLaunchKernel(naive_func,
                grid_naive_x, grid_naive_y, 1,   64, 2, 1,
                0, NULL, args, NULL));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(host_c_fp32, dev_c, c_bytes_fp32));
            auto r1 = check_fp32(host_c_fp32, host_ref, c_elems, 1e-1f, 1e-1f);
            print_check_result("hgemm_wmma (naive)", r1);
        }

        if (have_tiled) {
            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            void *args[] = { &dev_a_fp16, &dev_b_fp16, &dev_c, &M, &N, &K };
            CHECK_CU(cuLaunchKernel(tiled_func,
                grid_tiled_x, grid_tiled_y, 1,   256, 1, 1,
                0, NULL, args, NULL));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(host_c_fp32, dev_c, c_bytes_fp32));
            auto r2 = check_fp32(host_c_fp32, host_ref, c_elems, 1e-1f, 1e-1f);
            print_check_result("hgemm_tiled (128x128)", r2);
        }

        if (have_direct) {
            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            void *args[] = { &dev_a_fp16, &dev_b_fp16, &dev_c, &M, &N, &K };
            CHECK_CU(cuLaunchKernel(direct_func,
                grid_tiled_x, grid_tiled_y, 1,   256, 1, 1,
                0, NULL, args, NULL));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(host_c_fp32, dev_c, c_bytes_fp32));
            auto r3 = check_fp32(host_c_fp32, host_ref, c_elems, 1e-1f, 1e-1f);
            print_check_result("hgemm_direct (no smem epi)", r3);
        }

        if (have_w16) {
            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            void *args[] = { &dev_a_fp16, &dev_b_fp16, &dev_c, &M, &N, &K };
            CHECK_CU(cuLaunchKernel(w16_func,
                grid_tiled_x, grid_tiled_y, 1,   512, 1, 1,
                0, NULL, args, NULL));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(host_c_fp32, dev_c, c_bytes_fp32));
            auto r4 = check_fp32(host_c_fp32, host_ref, c_elems, 1e-1f, 1e-1f);
            print_check_result("hgemm_16warp (2 blk/SM)", r4);
        }

        if (have_w16e) {
            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            void *args[] = { &dev_a_fp16, &dev_b_fp16, &dev_c, &M, &N, &K };
            CHECK_CU(cuLaunchKernel(w16e_func,
                grid_tiled_x, grid_tiled_y, 1,   512, 1, 1,
                0, NULL, args, NULL));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(host_c_fp32, dev_c, c_bytes_fp32));
            auto r5 = check_fp32(host_c_fp32, host_ref, c_elems, 1e-1f, 1e-1f);
            print_check_result("hgemm_16warp_epi (2blk+epi)", r5);
        }

        printf("  Note: FP16 input has limited precision — tolerance 0.1\n");
    }

    // --- Performance benchmark ---
    int warmup_iters = 5;
    int bench_iters  = 50;
    printf("\nPerformance (avg of %d runs, %d warmup):\n", bench_iters, warmup_iters);

    double fp16_peak_gflops = 174000.0;  // RTX 3070 Ti FP16 Tensor Core peak
    double fp32_peak_gflops =  21700.0;  // RTX 3070 Ti FP32 peak

    void *args[] = { &dev_a_fp16, &dev_b_fp16, &dev_c, &M, &N, &K };

    // Helper to benchmark a kernel
    auto run_bench = [&](CUfunction fn, int gx, int gy, int bx, int by,
                         const char *label) -> double {
        for (int i = 0; i < warmup_iters; i++) {
            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            CHECK_CU(cuLaunchKernel(fn, gx, gy, 1, bx, by, 1, 0, NULL, args, NULL));
        }
        CHECK_CU(cuCtxSynchronize());

        float ms;
        {
            BenchTimer timer;
            timer.start();
            for (int i = 0; i < bench_iters; i++) {
                CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
                CHECK_CU(cuLaunchKernel(fn, gx, gy, 1, bx, by, 1, 0, NULL, args, NULL));
            }
            ms = timer.stop_ms() / bench_iters;
        }
        double gflops = compute_gflops_gemm(M, N, K, ms);
        printf("  %-35s %7.3f ms   %8.2f GFLOPS\n", label, ms, gflops);
        return gflops;
    };

    double naive_gflops = 0.0;
    if (have_naive)
        naive_gflops = run_bench(naive_func, grid_naive_x, grid_naive_y, 64, 2,
                                  "hgemm_wmma (naive 32x32)");

    double tiled_gflops = 0.0;
    if (have_tiled)
        tiled_gflops = run_bench(tiled_func, grid_tiled_x, grid_tiled_y, 256, 1,
                                  "hgemm_tiled (128x128 smem epi)");

    double direct_gflops = 0.0;
    if (have_direct)
        direct_gflops = run_bench(direct_func, grid_tiled_x, grid_tiled_y, 256, 1,
                                   "hgemm_direct (128x128 no epi)");

    double w16_gflops = 0.0;
    if (have_w16)
        w16_gflops = run_bench(w16_func, grid_tiled_x, grid_tiled_y, 512, 1,
                                "hgemm_16warp (128x128 2blk/SM)");

    double w16e_gflops = 0.0;
    if (have_w16e)
        w16e_gflops = run_bench(w16e_func, grid_tiled_x, grid_tiled_y, 512, 1,
                                 "hgemm_16warp_epi (2blk+epi)");

    printf("  %-35s %7s      %8.0f GFLOPS  (theoretical)\n", "FP32 peak (FFMA)", "--", fp32_peak_gflops);
    printf("  %-35s %7s      %8.0f GFLOPS  (theoretical)\n", "FP16 Tensor Core peak", "--", fp16_peak_gflops);
    printf("\n");

    double best_gflops = fmax(naive_gflops, fmax(tiled_gflops, fmax(direct_gflops,
                          fmax(w16_gflops, w16e_gflops))));
    printf("  Best: %.1f%% of FP16 peak\n", 100.0 * best_gflops / fp16_peak_gflops);

    if (tiled_gflops > 0 && naive_gflops > 0)
        printf("  Tiled vs naive: %.2f× (%+.1f%%)\n",
               tiled_gflops / naive_gflops,
               100.0 * (tiled_gflops - naive_gflops) / naive_gflops);
    if (direct_gflops > 0 && tiled_gflops > 0)
        printf("  Direct vs smem-epi: %.4f× (%+.2f%%)\n",
               direct_gflops / tiled_gflops,
               100.0 * (direct_gflops - tiled_gflops) / tiled_gflops);
    if (w16_gflops > 0 && tiled_gflops > 0)
        printf("  16-warp vs 8-warp: %.4f× (%+.2f%%)\n",
               w16_gflops / tiled_gflops,
               100.0 * (w16_gflops - tiled_gflops) / tiled_gflops);
    if (w16e_gflops > 0 && tiled_gflops > 0)
        printf("  16-warp+epi vs 8-warp: %.4f× (%+.2f%%)\n",
               w16e_gflops / tiled_gflops,
               100.0 * (w16e_gflops - tiled_gflops) / tiled_gflops);
    if (w16e_gflops > 0 && w16_gflops > 0)
        printf("  16-warp+epi vs 16-warp(direct): %.4f× (%+.2f%%)\n",
               w16e_gflops / w16_gflops,
               100.0 * (w16e_gflops - w16_gflops) / w16_gflops);

    // --- Cleanup ---
    cuMemFree(dev_a_fp16);
    cuMemFree(dev_b_fp16);
    cuMemFree(dev_c);
    if (naive_module) cuModuleUnload(naive_module);
    if (tiled_module) cuModuleUnload(tiled_module);
    if (direct_module) cuModuleUnload(direct_module);
    if (w16_module) cuModuleUnload(w16_module);
    if (w16e_module) cuModuleUnload(w16e_module);
    cuCtxDestroy(cu_context);
    free(host_a_fp32); free(host_b_fp32); free(host_c_fp32);
    free(host_ref); free(host_a_fp16); free(host_b_fp16);

    return 0;
}
