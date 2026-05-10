/*
 * bench_persistent.cu — A/B test of hgemm_16warp vs hgemm_16warp_persistent
 * across multiple problem sizes. Tests #86's claim that persistent dispatch
 * gives ~1.15x across GEMM kernels.
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o hgemm_16warp.sm_86.cubin            hgemm_16warp.cu
 *   nvcc --cubin -arch=sm_86 -O2 -o hgemm_16warp_persistent.sm_86.cubin hgemm_16warp_persistent.cu
 *   nvcc -arch=sm_86 -O2 -std=c++17 -o bench_persistent bench_persistent.cu -lcuda -I../common
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <utility>
#include <cuda.h>
#include <cuda_fp16.h>

#include "../common/bench.h"
#include "../common/check.h"

#define BM 128
#define BN 128
#define BLOCK_THREADS 512

static void cpu_gemm(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++) s += A[i*K+k] * B[k*N+j];
            C[i*N+j] = s;
        }
}

struct Result { double ms; double gflops; };

static Result run_one(CUfunction fn, CUdeviceptr dA, CUdeviceptr dB, CUdeviceptr dC,
                      int M, int N, int K,
                      int grid_x, int grid_y, int grid_z) {
    void *args[] = { &dA, &dB, &dC, &M, &N, &K };
    // Warmup
    for (int i = 0; i < 5; i++) {
        CHECK_CU(cuLaunchKernel(fn, grid_x, grid_y, grid_z,
                                BLOCK_THREADS, 1, 1, 0, 0, args, 0));
    }
    CHECK_CU(cuCtxSynchronize());

    BenchTimer t;
    const int iters = 30;
    t.start();
    for (int i = 0; i < iters; i++) {
        CHECK_CU(cuLaunchKernel(fn, grid_x, grid_y, grid_z,
                                BLOCK_THREADS, 1, 1, 0, 0, args, 0));
    }
    float ms = t.stop_ms() / iters;
    double gf = (2.0 * (double)M * N * K) / (ms / 1000.0) / 1e9;
    return { ms, gf };
}

int main(int argc, char **argv) {
    CHECK_CU(cuInit(0));
    CUdevice dev; CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx; CHECK_CU(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK_CU(cuCtxSetCurrent(ctx));

    char devname[256]; CHECK_CU(cuDeviceGetName(devname, sizeof(devname), dev));
    int sm_count = 0;
    CHECK_CU(cuDeviceGetAttribute(&sm_count,
        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    printf("Device: %s  (%d SMs)\n", devname, sm_count);

    CUmodule mod_orig, mod_per;
    CUfunction fn_orig, fn_per;
    CHECK_CU(cuModuleLoad(&mod_orig, "hgemm_16warp.sm_86.cubin"));
    CHECK_CU(cuModuleLoad(&mod_per,  "hgemm_16warp_persistent.sm_86.cubin"));
    CHECK_CU(cuModuleGetFunction(&fn_orig, mod_orig, "hgemm_16warp"));
    CHECK_CU(cuModuleGetFunction(&fn_per,  mod_per,  "hgemm_16warp_persistent"));

    // Sweep over problem sizes.
    struct Shape { int M; int N; int K; const char *label; };
    Shape shapes[] = {
        { 1024, 1024, 1024, "1024^3 (small)"   },
        { 2048, 2048, 2048, "2048^3 (medium)"  },
        { 4096, 4096, 4096, "4096^3 (large)"   },
        { 8192, 8192, 8192, "8192^3 (xlarge)"  },
        // Skinny cases where launch overhead matters more
        {  512,  512, 8192, "512x512x8192 (skinny K)" },
        {  256,  256, 8192, "256x256x8192 (very skinny)" }
    };
    int n_shapes = sizeof(shapes) / sizeof(Shape);

    printf("\n%-32s %12s %12s %12s %12s %8s\n",
           "shape", "orig ms", "orig GFLOPS", "per  ms", "per  GFLOPS", "speedup");
    printf("%-32s %12s %12s %12s %12s %8s\n",
           "-----", "-------", "-----------", "-------", "-----------", "-------");

    int n_tested  = 0;
    double sp_min = 1e9, sp_max = 0, sp_sum = 0;

    for (int i = 0; i < n_shapes; i++) {
        int M = shapes[i].M, N = shapes[i].N, K = shapes[i].K;
        if (M % BM || N % BN) {
            printf("SKIP %s (not multiple of 128)\n", shapes[i].label);
            continue;
        }

        size_t nA = (size_t)M*K, nB = (size_t)K*N, nC = (size_t)M*N;
        std::vector<__half> hA(nA), hB(nB);
        for (size_t k = 0; k < nA; k++) hA[k] = __float2half(((k*17+3)%11)/11.0f - 0.45f);
        for (size_t k = 0; k < nB; k++) hB[k] = __float2half(((k*23+5)%13)/13.0f - 0.45f);

        CUdeviceptr dA, dB, dC;
        CHECK_CU(cuMemAlloc(&dA, nA * sizeof(__half)));
        CHECK_CU(cuMemAlloc(&dB, nB * sizeof(__half)));
        CHECK_CU(cuMemAlloc(&dC, nC * sizeof(float)));
        CHECK_CU(cuMemcpyHtoD(dA, hA.data(), nA * sizeof(__half)));
        CHECK_CU(cuMemcpyHtoD(dB, hB.data(), nB * sizeof(__half)));

        // Original: one block per output tile (grid_x = N/BN, grid_y = M/BM).
        int grid_x_orig = N / BN, grid_y_orig = M / BM;
        Result r_orig = run_one(fn_orig, dA, dB, dC, M, N, K,
                                grid_x_orig, grid_y_orig, 1);

        // Persistent: one block per SM slot (sm_count * 2 = 92 on GA104).
        int n_blocks_per = sm_count * 2;
        Result r_per = run_one(fn_per, dA, dB, dC, M, N, K,
                               n_blocks_per, 1, 1);

        double speedup = r_orig.ms / r_per.ms;
        printf("%-32s %12.3f %12.0f %12.3f %12.0f %7.3fx\n",
               shapes[i].label,
               r_orig.ms, r_orig.gflops,
               r_per.ms,  r_per.gflops,
               speedup);
        n_tested++;
        sp_sum += speedup;
        if (speedup < sp_min) sp_min = speedup;
        if (speedup > sp_max) sp_max = speedup;

        cuMemFree(dA); cuMemFree(dB); cuMemFree(dC);
    }

    if (n_tested > 0) {
        printf("\nSpeedup geomean across %d shapes: ~%.3fx  (min %.3fx, max %.3fx)\n",
               n_tested, sp_sum / n_tested, sp_min, sp_max);
    }
    return 0;
}
