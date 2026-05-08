/*
 * bench_splitk.cu — A/B test of hgemm_16warp vs hgemm_16warp_splitk
 * across multiple problem sizes. Tests #87 claim that K-split gives
 * ~1.5× for skinny matrices.
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o hgemm_16warp.sm_86.cubin        hgemm_16warp.cu
 *   nvcc --cubin -arch=sm_86 -O2 -o hgemm_16warp_splitk.sm_86.cubin hgemm_16warp_splitk.cu
 *   nvcc -arch=sm_86 -O2 -std=c++17 -o bench_splitk bench_splitk.cu -lcuda -I../common
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda.h>
#include <cuda_fp16.h>

#include "../common/bench.h"
#include "../common/check.h"

#define BM 128
#define BN 128
#define BK 32
#define PAD_A 8
#define PAD_B 8
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

static Result time_orig(CUfunction fn, CUdeviceptr dA, CUdeviceptr dB, CUdeviceptr dC,
                        int M, int N, int K) {
    void *args[] = { &dA, &dB, &dC, &M, &N, &K };
    int gx = N / BN, gy = M / BM;
    for (int i = 0; i < 5; i++)
        CHECK_CU(cuLaunchKernel(fn, gx, gy, 1, BLOCK_THREADS, 1, 1, 0, 0, args, 0));
    CHECK_CU(cuCtxSynchronize());
    BenchTimer t;
    const int iters = 30;
    t.start();
    for (int i = 0; i < iters; i++)
        CHECK_CU(cuLaunchKernel(fn, gx, gy, 1, BLOCK_THREADS, 1, 1, 0, 0, args, 0));
    float ms = t.stop_ms() / iters;
    return { ms, (2.0 * (double)M * N * K) / (ms / 1000.0) / 1e9 };
}

static Result time_splitk(CUfunction fn, CUdeviceptr dA, CUdeviceptr dB, CUdeviceptr dC,
                          int M, int N, int K, int k_split,
                          size_t smem_dyn) {
    void *args[] = { &dA, &dB, &dC, &M, &N, &K, &k_split };
    int gx = N / BN, gy = M / BM;
    auto launch = [&]() {
        CHECK_CU(cuMemsetD8(dC, 0, (size_t)M * N * sizeof(float)));
        CHECK_CU(cuLaunchKernel(fn, gx, gy, k_split,
                                BLOCK_THREADS, 1, 1, (unsigned)smem_dyn, 0, args, 0));
    };
    for (int i = 0; i < 5; i++) launch();
    CHECK_CU(cuCtxSynchronize());
    BenchTimer t;
    const int iters = 30;
    t.start();
    for (int i = 0; i < iters; i++) launch();
    float ms = t.stop_ms() / iters;
    return { ms, (2.0 * (double)M * N * K) / (ms / 1000.0) / 1e9 };
}

int main(int /*argc*/, char ** /*argv*/) {
    CHECK_CU(cuInit(0));
    CUdevice dev; CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx; CHECK_CU(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK_CU(cuCtxSetCurrent(ctx));

    char devname[256]; CHECK_CU(cuDeviceGetName(devname, sizeof(devname), dev));
    int sm_count = 0;
    CHECK_CU(cuDeviceGetAttribute(&sm_count,
        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    printf("Device: %s  (%d SMs)\n", devname, sm_count);

    CUmodule mod_orig, mod_sk;
    CUfunction fn_orig, fn_sk;
    CHECK_CU(cuModuleLoad(&mod_orig, "hgemm_16warp.sm_86.cubin"));
    CHECK_CU(cuModuleLoad(&mod_sk,   "hgemm_16warp_splitk.sm_86.cubin"));
    CHECK_CU(cuModuleGetFunction(&fn_orig, mod_orig, "hgemm_16warp"));
    CHECK_CU(cuModuleGetFunction(&fn_sk,   mod_sk,   "hgemm_16warp_splitk"));

    int STRIDE_A = BK + PAD_A;
    int STRIDE_B = BN + PAD_B;
    size_t smem_dyn = 2 * BM * STRIDE_A * sizeof(__half)
                    + 2 * BK * STRIDE_B * sizeof(__half)
                    + 16 * 16 * 16 * sizeof(float);   // epi_tile NUM_WARPS=16
    CHECK_CU(cuFuncSetAttribute(fn_sk,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem_dyn));
    printf("Splitk dynamic smem: %.1f KB\n\n", smem_dyn / 1024.0);

    struct Shape { int M; int N; int K; const char *label; };
    Shape shapes[] = {
        { 4096, 4096, 4096, "4096^3 (square, large)" },
        { 2048, 2048, 2048, "2048^3 (square, med)" },
        { 1024, 1024, 1024, "1024^3 (square, small)" },
        // Skinny: small M*N, large K -- target regime for K-split
        {  256,  256, 4096, "256x256x4096 (skinny)" },
        {  256,  256, 8192, "256x256x8192 (very skinny)" },
        {  128,  128, 8192, "128x128x8192 (extreme skinny)" },
        {  512,  512, 4096, "512x512x4096 (mid skinny)" }
    };
    int n_shapes = sizeof(shapes) / sizeof(Shape);

    int k_splits[] = { 2, 4, 8, 16 };

    printf("%-30s %12s %12s %12s %12s %8s\n",
           "shape", "orig ms", "orig GFLOPS", "split ms", "split GFLOPS", "speedup");
    printf("%-30s %12s %12s %12s %12s %8s\n",
           "-----", "-------", "-----------", "--------", "------------", "-------");

    for (int s = 0; s < n_shapes; s++) {
        int M = shapes[s].M, N = shapes[s].N, K = shapes[s].K;
        if (M % BM || N % BN || K % BK) {
            printf("SKIP %s\n", shapes[s].label); continue;
        }

        size_t nA = (size_t)M*K, nB = (size_t)K*N, nC = (size_t)M*N;
        std::vector<__half> hA(nA), hB(nB);
        for (size_t i = 0; i < nA; i++) hA[i] = __float2half(((i*17+3)%11)/11.0f - 0.45f);
        for (size_t i = 0; i < nB; i++) hB[i] = __float2half(((i*23+5)%13)/13.0f - 0.45f);

        CUdeviceptr dA, dB, dC;
        CHECK_CU(cuMemAlloc(&dA, nA * sizeof(__half)));
        CHECK_CU(cuMemAlloc(&dB, nB * sizeof(__half)));
        CHECK_CU(cuMemAlloc(&dC, nC * sizeof(float)));
        CHECK_CU(cuMemcpyHtoD(dA, hA.data(), nA * sizeof(__half)));
        CHECK_CU(cuMemcpyHtoD(dB, hB.data(), nB * sizeof(__half)));

        Result r_orig = time_orig(fn_orig, dA, dB, dC, M, N, K);

        // Try several K_split values, pick the best.
        Result r_best = { 1e9, 0 };
        int best_split = 0;
        for (int ks : k_splits) {
            int k_tiles = (K + BK - 1) / BK;
            if (k_tiles % ks != 0) continue;       // require even split for cleanliness
            Result r = time_splitk(fn_sk, dA, dB, dC, M, N, K, ks, smem_dyn);
            if (r.ms < r_best.ms) { r_best = r; best_split = ks; }
        }

        double speedup = (r_best.gflops > 0) ? r_orig.ms / r_best.ms : 0.0;
        printf("%-30s %12.3f %12.0f %12.3f %12.0f %7.3fx (k_split=%d)\n",
               shapes[s].label,
               r_orig.ms, r_orig.gflops,
               r_best.ms, r_best.gflops,
               speedup, best_split);

        cuMemFree(dA); cuMemFree(dB); cuMemFree(dC);
    }
    return 0;
}
