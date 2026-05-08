/*
 * bench_aligned.cu — A/B test of hgemm_16warp vs hgemm_16warp_aligned.
 *
 * The aligned variant drops partial-tile boundary checks. Hypothesis: the
 * branches were generating the IMAD chain that produced
 * stall_math_throttle = 35.46 in NCU profile.
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o hgemm_16warp.sm_86.cubin hgemm_16warp.cu
 *   nvcc --cubin -arch=sm_86 -O2 -o hgemm_16warp_aligned.sm_86.cubin hgemm_16warp_aligned.cu
 *   nvcc -arch=sm_86 -O2 -o bench_aligned bench_aligned.cu -lcuda -I../common
 *
 * Run:
 *   ./bench_aligned [M] [N] [K]   # default 4096 4096 4096
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <utility>
#include <algorithm>
#include <cuda.h>
#include <cuda_fp16.h>

#include "../common/bench.h"
#include "../common/check.h"

#define BM 128
#define BN 128
#define BK 32

static void cpu_gemm(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++) s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 4096;
    int N = (argc > 2) ? atoi(argv[2]) : 4096;
    int K = (argc > 3) ? atoi(argv[3]) : 4096;
    if (M % BM || N % BN || K % BK) {
        fprintf(stderr, "M,N,K must be multiples of %d,%d,%d (aligned variant)\n",
                BM, BN, BK);
        return 1;
    }

    printf("=== HGEMM IMAD-chain hunt: 16warp vs 16warp_aligned ===\n");
    printf("M=%d N=%d K=%d\n\n", M, N, K);

    CHECK_CU(cuInit(0));
    CUdevice dev; CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx; CHECK_CU(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK_CU(cuCtxSetCurrent(ctx));

    char devname[256]; CHECK_CU(cuDeviceGetName(devname, sizeof(devname), dev));
    printf("Device: %s\n\n", devname);

    CUmodule mod_orig, mod_aln;
    CUfunction fn_orig, fn_aln;
    if (cuModuleLoad(&mod_orig, "hgemm_16warp.sm_86.cubin") != CUDA_SUCCESS) {
        fprintf(stderr, "Cannot load hgemm_16warp.sm_86.cubin\n"); return 1;
    }
    if (cuModuleLoad(&mod_aln, "hgemm_16warp_aligned.sm_86.cubin") != CUDA_SUCCESS) {
        fprintf(stderr, "Cannot load hgemm_16warp_aligned.sm_86.cubin\n"); return 1;
    }
    CHECK_CU(cuModuleGetFunction(&fn_orig, mod_orig, "hgemm_16warp"));
    CHECK_CU(cuModuleGetFunction(&fn_aln,  mod_aln,  "hgemm_16warp_aligned"));

    size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;

    // Host buffers
    float *hA = (float*)malloc(nA * sizeof(float));
    float *hB = (float*)malloc(nB * sizeof(float));
    float *hC_ref = (float*)malloc(nC * sizeof(float));
    float *hC_out = (float*)malloc(nC * sizeof(float));
    __half *hAh = (__half*)malloc(nA * sizeof(__half));
    __half *hBh = (__half*)malloc(nB * sizeof(__half));
    if (!hA || !hB || !hC_ref || !hC_out || !hAh || !hBh) {
        fprintf(stderr, "host alloc failed\n"); return 1;
    }
    for (size_t i = 0; i < nA; i++) hA[i] = (float)((i * 17 + 3) % 11) / 11.0f - 0.45f;
    for (size_t i = 0; i < nB; i++) hB[i] = (float)((i * 23 + 5) % 13) / 13.0f - 0.45f;
    for (size_t i = 0; i < nA; i++) hAh[i] = __float2half(hA[i]);
    for (size_t i = 0; i < nB; i++) hBh[i] = __float2half(hB[i]);

    // Device buffers
    CUdeviceptr dA, dB, dC;
    CHECK_CU(cuMemAlloc(&dA, nA * sizeof(__half)));
    CHECK_CU(cuMemAlloc(&dB, nB * sizeof(__half)));
    CHECK_CU(cuMemAlloc(&dC, nC * sizeof(float)));
    CHECK_CU(cuMemcpyHtoD(dA, hAh, nA * sizeof(__half)));
    CHECK_CU(cuMemcpyHtoD(dB, hBh, nB * sizeof(__half)));

    int grid_x = N / BN, grid_y = M / BM;
    void *args[] = { &dA, &dB, &dC, &M, &N, &K };

    // Correctness on small subset (skip full CPU GEMM at 4096³, too slow)
    int Ms = std::min(256, M), Ns = std::min(256, N), Ks = std::min(256, K);
    {
        printf("Correctness (M=N=K=%d subset):\n", Ms);
        // Allocate slim ref arrays
        std::vector<float> sA(Ms * Ks), sB(Ks * Ns), sRef(Ms * Ns);
        std::vector<__half> sAh(Ms * Ks), sBh(Ks * Ns);
        for (int i = 0; i < Ms; i++)
            for (int k = 0; k < Ks; k++) sA[i*Ks+k] = hA[i*K+k];
        for (int k = 0; k < Ks; k++)
            for (int j = 0; j < Ns; j++) sB[k*Ns+j] = hB[k*N+j];
        for (size_t i = 0; i < sA.size(); i++) sAh[i] = __float2half(sA[i]);
        for (size_t i = 0; i < sB.size(); i++) sBh[i] = __float2half(sB[i]);
        cpu_gemm(sA.data(), sB.data(), sRef.data(), Ms, Ns, Ks);

        CUdeviceptr sdA, sdB, sdC;
        CHECK_CU(cuMemAlloc(&sdA, sA.size() * sizeof(__half)));
        CHECK_CU(cuMemAlloc(&sdB, sB.size() * sizeof(__half)));
        CHECK_CU(cuMemAlloc(&sdC, sRef.size() * sizeof(float)));
        CHECK_CU(cuMemcpyHtoD(sdA, sAh.data(), sA.size() * sizeof(__half)));
        CHECK_CU(cuMemcpyHtoD(sdB, sBh.data(), sB.size() * sizeof(__half)));
        std::vector<float> sOut(sRef.size());
        int gx = Ns / BN, gy = Ms / BM;
        void *sargs[] = { &sdA, &sdB, &sdC, &Ms, &Ns, &Ks };

        for (auto [name, fn] : std::initializer_list<std::pair<const char*, CUfunction>>{
                {"hgemm_16warp", fn_orig}, {"hgemm_16warp_aligned", fn_aln}}) {
            CHECK_CU(cuMemsetD8(sdC, 0, sRef.size() * sizeof(float)));
            CHECK_CU(cuLaunchKernel(fn, gx, gy, 1, 512, 1, 1, 0, 0, sargs, 0));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(sOut.data(), sdC, sRef.size() * sizeof(float)));
            CheckResult cr = check_fp32(sRef.data(), sOut.data(), sRef.size(), 1e-2f, 1e-2f);
            print_check_result(name, cr);
        }
        cuMemFree(sdA); cuMemFree(sdB); cuMemFree(sdC);
    }
    printf("\n");

    // Performance: warmup + timed runs
    auto run_one = [&](const char *name, CUfunction fn) -> double {
        for (int i = 0; i < 5; i++) {
            CHECK_CU(cuLaunchKernel(fn, grid_x, grid_y, 1, 512, 1, 1, 0, 0, args, 0));
        }
        CHECK_CU(cuCtxSynchronize());
        BenchTimer t;
        const int iters = 20;
        t.start();
        for (int i = 0; i < iters; i++) {
            CHECK_CU(cuLaunchKernel(fn, grid_x, grid_y, 1, 512, 1, 1, 0, 0, args, 0));
        }
        float ms = t.stop_ms() / iters;
        double gf = (2.0 * M * N * K) / (ms / 1000.0) / 1e9;
        printf("  %-32s %7.3f ms  %8.1f GFLOPS\n", name, ms, gf);
        return gf;
    };

    printf("Performance (M=%d N=%d K=%d):\n", M, N, K);
    double g_orig = run_one("hgemm_16warp (baseline)", fn_orig);
    double g_aln  = run_one("hgemm_16warp_aligned",   fn_aln);
    printf("\nSpeedup aligned vs baseline: %.3fx\n", g_aln / g_orig);

    cuMemFree(dA); cuMemFree(dB); cuMemFree(dC);
    free(hA); free(hB); free(hC_ref); free(hC_out); free(hAh); free(hBh);
    return 0;
}
