/*
 * bench_epi_pad.cu — A/B test of hgemm_16warp_epi vs hgemm_16warp_epi_pad.
 *
 * NCU on the original epi variant (issue #99) showed 75.9% bank conflict
 * rate (335M load conflicts / 458M wavefronts), explaining its triple
 * stall (mio=21, short_sb=17, barrier=17 from Observation U).
 *
 * The padded variant adds PAD_B=8 (smem_b stride 128 → 136 halfs, breaks
 * 8-way conflict). PAD_A=0 because adding both pads exceeds the 48 KB
 * static smem cap. Padded variant uses dynamic smem.
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o hgemm_16warp_epi.sm_86.cubin     hgemm_16warp_epi.cu
 *   nvcc --cubin -arch=sm_86 -O2 -o hgemm_16warp_epi_pad.sm_86.cubin hgemm_16warp_epi_pad.cu
 *   nvcc -arch=sm_86 -O2 -o bench_epi_pad bench_epi_pad.cu -lcuda -I../common
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <utility>
#include <cuda.h>
#include <cuda_fp16.h>

#include "../../_common/bench.h"
#include "../../_common/check.h"

#define BM 128
#define BN 128
#define BK 32
#define PAD_A 8
#define PAD_B 8

static void cpu_gemm(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++) s += A[i*K+k] * B[k*N+j];
            C[i*N+j] = s;
        }
}

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 4096;
    int N = (argc > 2) ? atoi(argv[2]) : 4096;
    int K = (argc > 3) ? atoi(argv[3]) : 4096;
    if (M % BM || N % BN || K % BK) {
        fprintf(stderr, "M,N,K must be multiples of %d,%d,%d\n", BM, BN, BK);
        return 1;
    }

    printf("=== HGEMM 16warp_epi: unpadded vs +8 PAD_B ===\n");
    printf("M=%d N=%d K=%d\n\n", M, N, K);

    CHECK_CU(cuInit(0));
    CUdevice dev; CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx; CHECK_CU(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK_CU(cuCtxSetCurrent(ctx));

    char devname[256]; CHECK_CU(cuDeviceGetName(devname, sizeof(devname), dev));
    printf("Device: %s\n\n", devname);

    CUmodule mod_orig, mod_pad;
    CUfunction fn_orig, fn_pad;
    if (cuModuleLoad(&mod_orig, "hgemm_16warp_epi.sm_86.cubin") != CUDA_SUCCESS) {
        fprintf(stderr, "Cannot load hgemm_16warp_epi.sm_86.cubin\n"); return 1;
    }
    if (cuModuleLoad(&mod_pad, "hgemm_16warp_epi_pad.sm_86.cubin") != CUDA_SUCCESS) {
        fprintf(stderr, "Cannot load hgemm_16warp_epi_pad.sm_86.cubin\n"); return 1;
    }
    CHECK_CU(cuModuleGetFunction(&fn_orig, mod_orig, "hgemm_16warp_epi"));
    CHECK_CU(cuModuleGetFunction(&fn_pad,  mod_pad,  "hgemm_16warp_epi_pad"));

    // Smem sizes for the padded variant (dynamic smem)
    int STRIDE_A = BK + PAD_A;
    int STRIDE_B = BN + PAD_B;
    size_t smem_pad = 2 * BM * STRIDE_A * sizeof(__half)
                    + 2 * BK * STRIDE_B * sizeof(__half)
                    + 16 * 16 * 16 * sizeof(float);     // epilogue
    CHECK_CU(cuFuncSetAttribute(fn_pad,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem_pad));
    printf("Padded smem: %zu bytes (%.1f KB)\n\n", smem_pad, smem_pad / 1024.0);

    size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;
    float *hA = (float*)malloc(nA * sizeof(float));
    float *hB = (float*)malloc(nB * sizeof(float));
    __half *hAh = (__half*)malloc(nA * sizeof(__half));
    __half *hBh = (__half*)malloc(nB * sizeof(__half));
    if (!hA || !hB || !hAh || !hBh) { fprintf(stderr, "alloc fail\n"); return 1; }
    for (size_t i = 0; i < nA; i++) hA[i] = (float)((i*17+3) % 11) / 11.0f - 0.45f;
    for (size_t i = 0; i < nB; i++) hB[i] = (float)((i*23+5) % 13) / 13.0f - 0.45f;
    for (size_t i = 0; i < nA; i++) hAh[i] = __float2half(hA[i]);
    for (size_t i = 0; i < nB; i++) hBh[i] = __float2half(hB[i]);

    CUdeviceptr dA, dB, dC;
    CHECK_CU(cuMemAlloc(&dA, nA * sizeof(__half)));
    CHECK_CU(cuMemAlloc(&dB, nB * sizeof(__half)));
    CHECK_CU(cuMemAlloc(&dC, nC * sizeof(float)));
    CHECK_CU(cuMemcpyHtoD(dA, hAh, nA * sizeof(__half)));
    CHECK_CU(cuMemcpyHtoD(dB, hBh, nB * sizeof(__half)));

    int grid_x = N / BN, grid_y = M / BM;
    void *args[] = { &dA, &dB, &dC, &M, &N, &K };

    // Correctness on a smaller subset
    int Ms = std::min(256, M), Ns = std::min(256, N), Ks = std::min(256, K);
    {
        printf("Correctness (M=N=K=%d subset):\n", Ms);
        std::vector<float> sA(Ms*Ks), sB(Ks*Ns), sRef(Ms*Ns);
        std::vector<__half> sAh(Ms*Ks), sBh(Ks*Ns);
        for (int i = 0; i < Ms; i++) for (int k = 0; k < Ks; k++) sA[i*Ks+k] = hA[i*K+k];
        for (int k = 0; k < Ks; k++) for (int j = 0; j < Ns; j++) sB[k*Ns+j] = hB[k*N+j];
        for (size_t i = 0; i < sA.size(); i++) sAh[i] = __float2half(sA[i]);
        for (size_t i = 0; i < sB.size(); i++) sBh[i] = __float2half(sB[i]);
        cpu_gemm(sA.data(), sB.data(), sRef.data(), Ms, Ns, Ks);

        CUdeviceptr sdA, sdB, sdC;
        cuMemAlloc(&sdA, sA.size() * sizeof(__half));
        cuMemAlloc(&sdB, sB.size() * sizeof(__half));
        cuMemAlloc(&sdC, sRef.size() * sizeof(float));
        cuMemcpyHtoD(sdA, sAh.data(), sA.size() * sizeof(__half));
        cuMemcpyHtoD(sdB, sBh.data(), sB.size() * sizeof(__half));
        std::vector<float> sOut(sRef.size());
        int gx = Ns / BN, gy = Ms / BM;
        void *sargs[] = { &sdA, &sdB, &sdC, &Ms, &Ns, &Ks };

        for (auto [name, fn, smem] : std::initializer_list<std::tuple<const char*, CUfunction, size_t>>{
                {"hgemm_16warp_epi",     fn_orig, 0},
                {"hgemm_16warp_epi_pad", fn_pad,  smem_pad}}) {
            cuMemsetD8(sdC, 0, sRef.size() * sizeof(float));
            CHECK_CU(cuLaunchKernel(fn, gx, gy, 1, 512, 1, 1, (unsigned)smem, 0, sargs, 0));
            CHECK_CU(cuCtxSynchronize());
            cuMemcpyDtoH(sOut.data(), sdC, sRef.size() * sizeof(float));
            CheckResult cr = check_fp32(sRef.data(), sOut.data(), sRef.size(), 1e-2f, 1e-2f);
            print_check_result(name, cr);
        }
        cuMemFree(sdA); cuMemFree(sdB); cuMemFree(sdC);
    }
    printf("\n");

    auto run_one = [&](const char *name, CUfunction fn, size_t smem) -> double {
        for (int i = 0; i < 5; i++) {
            CHECK_CU(cuLaunchKernel(fn, grid_x, grid_y, 1, 512, 1, 1, (unsigned)smem, 0, args, 0));
        }
        CHECK_CU(cuCtxSynchronize());
        BenchTimer t;
        const int iters = 20;
        t.start();
        for (int i = 0; i < iters; i++)
            CHECK_CU(cuLaunchKernel(fn, grid_x, grid_y, 1, 512, 1, 1, (unsigned)smem, 0, args, 0));
        float ms = t.stop_ms() / iters;
        double gf = (2.0 * M * N * K) / (ms / 1000.0) / 1e9;
        printf("  %-32s %7.3f ms  %8.1f GFLOPS\n", name, ms, gf);
        return gf;
    };

    printf("Performance (M=%d N=%d K=%d):\n", M, N, K);
    double g_orig = run_one("hgemm_16warp_epi (baseline)", fn_orig, 0);
    double g_pad  = run_one("hgemm_16warp_epi_pad",        fn_pad,  smem_pad);
    printf("\nSpeedup pad vs baseline: %.3fx\n", g_pad / g_orig);

    cuMemFree(dA); cuMemFree(dB); cuMemFree(dC);
    free(hA); free(hB); free(hAh); free(hBh);
    return 0;
}
