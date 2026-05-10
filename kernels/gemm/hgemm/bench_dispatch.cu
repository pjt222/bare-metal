/*
 * bench_dispatch.cu — measure hgemm_dispatch wrapper vs each fixed variant
 * across the same shape sweep used by bench_splitk.cu. Validates that
 * the heuristic in hgemm_dispatch.cuh picks the right kernel per shape.
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o hgemm_16warp.sm_86.cubin        hgemm_16warp.cu
 *   nvcc --cubin -arch=sm_86 -O2 -o hgemm_16warp_splitk.sm_86.cubin hgemm_16warp_splitk.cu
 *   nvcc -arch=sm_86 -O2 -std=c++17 -o bench_dispatch bench_dispatch.cu -lcuda -I../common
 */

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cuda.h>
#include <cuda_fp16.h>

#include "../../_common/bench.h"
#include "../../_common/check.h"
#include "hgemm_dispatch.cuh"

#define BM 128
#define BN 128
#define BK 32
#define PAD_A 8
#define PAD_B 8
#define BLOCK_THREADS 512

struct Result { double ms; double gflops; const char *variant; };

static double run_kernel(CUfunction fn,
                         CUdeviceptr dA, CUdeviceptr dB, CUdeviceptr dC,
                         int M, int N, int K,
                         int grid_x, int grid_y, int grid_z,
                         size_t smem_dyn,
                         int k_split_arg /* 0 if not splitk */) {
    auto launch = [&]() {
        if (k_split_arg > 0) {
            cuMemsetD8(dC, 0, (size_t)M * N * sizeof(float));
            void *args[] = { &dA, &dB, &dC, &M, &N, &K, &k_split_arg };
            CHECK_CU(cuLaunchKernel(fn, grid_x, grid_y, grid_z,
                                    BLOCK_THREADS, 1, 1,
                                    (unsigned)smem_dyn, 0, args, 0));
        } else {
            void *args[] = { &dA, &dB, &dC, &M, &N, &K };
            CHECK_CU(cuLaunchKernel(fn, grid_x, grid_y, grid_z,
                                    BLOCK_THREADS, 1, 1, 0, 0, args, 0));
        }
    };
    for (int i = 0; i < 5; i++) launch();
    CHECK_CU(cuCtxSynchronize());
    BenchTimer t;
    const int iters = 30;
    t.start();
    for (int i = 0; i < iters; i++) launch();
    return t.stop_ms() / iters;
}

static double dispatch_ms(const hgemm_dispatch::Handles &h,
                          CUdeviceptr dA, CUdeviceptr dB, CUdeviceptr dC,
                          int M, int N, int K,
                          hgemm_dispatch::Variant *picked_out) {
    auto launch = [&]() {
        *picked_out = hgemm_dispatch::launch(h, dA, dB, dC, M, N, K);
    };
    for (int i = 0; i < 5; i++) launch();
    CHECK_CU(cuCtxSynchronize());
    BenchTimer t;
    const int iters = 30;
    t.start();
    for (int i = 0; i < iters; i++) launch();
    return t.stop_ms() / iters;
}

int main() {
    CHECK_CU(cuInit(0));
    CUdevice dev; CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx; CHECK_CU(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK_CU(cuCtxSetCurrent(ctx));

    char devname[256]; CHECK_CU(cuDeviceGetName(devname, sizeof(devname), dev));
    int sm_count = 0;
    CHECK_CU(cuDeviceGetAttribute(&sm_count,
        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    printf("Device: %s  (%d SMs)\n", devname, sm_count);

    CUmodule mod_std, mod_sk;
    CUfunction fn_std, fn_sk;
    CHECK_CU(cuModuleLoad(&mod_std, "hgemm_16warp.sm_86.cubin"));
    CHECK_CU(cuModuleLoad(&mod_sk,  "hgemm_16warp_splitk.sm_86.cubin"));
    CHECK_CU(cuModuleGetFunction(&fn_std, mod_std, "hgemm_16warp"));
    CHECK_CU(cuModuleGetFunction(&fn_sk,  mod_sk,  "hgemm_16warp_splitk"));

    int STRIDE_A = BK + PAD_A;
    int STRIDE_B = BN + PAD_B;
    size_t smem_dyn = 2 * BM * STRIDE_A * sizeof(__half)
                    + 2 * BK * STRIDE_B * sizeof(__half)
                    + 16 * 16 * 16 * sizeof(float);
    CHECK_CU(cuFuncSetAttribute(fn_sk,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem_dyn));

    hgemm_dispatch::Handles handles { fn_std, fn_sk, smem_dyn };

    struct Shape { int M; int N; int K; const char *label; };
    Shape shapes[] = {
        // Square / large
        { 4096, 4096, 4096, "4096^3 (square)" },
        { 2048, 2048, 2048, "2048^3 (square)" },
        { 1024, 1024, 1024, "1024^3 (square)" },
        // Skinny / target K-split regime
        {  512,  512, 4096, "512x512x4096" },
        {  256,  256, 4096, "256x256x4096" },
        {  256,  256, 8192, "256x256x8192" },
        {  128,  128, 8192, "128x128x8192" },
        // Border cases
        { 1024, 1024, 4096, "1024x1024x4096 (mid)" },
        {  512,  512, 1024, "512x512x1024 (low K)"  },
    };
    int n_shapes = sizeof(shapes) / sizeof(Shape);

    printf("\n%-25s %5s %12s %12s %12s %14s %8s\n",
           "shape", "pick", "std ms",
           "sk_best ms", "dispatch ms", "best variant", "speedup");
    printf("%-25s %5s %12s %12s %12s %14s %8s\n",
           "-----", "----", "------",
           "----------", "-----------", "------------", "-------");

    int correct_picks = 0, total = 0;
    double sp_sum = 0;

    for (int s = 0; s < n_shapes; s++) {
        int M = shapes[s].M, N = shapes[s].N, K = shapes[s].K;
        if (M % BM || N % BN || K % BK) continue;

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

        // Standard
        double std_ms = run_kernel(fn_std, dA, dB, dC, M, N, K,
                                    N/BN, M/BM, 1, 0, 0);

        // SplitK best across {2,4,8}
        double sk_best_ms = 1e9;
        int sk_best_factor = 0;
        for (int ks : {2, 4, 8}) {
            int k_tiles = (K + BK - 1) / BK;
            if (k_tiles % ks != 0) continue;
            double ms = run_kernel(fn_sk, dA, dB, dC, M, N, K,
                                    N/BN, M/BM, ks, smem_dyn, ks);
            if (ms < sk_best_ms) { sk_best_ms = ms; sk_best_factor = ks; }
        }

        // Best of all variants
        double best_ms = std_ms;
        const char *best_variant = "standard";
        if (sk_best_ms < best_ms) {
            best_ms = sk_best_ms;
            static char buf[16];
            snprintf(buf, sizeof(buf), "splitk_%d", sk_best_factor);
            best_variant = buf;
        }

        // Dispatch wrapper
        hgemm_dispatch::Variant picked;
        double disp_ms = dispatch_ms(handles, dA, dB, dC, M, N, K, &picked);
        const char *picked_name = hgemm_dispatch::variant_name(picked);

        bool correct = (disp_ms < best_ms * 1.05);  // within 5% of best
        if (correct) correct_picks++;
        total++;
        double speedup = std_ms / disp_ms;
        sp_sum += speedup;

        printf("%-25s %5s %12.3f %12.3f %12.3f %14s %7.3fx\n",
               shapes[s].label,
               picked_name, std_ms, sk_best_ms, disp_ms,
               best_variant, speedup);

        cuMemFree(dA); cuMemFree(dB); cuMemFree(dC);
    }

    printf("\nDispatch correct picks: %d / %d (within 5%% of measured best)\n",
           correct_picks, total);
    printf("Geomean speedup vs always-standard: %.3fx\n",
           (total > 0) ? (sp_sum / total) : 1.0);
    return 0;
}
