/*
 * bench_pipelined.cu — Pipelined vs baseline cross-attention (BenchDriver refactor)
 *
 * Build:
 *   nvcc -arch=sm_86 -O2 -o bench_pipelined bench_pipelined.cu -lcuda -I../../phase2/common
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>
#include <cuda_fp16.h>

#include "../../phase2/common/bench_driver.h"

#define D_HEAD   64
#define Br_BLOCK 64
#define Bc       64

static void fp32_to_fp16_arr(const float *src, unsigned short *dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        __half h = __float2half(src[i]);
        memcpy(&dst[i], &h, 2);
    }
}

static void cpu_cross_attn(
    const float *Q, const float *K, const float *V, float *O,
    int batch, int heads, int seq_q, int seq_kv, float scale
) {
    for (int b = 0; b < batch; b++)
    for (int h = 0; h < heads; h++) {
        size_t qo  = ((size_t)b * heads + h) * seq_q  * D_HEAD;
        size_t kvo = ((size_t)b * heads + h) * seq_kv * D_HEAD;
        for (int qi = 0; qi < seq_q; qi++) {
            float *scores = new float[seq_kv];
            float max_s = -1e38f;
            for (int kv = 0; kv < seq_kv; kv++) {
                double dot = 0.0;
                for (int d = 0; d < D_HEAD; d++)
                    dot += (double)Q[qo + qi*D_HEAD + d] * K[kvo + kv*D_HEAD + d];
                scores[kv] = (float)(dot * scale);
                if (scores[kv] > max_s) max_s = scores[kv];
            }
            float sum = 0.0f;
            for (int kv = 0; kv < seq_kv; kv++) {
                scores[kv] = expf(scores[kv] - max_s);
                sum += scores[kv];
            }
            for (int kv = 0; kv < seq_kv; kv++) scores[kv] /= sum;
            for (int d = 0; d < D_HEAD; d++) {
                double acc = 0.0;
                for (int kv = 0; kv < seq_kv; kv++)
                    acc += scores[kv] * V[kvo + kv*D_HEAD + d];
                O[qo + qi*D_HEAD + d] = (float)acc;
            }
            delete[] scores;
        }
    }
}

struct Config { int seq_q, seq_kv, batch, heads; const char *label; };

int main(void) {
    Config configs[] = {
        {  64,  77, 1, 8, "SD  8x8" },
        { 256,  77, 1, 8, "SD 16x16" },
        {1024,  77, 1, 8, "SD 32x32" },
        {4096,  77, 1, 8, "SD 64x64" },
        {1024, 512, 1, 8, "Long ctx" },
    };
    int ncfg = sizeof(configs) / sizeof(configs[0]);
    float scale = 1.0f / sqrtf((float)D_HEAD);

    BenchDriver driver;
    driver.init_context();

    CUfunction fn_base = driver.load_kernel("cross_attn.sm_86.cubin", "cross_attn_br16");
    CUfunction fn_pipe = driver.load_kernel("cross_attn_pipelined.sm_86.cubin", "cross_attn_pipelined");

    int smem_base = 48 * 1024;
    int smem_pipe = 64 * 1024;
    CHECK_CU(cuFuncSetAttribute(fn_base, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_base));
    CHECK_CU(cuFuncSetAttribute(fn_pipe, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_pipe));

    printf("=== Pipelined vs Baseline Cross-Attention ===\n\n");

    for (int ci = 0; ci < ncfg; ci++) {
        int sq = configs[ci].seq_q, skv = configs[ci].seq_kv;
        int batch = configs[ci].batch, heads = configs[ci].heads;
        size_t Qe  = (size_t)batch * heads * sq  * D_HEAD;
        size_t KVe = (size_t)batch * heads * skv * D_HEAD;

        auto hQ  = driver.host_alloc<float>(Qe);
        auto hK  = driver.host_alloc<float>(KVe);
        auto hV  = driver.host_alloc<float>(KVe);
        auto hRef= driver.host_alloc<float>(Qe);
        auto hBase=driver.host_alloc<float>(Qe);
        auto hPipe=driver.host_alloc<float>(Qe);

        srand(42 + ci);
        for (size_t i = 0; i < Qe;  i++) hQ[i] = 0.1f * ((float)rand()/RAND_MAX - 0.5f);
        for (size_t i = 0; i < KVe; i++) hK[i] = 0.1f * ((float)rand()/RAND_MAX - 0.5f);
        for (size_t i = 0; i < KVe; i++) hV[i] = 0.1f * ((float)rand()/RAND_MAX - 0.5f);

        bool ran_cpu = (sq * skv < 1024 * 512);
        if (ran_cpu) cpu_cross_attn(hQ.get(), hK.get(), hV.get(), hRef.get(), batch, heads, sq, skv, scale);

        auto hQh = driver.host_alloc<unsigned short>(Qe);
        auto hKh = driver.host_alloc<unsigned short>(KVe);
        auto hVh = driver.host_alloc<unsigned short>(KVe);
        fp32_to_fp16_arr(hQ.get(), hQh.get(), Qe);
        fp32_to_fp16_arr(hK.get(), hKh.get(), KVe);
        fp32_to_fp16_arr(hV.get(), hVh.get(), KVe);

        auto dQ = driver.device_alloc<unsigned short>(Qe);
        auto dK = driver.device_alloc<unsigned short>(KVe);
        auto dV = driver.device_alloc<unsigned short>(KVe);
        auto dO = driver.device_alloc<float>(Qe);

        driver.copy_h2d(dQ, hQh, Qe * 2);
        driver.copy_h2d(dK, hKh, KVe * 2);
        driver.copy_h2d(dV, hVh, KVe * 2);

        int gq = (sq + Br_BLOCK - 1) / Br_BLOCK;
        void *args[] = { &dQ.ptr, &dK.ptr, &dV.ptr, &dO.ptr, &sq, &skv, &heads, &scale };

        // Baseline
        CHECK_CU(cuLaunchKernel(fn_base, gq, heads, batch, 128, 1, 1, smem_base, nullptr, args, nullptr));
        CHECK_CU(cuCtxSynchronize());
        driver.copy_d2h(hBase, dO, Qe * sizeof(float));

        // Pipelined
        CHECK_CU(cuMemsetD32((CUdeviceptr)dO.ptr, 0, Qe));
        CHECK_CU(cuLaunchKernel(fn_pipe, gq, heads, batch, 128, 1, 1, smem_pipe, nullptr, args, nullptr));
        CHECK_CU(cuCtxSynchronize());
        driver.copy_d2h(hPipe, dO, Qe * sizeof(float));

        printf("--- %s (sq=%d skv=%d) ---\n", configs[ci].label, sq, skv);

        if (ran_cpu) {
            driver.check(hBase.get(), hRef.get(), (int)Qe, 1e-2f, 1.0f, "baseline vs CPU");
            driver.check(hPipe.get(), hRef.get(), (int)Qe, 1e-2f, 1.0f, "pipelined vs CPU");
        }
        driver.check(hPipe.get(), hBase.get(), (int)Qe, 1e-4f, 1e-4f, "pipelined vs baseline");

        // Performance
        auto bench = [&](CUfunction fn, int smem, const char *label) {
            for (int t = 0; t < 20; t++)
                CHECK_CU(cuLaunchKernel(fn, gq, heads, batch, 128, 1, 1, smem, nullptr, args, nullptr));
            CHECK_CU(cuCtxSynchronize());
            BenchTimer timer; timer.start();
            for (int t = 0; t < 200; t++)
                CHECK_CU(cuLaunchKernel(fn, gq, heads, batch, 128, 1, 1, smem, nullptr, args, nullptr));
            CHECK_CU(cuCtxSynchronize());
            float ms = timer.stop_ms() / 200.0f;
            double flops = 2.0 * sq * skv * D_HEAD * 2 * batch * heads;
            double gflops = flops / (ms * 1e-3) / 1e9;
            printf("  %-12s %.3f ms  -> %6.0f GFLOPS\n", label, ms, gflops);
            return ms;
        };

        float ms_base = bench(fn_base, smem_base, "Baseline:");
        float ms_pipe = bench(fn_pipe, smem_pipe, "Pipelined:");
        printf("  Speedup: %.2fx\n\n", ms_base / ms_pipe);
    }
    return 0;
}
