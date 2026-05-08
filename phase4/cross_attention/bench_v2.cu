/*
 * bench_v2.cu — Cross-attention v2 (nosmem) vs baseline
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o cross_attn.sm_86.cubin cross_attn.cu
 *   nvcc --cubin -arch=sm_86 -O2 -o cross_attn_v2.sm_86.cubin cross_attn_v2.cu
 *   nvcc -arch=sm_86 -O2 -o bench_v2 bench_v2.cu -lcuda -I../../phase2/common
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>
#include <cuda_fp16.h>

#include "../../phase2/common/bench_driver.h"

static void cpu_cross_attn(
    const float *Q, const float *K, const float *V, float *O,
    float *score_buf, int seq_q, int seq_kv, int d, float scale
) {
    for (int q = 0; q < seq_q; q++) {
        float row_max = -3.402823466e+38f;
        for (int k = 0; k < seq_kv; k++) {
            float dot = 0.0f;
            for (int i = 0; i < d; i++) dot += Q[q * d + i] * K[k * d + i];
            score_buf[k] = dot * scale;
            row_max = fmaxf(row_max, score_buf[k]);
        }
        float exp_sum = 0.0f;
        for (int k = 0; k < seq_kv; k++) {
            score_buf[k] = expf(score_buf[k] - row_max);
            exp_sum += score_buf[k];
        }
        for (int i = 0; i < d; i++) O[q * d + i] = 0.0f;
        float rcp = 1.0f / exp_sum;
        for (int k = 0; k < seq_kv; k++) {
            float w = score_buf[k] * rcp;
            for (int i = 0; i < d; i++) O[q * d + i] += w * V[k * d + i];
        }
    }
}

static void fp32_to_fp16(const float *src, __half *dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = __float2half(src[i]);
}

struct V {
    const char *cubin, *symbol, *label;
    size_t smem;
};

int main(int argc, char **argv) {
    int seq_q     = (argc > 1) ? atoi(argv[1]) : 256;
    int seq_kv    = (argc > 2) ? atoi(argv[2]) : 77;
    int num_heads = (argc > 3) ? atoi(argv[3]) : 8;
    const int batch = 1;
    const int d_head = 64;
    const int Br = 64;
    const int Bc = 64;
    float scale = 1.0f / sqrtf((float)d_head);

    if (seq_q % Br != 0) seq_q = ((seq_q + Br - 1) / Br) * Br;

    printf("=== Cross-Attention v2 vs baseline ===\n");
    printf("seq_q=%d seq_kv=%d heads=%d d_head=%d batch=%d\n\n",
           seq_q, seq_kv, num_heads, d_head, batch);

    BenchDriver driver;
    driver.init_context();

    size_t qe  = (size_t)seq_q  * d_head;
    size_t kve = (size_t)seq_kv * d_head;

    auto hQf  = driver.host_alloc<float>(qe);
    auto hKf  = driver.host_alloc<float>(kve);
    auto hVf  = driver.host_alloc<float>(kve);
    auto hRef = driver.host_alloc<float>(qe);
    auto hOut = driver.host_alloc<float>(qe);
    auto sBuf = driver.host_alloc<float>(seq_kv);

    fill_random(hQf.get(), qe,  10);
    fill_random(hKf.get(), kve, 11);
    fill_random(hVf.get(), kve, 12);
    cpu_cross_attn(hQf.get(), hKf.get(), hVf.get(), hRef.get(),
                   sBuf.get(), seq_q, seq_kv, d_head, scale);

    auto hQh = driver.host_alloc<__half>(qe);
    auto hKh = driver.host_alloc<__half>(kve);
    auto hVh = driver.host_alloc<__half>(kve);
    fp32_to_fp16(hQf.get(), hQh.get(), qe);
    fp32_to_fp16(hKf.get(), hKh.get(), kve);
    fp32_to_fp16(hVf.get(), hVh.get(), kve);

    // Single-head buffers for correctness check
    auto dQ = driver.device_alloc<__half>(qe);
    auto dK = driver.device_alloc<__half>(kve);
    auto dV = driver.device_alloc<__half>(kve);
    auto dO = driver.device_alloc<float>(qe);
    driver.copy_h2d(dQ, hQh, qe * sizeof(__half));
    driver.copy_h2d(dK, hKh, kve * sizeof(__half));
    driver.copy_h2d(dV, hVh, kve * sizeof(__half));

    // Multi-head buffers for performance test
    size_t total_q  = (size_t)batch * num_heads * qe;
    size_t total_kv = (size_t)batch * num_heads * kve;
    auto dQm = driver.device_alloc<__half>(total_q);
    auto dKm = driver.device_alloc<__half>(total_kv);
    auto dVm = driver.device_alloc<__half>(total_kv);
    auto dOm = driver.device_alloc<float>(total_q);
    CHECK_CU(cuMemsetD16((CUdeviceptr)dQm.ptr, 0x3800, total_q));
    CHECK_CU(cuMemsetD16((CUdeviceptr)dKm.ptr, 0x3800, total_kv));
    CHECK_CU(cuMemsetD16((CUdeviceptr)dVm.ptr, 0x3800, total_kv));

    size_t smem_baseline = 2 * (size_t)Bc * d_head * sizeof(__half)
                         + (size_t)Br * Bc * sizeof(float)
                         + (size_t)Br * d_head * sizeof(float);
    size_t smem_v2 = 2 * (size_t)Bc * d_head * sizeof(__half)
                   + (size_t)Br * Bc * sizeof(__half);

    V variants[] = {
        { "cross_attn.sm_86.cubin",    "cross_attn_br16", "cross_attn baseline (48 KB)", smem_baseline },
        { "cross_attn_v2.sm_86.cubin", "cross_attn_v2",   "cross_attn_v2  (24 KB)     ", smem_v2 },
    };

    int n1 = 1;
    int grid_x = seq_q / Br;
    double total_flops = (double)batch * num_heads * seq_q
                       * ((double)seq_kv * d_head * 2.0
                        + (double)seq_kv * 5.0
                        + (double)seq_kv * d_head * 2.0);

    printf("%-32s %-9s %-7s %-8s %-9s %-9s %-7s\n",
           "variant", "smem_KB", "regs", "blocks", "ms", "GFLOPS", "spdup");
    printf("------------------------------------------------------------------------------------------\n");

    float ms_base = 0.0f;
    for (auto &v : variants) {
        CUfunction fn = driver.load_kernel(v.cubin, v.symbol);
        CHECK_CU(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)v.smem));
        int regs = 0, blocks = 0;
        cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn);
        cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fn, 128, v.smem);

        // correctness (single-head)
        void *args1[] = { &dQ.ptr, &dK.ptr, &dV.ptr, &dO.ptr,
                          &seq_q, &seq_kv, &n1, &scale };
        CHECK_CU(cuMemsetD32((CUdeviceptr)dO.ptr, 0, qe));
        CHECK_CU(cuLaunchKernel(fn, grid_x, 1, 1, 128, 1, 1, (unsigned)v.smem,
                                nullptr, args1, nullptr));
        CHECK_CU(cuCtxSynchronize());
        driver.copy_d2h(hOut, dO, qe * sizeof(float));
        CheckResult cr = check_fp32(hOut.get(), hRef.get(), (int)qe, 1e-2f, 1.0f, false);
        bool ok = (cr.num_errors == 0);

        // performance
        void *argsm[] = { &dQm.ptr, &dKm.ptr, &dVm.ptr, &dOm.ptr,
                          &seq_q, &seq_kv, &num_heads, &scale };
        for (int w = 0; w < 5; w++)
            CHECK_CU(cuLaunchKernel(fn, grid_x, num_heads, batch, 128, 1, 1,
                                    (unsigned)v.smem, nullptr, argsm, nullptr));
        CHECK_CU(cuCtxSynchronize());
        BenchTimer timer; timer.start();
        for (int j = 0; j < 100; j++)
            CHECK_CU(cuLaunchKernel(fn, grid_x, num_heads, batch, 128, 1, 1,
                                    (unsigned)v.smem, nullptr, argsm, nullptr));
        CHECK_CU(cuCtxSynchronize());
        float ms = timer.stop_ms() / 100.0f;
        double gflops = total_flops / (ms / 1000.0) / 1e9;
        if (ms_base == 0.0f) ms_base = ms;
        float spdup = ms_base / ms;
        printf("%-32s %-9.2f %-7d %-8d %-9.4f %-9.0f %.2fx   %s\n",
               v.label, v.smem/1024.0f, regs, blocks, ms, gflops, spdup,
               ok ? "✓" : "✗");
    }
    return 0;
}
