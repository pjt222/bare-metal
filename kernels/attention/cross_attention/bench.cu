/*
 * bench.cu — Cross-Attention benchmark (BenchDriver refactor)
 *
 * Tests cross_attn_br16 across typical Stable Diffusion configurations.
 *
 * Build:
 *   nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../../kernels/_common
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>
#include <cuda_fp16.h>

#include "../../_common/bench_driver.h"

static void cpu_cross_attn(
    const float *Q, const float *K, const float *V, float *O,
    float *score_buf, int seq_q, int seq_kv, int d_head, float scale
) {
    for (int q = 0; q < seq_q; q++) {
        float row_max = -3.402823466e+38f;
        for (int k = 0; k < seq_kv; k++) {
            float dot = 0.0f;
            for (int d = 0; d < d_head; d++) dot += Q[q*d_head+d] * K[k*d_head+d];
            score_buf[k] = dot * scale;
            row_max = fmaxf(row_max, score_buf[k]);
        }
        float sum = 0.0f;
        for (int k = 0; k < seq_kv; k++) {
            score_buf[k] = expf(score_buf[k] - row_max);
            sum += score_buf[k];
        }
        float rcp = 1.0f / sum;
        for (int d = 0; d < d_head; d++) O[q*d_head+d] = 0.0f;
        for (int k = 0; k < seq_kv; k++) {
            float w = score_buf[k] * rcp;
            for (int d = 0; d < d_head; d++) O[q*d_head+d] += w * V[k*d_head+d];
        }
    }
}

static void fp32_to_fp16(const float *src, __half *dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = __float2half(src[i]);
}

int main(int argc, char **argv) {
    int seq_q     = (argc > 1) ? atoi(argv[1]) : 256;
    int seq_kv    = (argc > 2) ? atoi(argv[2]) : 77;
    int num_heads = (argc > 3) ? atoi(argv[3]) : 8;
    const int batch = 1;
    const int d_head = 64;
    const int Br = 64;
    float scale = 1.0f / sqrtf((float)d_head);

    if (seq_q % Br != 0) {
        seq_q = ((seq_q + Br - 1) / Br) * Br;
        printf("Note: seq_q padded to %d for grid alignment\n", seq_q);
    }

    printf("=== Cross-Attention Benchmark (BenchDriver refactor) ===\n");
    printf("seq_q=%d seq_kv=%d heads=%d d_head=%d\n\n", seq_q, seq_kv, num_heads, d_head);

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

    auto dQ = driver.device_alloc<__half>(qe);
    auto dK = driver.device_alloc<__half>(kve);
    auto dV = driver.device_alloc<__half>(kve);
    auto dO = driver.device_alloc<float>(qe);

    driver.copy_h2d(dQ, hQh, qe * sizeof(__half));
    driver.copy_h2d(dK, hKh, kve * sizeof(__half));
    driver.copy_h2d(dV, hVh, kve * sizeof(__half));

    CUfunction fn = driver.load_kernel("cross_attn.sm_86.cubin", "cross_attn_br16");

    size_t smem = 2 * Br * d_head * sizeof(__half)
                + Br * Br * sizeof(float)
                + Br * d_head * sizeof(float);
    CHECK_CU(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem));
    printf("Shared memory per block: %zu bytes (%.1f KB)\n\n", smem, smem / 1024.0f);

    int nh1 = 1;
    void *args_s[] = { &dQ.ptr, &dK.ptr, &dV.ptr, &dO.ptr,
                       &seq_q, &seq_kv, &nh1, &scale };
    CHECK_CU(cuMemsetD32((CUdeviceptr)dO.ptr, 0, qe));
    CHECK_CU(cuLaunchKernel(fn, seq_q / Br, 1, 1, 128, 1, 1,
                            (unsigned)smem, nullptr, args_s, nullptr));
    CHECK_CU(cuCtxSynchronize());
    driver.copy_d2h(hOut, dO, qe * sizeof(float));
    driver.check(hOut.get(), hRef.get(), (int)qe, 1e-2f, 1.0f,
                 "cross_attn_br16 (FP16 HMMA)");

    // Performance: multi-head
    size_t tot_q  = (size_t)batch * num_heads * qe;
    size_t tot_kv = (size_t)batch * num_heads * kve;

    auto dQm = driver.device_alloc<__half>(tot_q);
    auto dKm = driver.device_alloc<__half>(tot_kv);
    auto dVm = driver.device_alloc<__half>(tot_kv);
    auto dOm = driver.device_alloc<float>(tot_q);

    CHECK_CU(cuMemsetD16((CUdeviceptr)dQm.ptr, 0x3800, tot_q));
    CHECK_CU(cuMemsetD16((CUdeviceptr)dKm.ptr, 0x3800, tot_kv));
    CHECK_CU(cuMemsetD16((CUdeviceptr)dVm.ptr, 0x3800, tot_kv));
    CHECK_CU(cuMemsetD32((CUdeviceptr)dOm.ptr, 0, tot_q));

    void *args_m[] = { &dQm.ptr, &dKm.ptr, &dVm.ptr, &dOm.ptr,
                       &seq_q, &seq_kv, &num_heads, &scale };

    dim3 grid(seq_q / Br, num_heads, batch);
    dim3 block(128, 1, 1);

    printf("\nPerformance (batch=%d, heads=%d):\n", batch, num_heads);
    float ms = driver.benchmark_kernel(fn, grid, block, (unsigned)smem, args_m, 5, 100);

    double flops = 4.0 * batch * num_heads * seq_q * seq_kv * d_head;
    double gflops = flops / 1e9 / (ms / 1000.0);
    double ideal_bytes = sizeof(__half) * (tot_q + 2*tot_kv) + sizeof(float) * tot_q;
    double ideal_bw = ideal_bytes / 1e9 / (ms / 1000.0);

    printf("  %-45s %7.3f ms\n", "cross_attn_br16", ms);
    printf("  FLOPS: %.2f GFLOPS (%.1f%% of 174 TFLOPS peak)\n",
           gflops, gflops / 174000.0 * 100.0);
    printf("  Ideal BW: %.1f GB/s (peak: 608 GB/s)\n", ideal_bw);

    printf("\nKey asymmetry vs self-attn: KV iters=%d (vs %d for self)\n",
           (seq_kv + 63) / 64, (seq_q + 63) / 64);
    return 0;
}
