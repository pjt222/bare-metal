/*
 * bench.cu — Flash Attention benchmark (BenchDriver refactor)
 *
 * Tests correctness (single-head) and throughput (multi-head).
 *
 * Build:
 *   nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../../kernels/_common
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>

#include "../../kernels/_common/bench_driver.h"

static void cpu_attention(
    const float *Q, const float *K, const float *V, float *O,
    float *score_buf, int seq_len, int d_head, float scale
) {
    for (int q = 0; q < seq_len; q++) {
        float row_max = -3.402823466e+38f;
        for (int k = 0; k < seq_len; k++) {
            float dot = 0.0f;
            for (int d = 0; d < d_head; d++) dot += Q[q*d_head+d] * K[k*d_head+d];
            score_buf[k] = dot * scale;
            row_max = fmaxf(row_max, score_buf[k]);
        }
        float exp_sum = 0.0f;
        for (int k = 0; k < seq_len; k++) {
            score_buf[k] = expf(score_buf[k] - row_max);
            exp_sum += score_buf[k];
        }
        for (int d = 0; d < d_head; d++) O[q*d_head+d] = 0.0f;
        float rcp = 1.0f / exp_sum;
        for (int k = 0; k < seq_len; k++) {
            float w = score_buf[k] * rcp;
            for (int d = 0; d < d_head; d++) O[q*d_head+d] += w * V[k*d_head+d];
        }
    }
}

int main(int argc, char **argv) {
    int seq_len    = (argc > 1) ? atoi(argv[1]) : 512;
    int batch_size = (argc > 2) ? atoi(argv[2]) : 1;
    int num_heads  = (argc > 3) ? atoi(argv[3]) : 1;
    const int d_head = 64, block_kv = 32;
    float scale = 1.0f / sqrtf((float)d_head);

    if (seq_len % block_kv != 0) {
        fprintf(stderr, "seq_len must be multiple of %d\n", block_kv);
        return 1;
    }

    printf("=== Flash Attention Benchmark (BenchDriver refactor) ===\n");
    printf("seq=%d d=%d batch=%d heads=%d\n\n", seq_len, d_head, batch_size, num_heads);

    BenchDriver driver;
    driver.init_context();

    size_t sh = (size_t)seq_len * d_head;
    size_t sb = sh * sizeof(float);

    // Single-head correctness
    auto hQ  = driver.host_alloc<float>(sh);
    auto hK  = driver.host_alloc<float>(sh);
    auto hV  = driver.host_alloc<float>(sh);
    auto hRef= driver.host_alloc<float>(sh);
    auto hOut= driver.host_alloc<float>(sh);
    auto sBuf= driver.host_alloc<float>(seq_len);

    fill_random(hQ.get(), sh, 42);
    fill_random(hK.get(), sh, 43);
    fill_random(hV.get(), sh, 44);
    cpu_attention(hQ.get(), hK.get(), hV.get(), hRef.get(), sBuf.get(), seq_len, d_head, scale);

    auto dQ = driver.device_alloc<float>(sh);
    auto dK = driver.device_alloc<float>(sh);
    auto dV = driver.device_alloc<float>(sh);
    auto dO = driver.device_alloc<float>(sh);
    driver.copy_h2d(dQ, hQ, sb);
    driver.copy_h2d(dK, hK, sb);
    driver.copy_h2d(dV, hV, sb);

    CUfunction fn_single = driver.load_kernel("flash_attn.sm_86.cubin", "flash_attn_1warp");
    CUfunction fn_multi  = driver.load_kernel("flash_attn.sm_86.cubin", "flash_attn_multihead");

    void *args1[] = { &dQ.ptr, &dK.ptr, &dV.ptr, &dO.ptr, &seq_len, &scale };
    CHECK_CU(cuMemsetD32((CUdeviceptr)dO.ptr, 0, sh));
    CHECK_CU(cuLaunchKernel(fn_single, seq_len, 1, 1, 32, 1, 1, 0, nullptr, args1, nullptr));
    CHECK_CU(cuCtxSynchronize());
    driver.copy_d2h(hOut, dO, sb);
    driver.check(hOut.get(), hRef.get(), (int)sh, 1e-3f, 1e-3f, "flash_attn_1warp");

    // Multi-head performance
    size_t tot = (size_t)batch_size * num_heads * sh;
    size_t tb  = tot * sizeof(float);

    auto dQm = driver.device_alloc<float>(tot);
    auto dKm = driver.device_alloc<float>(tot);
    auto dVm = driver.device_alloc<float>(tot);
    auto dOm = driver.device_alloc<float>(tot);

    auto h_multi = driver.host_alloc<float>(tot);
    fill_random(h_multi.get(), tot, 99);
    driver.copy_h2d(dQm, h_multi, tb);
    driver.copy_h2d(dKm, h_multi, tb);
    driver.copy_h2d(dVm, h_multi, tb);

    void *args_m[] = { &dQm.ptr, &dKm.ptr, &dVm.ptr, &dOm.ptr,
                       &seq_len, &num_heads, &scale };

    printf("\nPerformance (batch=%d, heads=%d, seq=%d):\n", batch_size, num_heads, seq_len);
    float ms = driver.benchmark_kernel(fn_multi,
        dim3(seq_len, num_heads, batch_size), dim3(32,1,1), 0, args_m, 5, 50);

    int kv_iters = seq_len / block_kv;
    double bytes_read  = (double)tb * (1.0 + 2.0 * kv_iters);
    double total_bytes = bytes_read + tb;  // + O write
    double bw = (total_bytes / 1e9) / (ms / 1000.0);
    double ideal_bw = (4.0 * tb / 1e9) / (ms / 1000.0);
    double flops = (double)batch_size * num_heads * seq_len * (
        (double)seq_len * d_head * 2 + (double)seq_len * 5 + (double)seq_len * d_head * 2);
    double gflops = flops / (ms / 1000.0) / 1e9;

    printf("  %-30s %7.3f ms  %7.1f GB/s  %7.1f GFLOPS\n",
           "flash_attn_multihead", ms, bw, gflops);
    printf("  (ideal BW if K/V cached: %.1f GB/s)\n", ideal_bw);

    return 0;
}
