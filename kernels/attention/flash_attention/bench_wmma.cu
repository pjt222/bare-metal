/*
 * bench.cu — Flash Attention: 4-warp vs 1-warp benchmark (BenchDriver refactor)
 *
 * Build:
 *   nvcc -arch=sm_86 -O2 -o bench_wmma bench_wmma.cu -lcuda -I../../kernels/_common
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>

#include "../../_common/bench_driver.h"

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
    int seq_len    = (argc > 1) ? atoi(argv[1]) : 1024;
    int batch_size = (argc > 2) ? atoi(argv[2]) : 8;
    int num_heads  = (argc > 3) ? atoi(argv[3]) : 8;
    const int d_head = 64, block_kv_v1 = 32, block_kv_v2 = 64, num_warps = 4;
    float scale = 1.0f / sqrtf((float)d_head);

    if (seq_len % block_kv_v1 != 0 || seq_len % num_warps != 0) {
        fprintf(stderr, "seq_len must be divisible by %d and %d\n", block_kv_v1, num_warps);
        return 1;
    }

    printf("=== Flash Attention Benchmark (BenchDriver refactor) ===\n");
    printf("seq=%d d=%d batch=%d heads=%d\n\n", seq_len, d_head, batch_size, num_heads);

    BenchDriver driver;
    driver.init_context();

    size_t sh = (size_t)seq_len * d_head;
    size_t sb = sh * sizeof(float);

    auto hQ  = driver.host_alloc<float>(sh);
    auto hK  = driver.host_alloc<float>(sh);
    auto hV  = driver.host_alloc<float>(sh);
    auto hRef= driver.host_alloc<float>(sh);
    auto hOut= driver.host_alloc<float>(sh);
    auto sBuf= driver.host_alloc<float>(seq_len);

    fill_random(hQ.get(), sh, 10);
    fill_random(hK.get(), sh, 11);
    fill_random(hV.get(), sh, 12);
    cpu_attention(hQ.get(), hK.get(), hV.get(), hRef.get(), sBuf.get(), seq_len, d_head, scale);

    auto dQ = driver.device_alloc<float>(sh);
    auto dK = driver.device_alloc<float>(sh);
    auto dV = driver.device_alloc<float>(sh);
    auto dO = driver.device_alloc<float>(sh);
    driver.copy_h2d(dQ, hQ, sb);
    driver.copy_h2d(dK, hK, sb);
    driver.copy_h2d(dV, hV, sb);

    CUfunction fn_v1 = driver.load_kernel("flash_attn.sm_86.cubin", "flash_attn_multihead");
    CUfunction fn_v2 = driver.load_kernel("flash_attn_wmma.sm_86.cubin", "flash_attn_4warp");

    int n1 = 1;
    void *args1[] = { &dQ.ptr, &dK.ptr, &dV.ptr, &dO.ptr, &seq_len, &n1, &scale };
    void *args2[] = { &dQ.ptr, &dK.ptr, &dV.ptr, &dO.ptr, &seq_len, &n1, &scale };

    CHECK_CU(cuMemsetD32((CUdeviceptr)dO.ptr, 0, sh));
    CHECK_CU(cuLaunchKernel(fn_v1, seq_len, 1, 1, 32, 1, 1, 0, nullptr, args1, nullptr));
    CHECK_CU(cuCtxSynchronize());
    driver.copy_d2h(hOut, dO, sb);
    driver.check(hOut.get(), hRef.get(), (int)sh, 1e-3f, 1e-1f,
                 "flash_attn_multihead (v1 scalar)");

    CHECK_CU(cuMemsetD32((CUdeviceptr)dO.ptr, 0, sh));
    CHECK_CU(cuLaunchKernel(fn_v2, seq_len / num_warps, 1, 1, 128, 1, 1, 0, nullptr, args2, nullptr));
    CHECK_CU(cuCtxSynchronize());
    driver.copy_d2h(hOut, dO, sb);
    driver.check(hOut.get(), hRef.get(), (int)sh, 1e-3f, 1e-1f,
                 "flash_attn_4warp (v2 4-warp)");

    // =====================================================================
    // Performance: multi-head/batch
    // =====================================================================
    size_t tot = (size_t)batch_size * num_heads * sh;
    size_t tb  = tot * sizeof(float);

    auto dQm = driver.device_alloc<float>(tot);
    auto dKm = driver.device_alloc<float>(tot);
    auto dVm = driver.device_alloc<float>(tot);
    auto dOm = driver.device_alloc<float>(tot);
    CHECK_CU(cuMemsetD32((CUdeviceptr)dQm.ptr, 0x3f000000, tot));
    CHECK_CU(cuMemsetD32((CUdeviceptr)dKm.ptr, 0x3f000000, tot));
    CHECK_CU(cuMemsetD32((CUdeviceptr)dVm.ptr, 0x3f000000, tot));

    void *args_m[] = { &dQm.ptr, &dKm.ptr, &dVm.ptr, &dOm.ptr, &seq_len, &num_heads, &scale };

    printf("\nPerformance (batch=%d, heads=%d):\n\n", batch_size, num_heads);

    int kv_v1 = seq_len / block_kv_v1;
    int kv_v2 = seq_len / block_kv_v2;

    auto bench = [&](CUfunction fn, int gx, int bx, const char *label, int kv) {
        for (int i = 0; i < 5; i++)
            CHECK_CU(cuLaunchKernel(fn, gx, num_heads, batch_size, bx, 1, 1, 0, nullptr, args_m, nullptr));
        CHECK_CU(cuCtxSynchronize());
        BenchTimer timer; timer.start();
        for (int i = 0; i < 50; i++)
            CHECK_CU(cuLaunchKernel(fn, gx, num_heads, batch_size, bx, 1, 1, 0, nullptr, args_m, nullptr));
        CHECK_CU(cuCtxSynchronize());
        float ms = timer.stop_ms() / 50.0f;
        double bytes = (1.0 + 2.0 * kv + 1.0) * tb;
        double bw = (bytes / 1e9) / (ms / 1000.0);
        double ideal = (4.0 * tb / 1e9) / (ms / 1000.0);
        printf("  %-40s %7.3f ms   %6.1f GB/s  (ideal: %5.1f GB/s)\n",
               label, ms, bw, ideal);
    };

    bench(fn_v1, seq_len,           32, "flash_attn_multihead (1-warp, BKV=32)", kv_v1);
    bench(fn_v2, seq_len / num_warps, 128, "flash_attn_4warp     (4-warp, BKV=64)", kv_v2);

    return 0;
}
