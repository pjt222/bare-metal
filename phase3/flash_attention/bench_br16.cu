/*
 * bench_br16.cu — Flash Attention Br=16 HMMA benchmark (BenchDriver refactor)
 *
 * Tests flash_attn_br16 (FP16 HMMA) vs flash_attn_4warp (FP32 scalar).
 *
 * Build:
 *   nvcc -arch=sm_86 -O2 -o bench_br16 bench_br16.cu -lcuda -I../../phase2/common
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>
#include <cuda_fp16.h>

#include "../../phase2/common/bench_driver.h"

static void cpu_attention(
    const float *Q, const float *K, const float *V, float *O,
    float *score_buf, int seq, int d, float scale
) {
    for (int q = 0; q < seq; q++) {
        float row_max = -3.402823466e+38f;
        for (int k = 0; k < seq; k++) {
            float dot = 0.0f;
            for (int i = 0; i < d; i++) dot += Q[q*d+i] * K[k*d+i];
            score_buf[k] = dot * scale;
            row_max = fmaxf(row_max, score_buf[k]);
        }
        float exp_sum = 0.0f;
        for (int k = 0; k < seq; k++) {
            score_buf[k] = expf(score_buf[k] - row_max);
            exp_sum += score_buf[k];
        }
        for (int i = 0; i < d; i++) O[q*d+i] = 0.0f;
        float rcp = 1.0f / exp_sum;
        for (int k = 0; k < seq; k++) {
            float w = score_buf[k] * rcp;
            for (int i = 0; i < d; i++) O[q*d+i] += w * V[k*d+i];
        }
    }
}

static void fp32_to_fp16(const float *src, __half *dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = __float2half(src[i]);
}

int main(int argc, char **argv) {
    int seq  = (argc > 1) ? atoi(argv[1]) : 1024;
    int batch= (argc > 2) ? atoi(argv[2]) : 8;
    int heads= (argc > 3) ? atoi(argv[3]) : 8;
    const int d = 64, Br_block = 64, Bc = 64, num_warps = 4;
    float scale = 1.0f / sqrtf((float)d);

    if (seq % Br_block != 0) {
        fprintf(stderr, "seq must be divisible by %d\n", Br_block);
        return 1;
    }

    printf("=== Flash Attention Br=16 HMMA Benchmark (BenchDriver refactor) ===\n");
    printf("seq=%d d=%d batch=%d heads=%d\n\n", seq, d, batch, heads);

    BenchDriver driver;
    driver.init_context();

    size_t ne = (size_t)seq * d;
    size_t nb = ne * sizeof(float);

    // Single-head correctness
    auto hQ  = driver.host_alloc<float>(ne);
    auto hK  = driver.host_alloc<float>(ne);
    auto hV  = driver.host_alloc<float>(ne);
    auto hRef= driver.host_alloc<float>(ne);
    auto hOut= driver.host_alloc<float>(ne);
    auto sBuf= driver.host_alloc<float>(seq);

    fill_random(hQ.get(), ne, 20);
    fill_random(hK.get(), ne, 21);
    fill_random(hV.get(), ne, 22);
    cpu_attention(hQ.get(), hK.get(), hV.get(), hRef.get(), sBuf.get(), seq, d, scale);

    auto hQh = driver.host_alloc<__half>(ne);
    auto hKh = driver.host_alloc<__half>(ne);
    auto hVh = driver.host_alloc<__half>(ne);
    fp32_to_fp16(hQ.get(), hQh.get(), ne);
    fp32_to_fp16(hK.get(), hKh.get(), ne);
    fp32_to_fp16(hV.get(), hVh.get(), ne);

    auto dQf = driver.device_alloc<float>(ne);
    auto dKf = driver.device_alloc<float>(ne);
    auto dVf = driver.device_alloc<float>(ne);
    auto dO  = driver.device_alloc<float>(ne);
    auto dQh = driver.device_alloc<__half>(ne);
    auto dKh = driver.device_alloc<__half>(ne);
    auto dVh = driver.device_alloc<__half>(ne);

    driver.copy_h2d(dQf, hQ, nb);
    driver.copy_h2d(dKf, hK, nb);
    driver.copy_h2d(dVf, hV, nb);
    driver.copy_h2d(dQh, hQh, ne * sizeof(__half));
    driver.copy_h2d(dKh, hKh, ne * sizeof(__half));
    driver.copy_h2d(dVh, hVh, ne * sizeof(__half));

    CUfunction fn_4w = driver.load_kernel("flash_wmma.sm_86.cubin", "flash_attn_4warp");
    CUfunction fn_br = driver.load_kernel("flash_br16.sm_86.cubin", "flash_attn_br16");

    size_t smem_br = 2 * Bc * d * sizeof(__half)
                   + Br_block * Bc * sizeof(float)
                   + Br_block * d * sizeof(float);
    CHECK_CU(cuFuncSetAttribute(fn_br, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem_br));
    printf("br16 smem: %zu bytes (%.1f KB)\n\n", smem_br, smem_br / 1024.0f);

    int n1 = 1;
    void *a4w[] = { &dQf.ptr, &dKf.ptr, &dVf.ptr, &dO.ptr, &seq, &n1, &scale };
    void *abr[] = { &dQh.ptr, &dKh.ptr, &dVh.ptr, &dO.ptr, &seq, &n1, &scale };

    CHECK_CU(cuMemsetD32((CUdeviceptr)dO.ptr, 0, ne));
    CHECK_CU(cuLaunchKernel(fn_4w, seq / num_warps, 1, 1, 128, 1, 1, 0, nullptr, a4w, nullptr));
    CHECK_CU(cuCtxSynchronize());
    driver.copy_d2h(hOut, dO, nb);
    driver.check(hOut.get(), hRef.get(), (int)ne, 1e-3f, 1e-1f, "flash_attn_4warp (FP32)");

    CHECK_CU(cuMemsetD32((CUdeviceptr)dO.ptr, 0, ne));
    CHECK_CU(cuLaunchKernel(fn_br, seq / Br_block, 1, 1, 128, 1, 1, (unsigned)smem_br, nullptr, abr, nullptr));
    CHECK_CU(cuCtxSynchronize());
    driver.copy_d2h(hOut, dO, nb);
    driver.check(hOut.get(), hRef.get(), (int)ne, 1e-2f, 1.0f, "flash_attn_br16 (FP16 HMMA)");

    // =====================================================================
    // Performance: multi-head
    // =====================================================================
    size_t tot = (size_t)batch * heads * ne;

    auto dQF = driver.device_alloc<float>(tot);
    auto dKF = driver.device_alloc<float>(tot);
    auto dVF = driver.device_alloc<float>(tot);
    auto dOF = driver.device_alloc<float>(tot);
    auto dQH = driver.device_alloc<__half>(tot);
    auto dKH = driver.device_alloc<__half>(tot);
    auto dVH = driver.device_alloc<__half>(tot);

    CHECK_CU(cuMemsetD32((CUdeviceptr)dQF.ptr, 0x3f000000, tot));
    CHECK_CU(cuMemsetD32((CUdeviceptr)dKF.ptr, 0x3f000000, tot));
    CHECK_CU(cuMemsetD32((CUdeviceptr)dVF.ptr, 0x3f000000, tot));
    CHECK_CU(cuMemsetD16((CUdeviceptr)dQH.ptr, 0x3800, tot));
    CHECK_CU(cuMemsetD16((CUdeviceptr)dKH.ptr, 0x3800, tot));
    CHECK_CU(cuMemsetD16((CUdeviceptr)dVH.ptr, 0x3800, tot));

    void *a4wm[] = { &dQF.ptr, &dKF.ptr, &dVF.ptr, &dOF.ptr, &seq, &heads, &scale };
    void *abrm[] = { &dQH.ptr, &dKH.ptr, &dVH.ptr, &dOF.ptr, &seq, &heads, &scale };

    auto bench = [&](CUfunction fn, int gx, int smem, void **args, const char *label) {
        for (int i = 0; i < 5; i++)
            CHECK_CU(cuLaunchKernel(fn, gx, heads, batch, 128, 1, 1, (unsigned)smem, nullptr, args, nullptr));
        CHECK_CU(cuCtxSynchronize());
        BenchTimer timer; timer.start();
        for (int i = 0; i < 50; i++)
            CHECK_CU(cuLaunchKernel(fn, gx, heads, batch, 128, 1, 1, (unsigned)smem, nullptr, args, nullptr));
        CHECK_CU(cuCtxSynchronize());
        float ms = timer.stop_ms() / 50.0f;
        double bw_elem = (double)tot * (1 + 2.0 * (seq / 64) + 1);
        double bw = bw_elem * sizeof(float) / 1e9 / (ms / 1000.0);
        double ideal = 4.0 * tot * sizeof(float) / 1e9 / (ms / 1000.0);
        printf("  %-40s %7.3f ms  %6.1f GB/s  (ideal: %5.1f GB/s)\n", label, ms, bw, ideal);
    };

    printf("\nPerformance (batch=%d, heads=%d):\n\n", batch, heads);
    bench(fn_4w, seq / num_warps, 0,        a4wm, "flash_attn_4warp (FP32, BKV=64)");
    bench(fn_br, seq / Br_block,  smem_br, abrm, "flash_attn_br16  (FP16, HMMA)  ");

    return 0;
}
