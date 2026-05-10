/*
 * bench_br16_regpv.cu — Register-PV vs baseline Flash Attention benchmark (BenchDriver)
 *
 * Tests flash_attn_br16_regpv (32 KB smem, reg-resident PV) vs
 * flash_attn_br16 baseline (48 KB smem, smem PV).
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o flash_br16.sm_86.cubin flash_attn_br16.cu
 *   nvcc --cubin -arch=sm_86 -O2 -o flash_br16_regpv.sm_86.cubin flash_attn_br16_regpv.cu
 *   nvcc -arch=sm_86 -O2 -o bench_br16_regpv bench_br16_regpv.cu -lcuda -I../../kernels/_common
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>
#include <cuda_fp16.h>

#include "../../_common/bench_driver.h"

static void cpu_attention(
    const float *Q, const float *K, const float *V, float *O,
    float *score_buf, int seq, int d, float scale
) {
    for (int q = 0; q < seq; q++) {
        float row_max = -3.402823466e+38f;
        for (int k = 0; k < seq; k++) {
            float dot = 0.0f;
            for (int i = 0; i < d; i++) dot += Q[q * d + i] * K[k * d + i];
            score_buf[k] = dot * scale;
            row_max = fmaxf(row_max, score_buf[k]);
        }
        float exp_sum = 0.0f;
        for (int k = 0; k < seq; k++) {
            score_buf[k] = expf(score_buf[k] - row_max);
            exp_sum += score_buf[k];
        }
        for (int i = 0; i < d; i++) O[q * d + i] = 0.0f;
        float rcp = 1.0f / exp_sum;
        for (int k = 0; k < seq; k++) {
            float w = score_buf[k] * rcp;
            for (int i = 0; i < d; i++) O[q * d + i] += w * V[k * d + i];
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
    const int d = 64, Br_block = 64, Bc = 64;
    float scale = 1.0f / sqrtf((float)d);

    if (seq % Br_block != 0) {
        fprintf(stderr, "seq=%d must be divisible by %d\n", seq, Br_block);
        return 1;
    }

    printf("=== Flash Attention: RegPV (32 KB) vs Baseline (48 KB) ===\n");
    printf("seq=%d d=%d batch=%d heads=%d\n\n", seq, d, batch, heads);

    BenchDriver driver;
    driver.init_context();

    size_t ne = (size_t)seq * d;
    size_t nb = ne * sizeof(float);

    // --- Correctness (single head) ---
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

    auto dQh = driver.device_alloc<__half>(ne);
    auto dKh = driver.device_alloc<__half>(ne);
    auto dVh = driver.device_alloc<__half>(ne);
    auto dO  = driver.device_alloc<float>(ne);

    driver.copy_h2d(dQh, hQh, ne * sizeof(__half));
    driver.copy_h2d(dKh, hKh, ne * sizeof(__half));
    driver.copy_h2d(dVh, hVh, ne * sizeof(__half));

    CUfunction fn_base  = driver.load_kernel("flash_br16.sm_86.cubin",       "flash_attn_br16");
    CUfunction fn_regpv = driver.load_kernel("flash_br16_regpv.sm_86.cubin", "flash_attn_br16_regpv");

    size_t smem_base  = 2 * Bc * d * sizeof(__half) + Br_block * Bc * sizeof(float) + Br_block * d * sizeof(float);
    size_t smem_regpv = 2 * Bc * d * sizeof(__half) + Br_block * Bc * sizeof(float);
    CHECK_CU(cuFuncSetAttribute(fn_base,  CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem_base));
    CHECK_CU(cuFuncSetAttribute(fn_regpv, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem_regpv));

    printf("Baseline smem: %zu bytes (%.1f KB)\n", smem_base,  smem_base  / 1024.0f);
    printf("RegPV    smem: %zu bytes (%.1f KB)\n\n", smem_regpv, smem_regpv / 1024.0f);

    int regs_base = 0, regs_regpv = 0;
    cuFuncGetAttribute(&regs_base,  CU_FUNC_ATTRIBUTE_NUM_REGS, fn_base);
    cuFuncGetAttribute(&regs_regpv, CU_FUNC_ATTRIBUTE_NUM_REGS, fn_regpv);
    printf("Registers/thread: baseline=%d  regpv=%d\n", regs_base, regs_regpv);

    int max_base = 0, max_regpv = 0;
    cuOccupancyMaxActiveBlocksPerMultiprocessor(&max_base,  fn_base,  128, smem_base);
    cuOccupancyMaxActiveBlocksPerMultiprocessor(&max_regpv, fn_regpv, 128, smem_regpv);
    printf("Blocks/SM:        baseline=%d  regpv=%d\n", max_base, max_regpv);
    printf("Warps/SM:         baseline=%d  regpv=%d\n\n", max_base * 4, max_regpv * 4);

    int n1 = 1;
    void *a[] = { &dQh.ptr, &dKh.ptr, &dVh.ptr, &dO.ptr, &seq, &n1, &scale };

    printf("Correctness:\n");
    CHECK_CU(cuMemsetD32((CUdeviceptr)dO.ptr, 0, ne));
    CHECK_CU(cuLaunchKernel(fn_base, seq / Br_block, 1, 1, 128, 1, 1, (unsigned)smem_base, nullptr, a, nullptr));
    CHECK_CU(cuCtxSynchronize());
    driver.copy_d2h(hOut, dO, nb);
    driver.check(hOut.get(), hRef.get(), (int)ne, 1e-2f, 1.0f, "flash_attn_br16       (baseline, 48 KB)");

    CHECK_CU(cuMemsetD32((CUdeviceptr)dO.ptr, 0, ne));
    CHECK_CU(cuLaunchKernel(fn_regpv, seq / Br_block, 1, 1, 128, 1, 1, (unsigned)smem_regpv, nullptr, a, nullptr));
    CHECK_CU(cuCtxSynchronize());
    driver.copy_d2h(hOut, dO, nb);
    driver.check(hOut.get(), hRef.get(), (int)ne, 1e-2f, 1.0f, "flash_attn_br16_regpv (reg PV,   32 KB)");
    printf("\n");

    // --- Performance (multi-head) ---
    printf("Performance (batch=%d, heads=%d):\n\n", batch, heads);
    size_t tot = (size_t)batch * heads * ne;

    auto dQm = driver.device_alloc<__half>(tot);
    auto dKm = driver.device_alloc<__half>(tot);
    auto dVm = driver.device_alloc<__half>(tot);
    auto dOm = driver.device_alloc<float>(tot);

    CHECK_CU(cuMemsetD16((CUdeviceptr)dQm.ptr, 0x3800, tot));
    CHECK_CU(cuMemsetD16((CUdeviceptr)dKm.ptr, 0x3800, tot));
    CHECK_CU(cuMemsetD16((CUdeviceptr)dVm.ptr, 0x3800, tot));

    double total_flops = (double)batch * heads * seq
                       * ((double)seq * d * 2.0 + (double)seq * 5.0 + (double)seq * d * 2.0);
    int grid_x = seq / Br_block;

    auto bench = [&](CUfunction fn, size_t smem, const char *label) {
        void *args[] = { &dQm.ptr, &dKm.ptr, &dVm.ptr, &dOm.ptr, &seq, &heads, &scale };
        for (int i = 0; i < 5; i++)
            CHECK_CU(cuLaunchKernel(fn, grid_x, heads, batch, 128, 1, 1, (unsigned)smem, nullptr, args, nullptr));
        CHECK_CU(cuCtxSynchronize());
        BenchTimer timer; timer.start();
        for (int i = 0; i < 50; i++)
            CHECK_CU(cuLaunchKernel(fn, grid_x, heads, batch, 128, 1, 1, (unsigned)smem, nullptr, args, nullptr));
        CHECK_CU(cuCtxSynchronize());
        float ms = timer.stop_ms() / 50.0f;
        double gflops = total_flops / (ms / 1000.0) / 1e9;
        printf("  %-45s %7.3f ms  %8.0f GFLOPS\n", label, ms, gflops);
        return ms;
    };

    float ms_base  = bench(fn_base,  smem_base,  "flash_attn_br16       (baseline, 48 KB)");
    float ms_regpv = bench(fn_regpv, smem_regpv, "flash_attn_br16_regpv (reg PV,   32 KB)");
    printf("\n  Speedup: %.2fx (%.1f%% %s)\n",
           ms_base / ms_regpv,
           fabsf(ms_base - ms_regpv) / ms_base * 100.0f,
           ms_regpv < ms_base ? "faster" : "slower");

    printf("\nSASS:\n");
    printf("  cuobjdump -sass flash_br16_regpv.sm_86.cubin | grep -c HMMA\n");
    printf("  cuobjdump -sass flash_br16_regpv.sm_86.cubin | grep -c 'LDS\\|STS'\n");

    return 0;
}
