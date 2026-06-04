/*
 * bench_persistent.cu — Persistent grid Flash Attention benchmark (BenchDriver)
 *
 * Tests SM utilization improvement at small batch sizes.
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o flash_attn_br16.sm_86.cubin flash_attn_br16.cu
 *   nvcc --cubin -arch=sm_86 -O2 -o flash_attn_persistent.sm_86.cubin flash_attn_persistent.cu
 *   nvcc -arch=sm_86 -O2 -o bench_persistent bench_persistent.cu -lcuda -I../../kernels/_common
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
    int batch = (argc > 1) ? atoi(argv[1]) : 1;
    int heads = (argc > 2) ? atoi(argv[2]) : 8;
    const int d = 64, Br_block = 64, Bc = 64;
    float scale = 1.0f / sqrtf((float)d);

    printf("=== Persistent Grid Flash Attention Benchmark ===\n");
    printf("batch=%d heads=%d d=%d\n\n", batch, heads, d);

    BenchDriver driver;
    driver.init_context();

    int num_sms = 0;
    CHECK_CU(cuDeviceGetAttribute(&num_sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, driver.device));
    printf("SMs: %d\n\n", num_sms);

    CUfunction fn_br16    = driver.load_kernel("flash_attn_br16.sm_86.cubin",      "flash_attn_br16");
    CUfunction fn_persist = driver.load_kernel("flash_attn_persistent.sm_86.cubin","flash_attn_persistent");

    size_t smem = 2 * Bc * d * sizeof(__half) + Br_block * Bc * sizeof(float) + Br_block * d * sizeof(float);
    CHECK_CU(cuFuncSetAttribute(fn_br16,    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem));
    CHECK_CU(cuFuncSetAttribute(fn_persist, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem));
    printf("Shared memory: %zu bytes (%.1f KB)\n", smem, smem / 1024.0f);

    int persistent_grid = num_sms * 2;
    printf("Persistent grid: %d blocks (%d SMs x 2)\n\n", persistent_grid, num_sms);

    auto dev_counter = driver.device_alloc<int>(1);

    // --- Correctness (single head, seq=256) ---
    {
        int seq = 256;
        size_t ne = (size_t)seq * d;

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

        int n1 = 1;
        printf("Correctness (seq=%d, single head):\n", seq);

        CHECK_CU(cuMemsetD32((CUdeviceptr)dO.ptr, 0, ne));
        void *a_br[] = { &dQh.ptr, &dKh.ptr, &dVh.ptr, &dO.ptr, &seq, &n1, &scale };
        CHECK_CU(cuLaunchKernel(fn_br16, seq / Br_block, 1, 1, 128, 1, 1, (unsigned)smem, nullptr, a_br, nullptr));
        CHECK_CU(cuCtxSynchronize());
        driver.copy_d2h(hOut, dO, ne * sizeof(float));
        driver.check(hOut.get(), hRef.get(), (int)ne, 1e-2f, 1.0f, "flash_attn_br16");

        int batch_1 = 1, total_tiles = seq / Br_block, q_tiles = seq / Br_block;
        CHECK_CU(cuMemsetD32((CUdeviceptr)dO.ptr, 0, ne));
        CHECK_CU(cuMemsetD32((CUdeviceptr)dev_counter.ptr, 0, 1));
        void *a_p[] = { &dQh.ptr, &dKh.ptr, &dVh.ptr, &dO.ptr, &seq, &n1, &batch_1, &scale,
                        &total_tiles, &q_tiles, &dev_counter.ptr };
        CHECK_CU(cuLaunchKernel(fn_persist, persistent_grid, 1, 1, 128, 1, 1, (unsigned)smem, nullptr, a_p, nullptr));
        CHECK_CU(cuCtxSynchronize());
        driver.copy_d2h(hOut, dO, ne * sizeof(float));
        driver.check(hOut.get(), hRef.get(), (int)ne, 1e-2f, 1.0f, "flash_attn_persistent");
        printf("\n");
    }

    // --- Performance sweep ---
    int seqs[] = { 256, 512, 1024 };
    printf("Performance (batch=%d, heads=%d):\n", batch, heads);
    printf("%-8s %-6s %-12s %-12s %-10s %-10s\n", "seq", "tiles", "br16(ms)", "persist(ms)", "speedup", "note");
    printf("------- ------ ---------- ----------- ---------- ----------\n");

    for (int seq : seqs) {
        int q_tiles = seq / Br_block;
        int total_tiles = q_tiles * heads * batch;
        size_t tot = (size_t)batch * heads * seq * d;

        auto dQm = driver.device_alloc<__half>(tot);
        auto dKm = driver.device_alloc<__half>(tot);
        auto dVm = driver.device_alloc<__half>(tot);
        auto dOm = driver.device_alloc<float>(tot);
        CHECK_CU(cuMemsetD16((CUdeviceptr)dQm.ptr, 0x3800, tot));
        CHECK_CU(cuMemsetD16((CUdeviceptr)dKm.ptr, 0x3800, tot));
        CHECK_CU(cuMemsetD16((CUdeviceptr)dVm.ptr, 0x3800, tot));

        // br16
        void *a_br[] = { &dQm.ptr, &dKm.ptr, &dVm.ptr, &dOm.ptr, &seq, &heads, &scale };
        for (int i = 0; i < 5; i++)
            CHECK_CU(cuLaunchKernel(fn_br16, q_tiles, heads, batch, 128, 1, 1, (unsigned)smem, nullptr, a_br, nullptr));
        CHECK_CU(cuCtxSynchronize());
        BenchTimer t1; t1.start();
        for (int i = 0; i < 50; i++)
            CHECK_CU(cuLaunchKernel(fn_br16, q_tiles, heads, batch, 128, 1, 1, (unsigned)smem, nullptr, a_br, nullptr));
        CHECK_CU(cuCtxSynchronize());
        float ms_br = t1.stop_ms() / 50.0f;

        // persistent
        void *a_p[] = { &dQm.ptr, &dKm.ptr, &dVm.ptr, &dOm.ptr, &seq, &heads, &batch,
                        &scale, &total_tiles, &q_tiles, &dev_counter.ptr };
        for (int i = 0; i < 5; i++) {
            CHECK_CU(cuMemsetD32((CUdeviceptr)dev_counter.ptr, 0, 1));
            CHECK_CU(cuLaunchKernel(fn_persist, persistent_grid, 1, 1, 128, 1, 1, (unsigned)smem, nullptr, a_p, nullptr));
        }
        CHECK_CU(cuCtxSynchronize());
        BenchTimer t2; t2.start();
        for (int i = 0; i < 50; i++) {
            CHECK_CU(cuMemsetD32((CUdeviceptr)dev_counter.ptr, 0, 1));
            CHECK_CU(cuLaunchKernel(fn_persist, persistent_grid, 1, 1, 128, 1, 1, (unsigned)smem, nullptr, a_p, nullptr));
        }
        CHECK_CU(cuCtxSynchronize());
        float ms_p = t2.stop_ms() / 50.0f;

        float util = 100.0f * total_tiles / (num_sms * 2);
        if (util > 100.0f) util = 100.0f;
        const char *note = "";
        if (total_tiles < num_sms)     note = "SMs idle";
        else if (total_tiles < num_sms * 2) note = "1 blk/SM";

        printf("%-8d %-6d %-12.3f %-12.3f %-10.4f %s\n", seq, total_tiles, ms_br, ms_p, ms_br / ms_p, note);
    }
    printf("\n");

    return 0;
}
