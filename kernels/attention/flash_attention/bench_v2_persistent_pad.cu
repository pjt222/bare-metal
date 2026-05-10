/*
 * bench_v2_persistent_pad.cu — A/B test of v2_persistent vs v2_persistent_pad.
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o flash_v2_persistent.sm_86.cubin     flash_attn_v2_persistent.cu
 *   nvcc --cubin -arch=sm_86 -O2 -o flash_v2_persistent_pad.sm_86.cubin flash_attn_v2_persistent_pad.cu
 *   nvcc -arch=sm_86 -O2 -o bench_v2_persistent_pad bench_v2_persistent_pad.cu \
 *        -lcuda -I../../kernels/_common
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <cuda_fp16.h>

#include "../../_common/bench_driver.h"

static void cpu_attention(const float *Q, const float *K, const float *V,
                          float *O, float *score_buf,
                          int seq, int d, float scale) {
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
    int seq   = (argc > 1) ? atoi(argv[1]) : 1024;
    int batch = (argc > 2) ? atoi(argv[2]) : 8;
    int heads = (argc > 3) ? atoi(argv[3]) : 8;
    const int d = 64, Br_block = 64, Bc = 64;
    float scale = 1.0f / sqrtf((float)d);

    if (seq % Br_block != 0) {
        fprintf(stderr, "seq=%d must be divisible by %d\n", seq, Br_block);
        return 1;
    }

    printf("=== FA v2 persistent: unpadded vs +8 padded ===\n");
    printf("seq=%d d=%d batch=%d heads=%d\n\n", seq, d, batch, heads);

    BenchDriver driver;
    driver.init_context();

    size_t ne = (size_t)seq * d;
    size_t nb = ne * sizeof(float);

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

    size_t tot = (size_t)batch * heads * ne;
    auto dQm = driver.device_alloc<__half>(tot);
    auto dKm = driver.device_alloc<__half>(tot);
    auto dVm = driver.device_alloc<__half>(tot);
    auto dOm = driver.device_alloc<float>(tot);
    CHECK_CU(cuMemsetD16((CUdeviceptr)dQm.ptr, 0x3800, tot));
    CHECK_CU(cuMemsetD16((CUdeviceptr)dKm.ptr, 0x3800, tot));
    CHECK_CU(cuMemsetD16((CUdeviceptr)dVm.ptr, 0x3800, tot));

    int grid_x = seq / Br_block;
    int total_tiles = grid_x * heads * batch;
    int persistent_blocks = total_tiles < 96 ? total_tiles : 96;
    double total_flops = (double)batch * heads * seq
                       * ((double)seq * d * 2.0 + (double)seq * 5.0 + (double)seq * d * 2.0);

    printf("%-36s %-10s %-7s %-8s %-9s %-9s\n",
           "variant", "smem_KB", "regs", "blocks", "ms", "GFLOPS");
    printf("------------------------------------------------------------------------------------------\n");

    auto run_variant = [&](const char *cubin, const char *kname,
                            const char *label, size_t smem) -> double {
        CUfunction fn = driver.load_kernel(cubin, kname);
        CHECK_CU(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem));
        int regs = 0, blocks = 0;
        cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn);
        cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fn, 128, smem);

        CUdeviceptr d_counter;
        cuMemAlloc(&d_counter, sizeof(int));

        // correctness (single-head)
        int t_tile_one = grid_x;
        int t_qs_one   = grid_x;
        int batch_one = 1, heads_one = 1;
        cuMemsetD32(d_counter, 0, 1);
        void *args1[] = { &dQh.ptr, &dKh.ptr, &dVh.ptr, &dO.ptr,
                          &seq, &heads_one, &batch_one, &scale,
                          &t_tile_one, &t_qs_one, &d_counter };
        CHECK_CU(cuMemsetD32((CUdeviceptr)dO.ptr, 0, ne));
        int p_blocks_one = grid_x < 96 ? grid_x : 96;
        CHECK_CU(cuLaunchKernel(fn, p_blocks_one, 1, 1, 128, 1, 1, (unsigned)smem, nullptr, args1, nullptr));
        CHECK_CU(cuCtxSynchronize());
        driver.copy_d2h(hOut, dO, nb);
        CheckResult cr = check_fp32(hOut.get(), hRef.get(), (int)ne, 1e-2f, 1.0f, false);
        bool ok = (cr.num_errors == 0);

        // performance
        int q_tiles_per_seq = grid_x;
        void *argsm[] = { &dQm.ptr, &dKm.ptr, &dVm.ptr, &dOm.ptr,
                          &seq, &heads, &batch, &scale,
                          &total_tiles, &q_tiles_per_seq, &d_counter };
        for (int w = 0; w < 5; w++) {
            cuMemsetD32(d_counter, 0, 1);
            CHECK_CU(cuLaunchKernel(fn, persistent_blocks, 1, 1, 128, 1, 1, (unsigned)smem, nullptr, argsm, nullptr));
        }
        CHECK_CU(cuCtxSynchronize());
        BenchTimer timer; timer.start();
        const int iters = 50;
        for (int j = 0; j < iters; j++) {
            cuMemsetD32(d_counter, 0, 1);
            CHECK_CU(cuLaunchKernel(fn, persistent_blocks, 1, 1, 128, 1, 1, (unsigned)smem, nullptr, argsm, nullptr));
        }
        CHECK_CU(cuCtxSynchronize());
        float ms = timer.stop_ms() / iters;
        double gflops = total_flops / (ms / 1000.0) / 1e9;
        printf("%-36s %-10.2f %-7d %-8d %-9.3f %-9.0f %s\n",
               label, smem/1024.0f, regs, blocks, ms, gflops, ok ? "✓" : "✗");
        cuMemFree(d_counter);
        return gflops;
    };

    size_t smem_unpad = 2 * (size_t)Bc * d * sizeof(__half)
                      + (size_t)Br_block * Bc * sizeof(__half);
    double gf_unpad = run_variant("flash_v2_persistent.sm_86.cubin",
                                   "flash_attn_v2_persistent",
                                   "v2 persistent (unpadded, 24 KB)", smem_unpad);

    const int D_STRIDE = 72;
    size_t smem_pad = 2 * (size_t)Bc * D_STRIDE * sizeof(__half)
                    + (size_t)Br_block * D_STRIDE * sizeof(__half);
    double gf_pad   = run_variant("flash_v2_persistent_pad.sm_86.cubin",
                                   "flash_attn_v2_persistent_pad",
                                   "v2 persistent_pad (+8, 27 KB)", smem_pad);

    printf("\nSpeedup pad vs unpad: %.3fx\n", gf_pad / gf_unpad);
    return 0;
}
