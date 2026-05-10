/*
 * bench_split_q.cu — Split-Q Flash Attention benchmark (BenchDriver)
 *
 * Tests flash_attn_split_q + reduce vs flash_attn_br16 baseline.
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o flash_br16.sm_86.cubin flash_attn_br16.cu
 *   nvcc --cubin -arch=sm_86 -O2 -o flash_split_q.sm_86.cubin flash_attn_split_q.cu
 *   nvcc -arch=sm_86 -O2 -o bench_split_q bench_split_q.cu -lcuda -I../../kernels/_common
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <cuda_fp16.h>

#include "../../kernels/_common/bench_driver.h"

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
    int seq   = (argc > 1) ? atoi(argv[1]) : 1024;
    int batch = (argc > 2) ? atoi(argv[2]) : 8;
    int heads = (argc > 3) ? atoi(argv[3]) : 8;
    int force = (argc > 4) ? atoi(argv[4]) : 0;
    const int d = 64, Br = 64, Bc = 64;
    float scale = 1.0f / sqrtf((float)d);
    int num_kv = seq / Bc;

    if (seq % Br != 0) { fprintf(stderr, "seq must divide %d\n", Br); return 1; }

    printf("=== Split-Q Flash Attention vs br16 ===\n");
    printf("seq=%d d=%d batch=%d heads=%d kv_tiles=%d\n\n", seq, d, batch, heads, num_kv);

    BenchDriver driver;
    driver.init_context();

    CUfunction fn_br16   = driver.load_kernel("flash_br16.sm_86.cubin", "flash_attn_br16");

    // Manual module load for split-q (need 2 kernels from 1 module)
    CUmodule mod_split;
    CUfunction fn_split, fn_reduce;
    if (cuModuleLoad(&mod_split, "flash_split_q.sm_86.cubin") != CUDA_SUCCESS) {
        fprintf(stderr, "Cannot load flash_split_q.sm_86.cubin\n"); return 1;
    }
    CHECK_CU(cuModuleGetFunction(&fn_split, mod_split, "flash_attn_split_q"));
    CHECK_CU(cuModuleGetFunction(&fn_reduce, mod_split, "flash_attn_split_q_reduce"));

    size_t smem = 2 * Bc * d * sizeof(__half) + Br * Bc * sizeof(float) + Br * d * sizeof(float);
    CHECK_CU(cuFuncSetAttribute(fn_br16,  CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem));
    CHECK_CU(cuFuncSetAttribute(fn_split, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem));
    printf("Shared memory: %zu bytes (%.1f KB)\n\n", smem, smem / 1024.0f);

    // --- Correctness (single head) ---
    {
        size_t ne = (size_t)seq * d;

        auto hQ  = driver.host_alloc<float>(ne);
        auto hK  = driver.host_alloc<float>(ne);
        auto hV  = driver.host_alloc<float>(ne);
        auto hRef= driver.host_alloc<float>(ne);
        auto hOut= driver.host_alloc<float>(ne);
        auto sBuf= driver.host_alloc<float>(seq);

        fill_random(hQ.get(), ne, 30);
        fill_random(hK.get(), ne, 31);
        fill_random(hV.get(), ne, 32);
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

        int one = 1;
        void *a[] = { &dQh.ptr, &dKh.ptr, &dVh.ptr, &dO.ptr, &seq, &one, &scale };

        printf("Correctness:\n");
        CHECK_CU(cuMemsetD32((CUdeviceptr)dO.ptr, 0, ne));
        CHECK_CU(cuLaunchKernel(fn_br16, seq / Br, 1, 1, 128, 1, 1, (unsigned)smem, nullptr, a, nullptr));
        CHECK_CU(cuCtxSynchronize());
        driver.copy_d2h(hOut, dO, ne * sizeof(float));
        driver.check(hOut.get(), hRef.get(), (int)ne, 1e-2f, 1.0f, "  br16 baseline");

        int t[] = {2, 4, num_kv};
        for (int ti = 0; ti < 3; ti++) {
            int ns = (t[ti] > num_kv) ? num_kv : t[ti];
            size_t pm = (size_t)ns * seq;
            size_t po = pm * d;

            auto dPm = driver.device_alloc<float>(pm);
            auto dPl = driver.device_alloc<float>(pm);
            auto dPo = driver.device_alloc<float>(po);

            void *sa[] = { &dQh.ptr, &dKh.ptr, &dVh.ptr, &dPo.ptr, &dPm.ptr, &dPl.ptr, &seq, &one, &ns, &scale };
            void *ra[] = { &dPo.ptr, &dPm.ptr, &dPl.ptr, &dO.ptr, &seq, &one, &ns };

            CHECK_CU(cuLaunchKernel(fn_split, ns, 1, 1, 128, 1, 1, (unsigned)smem, nullptr, sa, nullptr));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemsetD32((CUdeviceptr)dO.ptr, 0, ne));
            CHECK_CU(cuLaunchKernel(fn_reduce, seq / Br, 1, 1, 128, 1, 1, 0, nullptr, ra, nullptr));
            CHECK_CU(cuCtxSynchronize());
            driver.copy_d2h(hOut, dO, ne * sizeof(float));

            char label[64]; snprintf(label, sizeof(label), "  split-Q (splits=%d)", ns);
            driver.check(hOut.get(), hRef.get(), (int)ne, 1e-2f, 1.0f, label);
        }
        printf("\n");
    }

    // --- Performance ---
    printf("Performance (batch=%d, heads=%d, seq=%d):\n\n", batch, heads, seq);

    size_t tot = (size_t)batch * heads * seq * d;
    auto dQm = driver.device_alloc<__half>(tot);
    auto dKm = driver.device_alloc<__half>(tot);
    auto dVm = driver.device_alloc<__half>(tot);
    auto dOm = driver.device_alloc<float>(tot);
    CHECK_CU(cuMemsetD16((CUdeviceptr)dQm.ptr, 0x3800, tot));
    CHECK_CU(cuMemsetD16((CUdeviceptr)dKm.ptr, 0x3800, tot));
    CHECK_CU(cuMemsetD16((CUdeviceptr)dVm.ptr, 0x3800, tot));

    double flops = (double)batch * heads * seq * ((double)seq * d * 2 + (double)seq * 5 + (double)seq * d * 2);

    // br16
    void *a_br[] = { &dQm.ptr, &dKm.ptr, &dVm.ptr, &dOm.ptr, &seq, &heads, &scale };
    for (int i = 0; i < 5; i++)
        CHECK_CU(cuLaunchKernel(fn_br16, seq / Br, heads, batch, 128, 1, 1, (unsigned)smem, nullptr, a_br, nullptr));
    CHECK_CU(cuCtxSynchronize());
    BenchTimer t1; t1.start();
    for (int i = 0; i < 50; i++)
        CHECK_CU(cuLaunchKernel(fn_br16, seq / Br, heads, batch, 128, 1, 1, (unsigned)smem, nullptr, a_br, nullptr));
    CHECK_CU(cuCtxSynchronize());
    float ms_br = t1.stop_ms() / 50.0f;
    double gflops_br = flops / (ms_br / 1000.0) / 1e9;
    printf("  %-45s %7.3f ms  %7.1f GFLOPS\n", "br16 baseline", ms_br, gflops_br);

    // split-Q sweep
    int sweep[4]; int nsweep = 0;
    if (force > 0) { sweep[0] = force; nsweep = 1; }
    else {
        sweep[nsweep++] = 2;
        if (num_kv >= 4) sweep[nsweep++] = 4;
        if (num_kv >= 8) sweep[nsweep++] = 8;
        if (num_kv > 8)  sweep[nsweep++] = num_kv;
    }

    for (int si = 0; si < nsweep; si++) {
        int ns = sweep[si];
        if (ns > num_kv) ns = num_kv;

        size_t pm = (size_t)ns * batch * heads * seq;
        size_t po = pm * d;
        size_t pb = 2 * pm * sizeof(float) + po * sizeof(float);

        size_t free_mem = 0, total_mem = 0;
        cuMemGetInfo(&free_mem, &total_mem);
        if (pb > free_mem * 0.8) {
            printf("  split-Q (splits=%d) — skipped (%.0f MB > %.0f MB free)\n",
                   ns, pb / 1e6, free_mem / 1e6);
            continue;
        }

        auto dPm = driver.device_alloc<float>(pm);
        auto dPl = driver.device_alloc<float>(pm);
        auto dPo = driver.device_alloc<float>(po);

        void *sa[] = { &dQm.ptr, &dKm.ptr, &dVm.ptr, &dPo.ptr, &dPm.ptr, &dPl.ptr, &seq, &heads, &ns, &scale };
        void *ra[] = { &dPo.ptr, &dPm.ptr, &dPl.ptr, &dOm.ptr, &seq, &heads, &ns };

        for (int i = 0; i < 5; i++) {
            CHECK_CU(cuLaunchKernel(fn_split, ns, heads, batch, 128, 1, 1, (unsigned)smem, nullptr, sa, nullptr));
            CHECK_CU(cuLaunchKernel(fn_reduce, seq / Br, heads, batch, 128, 1, 1, 0, nullptr, ra, nullptr));
        }
        CHECK_CU(cuCtxSynchronize());
        BenchTimer t2; t2.start();
        for (int i = 0; i < 50; i++) {
            CHECK_CU(cuLaunchKernel(fn_split, ns, heads, batch, 128, 1, 1, (unsigned)smem, nullptr, sa, nullptr));
            CHECK_CU(cuLaunchKernel(fn_reduce, seq / Br, heads, batch, 128, 1, 1, 0, nullptr, ra, nullptr));
        }
        CHECK_CU(cuCtxSynchronize());
        float ms_sp = t2.stop_ms() / 50.0f;
        double gflops_sp = flops / (ms_sp / 1000.0) / 1e9;

        char label[64]; snprintf(label, sizeof(label), "split-Q (splits=%d, %.0f MB)", ns, pb / 1e6);
        printf("  %-45s %7.3f ms  %7.1f GFLOPS  %.2fx\n", label, ms_sp, gflops_sp, ms_br / ms_sp);
    }

    cuModuleUnload(mod_split);
    return 0;
}
