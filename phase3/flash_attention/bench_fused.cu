/*
 * bench_fused.cu — Benchmark: flash_attn_fused (BSHD) vs flash_attn_br16 (BHSD)
 *
 * Eliminates transpose kernels by accepting [B,S,H,D] layout directly.
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o flash_fused.sm_86.cubin flash_attn_fused.cu
 *   nvcc --cubin -arch=sm_86 -O2 -o flash_br16.sm_86.cubin flash_attn_br16.cu
 *   nvcc -arch=sm_86 -O2 -o bench_fused bench_fused.cu -lcuda -I../../kernels/_common
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
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
    int seq  = (argc > 1) ? atoi(argv[1]) : 1024;
    int batch= (argc > 2) ? atoi(argv[2]) : 8;
    int heads= (argc > 3) ? atoi(argv[3]) : 8;
    const int d = 64, Br_block = 64;
    float scale = 1.0f / sqrtf((float)d);

    if (seq % Br_block != 0) {
        fprintf(stderr, "seq=%d must be divisible by %d\n", seq, Br_block);
        return 1;
    }
    int d_model = heads * d;

    printf("=== Flash Attention: BSHD fused vs BHSD br16 ===\n");
    printf("seq=%d d=%d d_model=%d batch=%d heads=%d\n\n", seq, d, d_model, batch, heads);

    BenchDriver driver;
    driver.init_context();

    size_t ne  = (size_t)seq * d;
    size_t nb  = ne * sizeof(float);
    size_t nbh = ne * sizeof(__half);

    // --- Single-head correctness ---
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
    driver.copy_h2d(dQh, hQh, nbh);
    driver.copy_h2d(dKh, hKh, nbh);
    driver.copy_h2d(dVh, hVh, nbh);

    CUfunction fn_br16  = driver.load_kernel("flash_br16.sm_86.cubin",  "flash_attn_br16");
    CUfunction fn_fused = driver.load_kernel("flash_fused.sm_86.cubin", "flash_attn_fused");

    size_t smem = 2 * Br_block * d * sizeof(__half) + Br_block * Br_block * sizeof(float) + Br_block * d * sizeof(float);
    CHECK_CU(cuFuncSetAttribute(fn_br16,  CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem));
    CHECK_CU(cuFuncSetAttribute(fn_fused, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem));
    printf("Shared memory: %zu bytes (%.1f KB)\n\n", smem, smem / 1024.0f);

    int n1 = 1;
    void *a[] = { &dQh.ptr, &dKh.ptr, &dVh.ptr, &dO.ptr, &seq, &n1, &scale };

    printf("Correctness (single-head):\n");
    CHECK_CU(cuMemsetD32((CUdeviceptr)dO.ptr, 0, ne));
    CHECK_CU(cuLaunchKernel(fn_br16, seq / Br_block, 1, 1, 128, 1, 1, (unsigned)smem, nullptr, a, nullptr));
    CHECK_CU(cuCtxSynchronize());
    driver.copy_d2h(hOut, dO, nb);
    driver.check(hOut.get(), hRef.get(), (int)ne, 1e-2f, 1.0f, "br16  (BHSD)");

    CHECK_CU(cuMemsetD32((CUdeviceptr)dO.ptr, 0, ne));
    CHECK_CU(cuLaunchKernel(fn_fused, seq / Br_block, 1, 1, 128, 1, 1, (unsigned)smem, nullptr, a, nullptr));
    CHECK_CU(cuCtxSynchronize());
    driver.copy_d2h(hOut, dO, nb);
    driver.check(hOut.get(), hRef.get(), (int)ne, 1e-2f, 1.0f, "fused (BSHD)");
    printf("\n");

    // --- Multi-head correctness (fused, BSHD) ---
    {
        int tb = 1, th = 2, ts = 256;
        int td = th * d;
        size_t tne = (size_t)tb * ts * td;

        auto h_Qf = driver.host_alloc<float>(tne);
        auto h_Kf = driver.host_alloc<float>(tne);
        auto h_Vf = driver.host_alloc<float>(tne);
        fill_random(h_Qf.get(), tne, 30);
        fill_random(h_Kf.get(), tne, 31);
        fill_random(h_Vf.get(), tne, 32);

        auto h_Qh = driver.host_alloc<__half>(tne);
        auto h_Kh = driver.host_alloc<__half>(tne);
        auto h_Vh = driver.host_alloc<__half>(tne);
        fp32_to_fp16(h_Qf.get(), h_Qh.get(), tne);
        fp32_to_fp16(h_Kf.get(), h_Kh.get(), tne);
        fp32_to_fp16(h_Vf.get(), h_Vh.get(), tne);

        auto d_Q = driver.device_alloc<__half>(tne);
        auto d_K = driver.device_alloc<__half>(tne);
        auto d_V = driver.device_alloc<__half>(tne);
        auto d_O = driver.device_alloc<float>(tne);
        driver.copy_h2d(d_Q, h_Qh, tne * sizeof(__half));
        driver.copy_h2d(d_K, h_Kh, tne * sizeof(__half));
        driver.copy_h2d(d_V, h_Vh, tne * sizeof(__half));

        void *args[] = { &d_Q.ptr, &d_K.ptr, &d_V.ptr, &d_O.ptr, &ts, &th, &scale };
        CHECK_CU(cuMemsetD32((CUdeviceptr)d_O.ptr, 0, tne));
        CHECK_CU(cuLaunchKernel(fn_fused, ts / Br_block, th, tb, 128, 1, 1, (unsigned)smem, nullptr, args, nullptr));
        CHECK_CU(cuCtxSynchronize());

        auto h_O = driver.host_alloc<float>(tne);
        driver.copy_d2h(h_O, d_O, tne * sizeof(float));

        auto h_qs = driver.host_alloc<float>(ts * d);
        auto h_ks = driver.host_alloc<float>(ts * d);
        auto h_vs = driver.host_alloc<float>(ts * d);
        auto h_os = driver.host_alloc<float>(ts * d);
        auto h_ref = driver.host_alloc<float>(ts * d);
        auto sbuf  = driver.host_alloc<float>(ts);

        printf("Multi-head correctness (batch=%d, heads=%d, seq=%d):\n", tb, th, ts);
        for (int h = 0; h < th; h++) {
            for (int s = 0; s < ts; s++) {
                for (int dd = 0; dd < d; dd++) {
                    size_t idx = (size_t)s * td + h * d + dd;
                    h_qs[s * d + dd] = __half2float(h_Qh[idx]);
                    h_ks[s * d + dd] = __half2float(h_Kh[idx]);
                    h_vs[s * d + dd] = __half2float(h_Vh[idx]);
                    h_os[s * d + dd] = h_O[idx];
                }
            }
            cpu_attention(h_qs.get(), h_ks.get(), h_vs.get(), h_ref.get(), sbuf.get(), ts, d, scale);
            auto r = check_fp32(h_os.get(), h_ref.get(), ts * d, 1e-2f, 1e-0f);
            char label[64]; snprintf(label, sizeof(label), "  head %d", h);
            print_check_result(label, r);
        }
        printf("\n");
    }

    // --- Performance ---
    printf("Performance (batch=%d, heads=%d, seq=%d):\n\n", batch, heads, seq);

    size_t tot_bhsd = (size_t)batch * heads * seq * d;
    size_t tot_bshd = (size_t)batch * seq * d_model;

    auto dQH_bhsd = driver.device_alloc<__half>(tot_bhsd);
    auto dKH_bhsd = driver.device_alloc<__half>(tot_bhsd);
    auto dVH_bhsd = driver.device_alloc<__half>(tot_bhsd);
    auto dO_bhsd  = driver.device_alloc<float>(tot_bhsd);
    auto dQH_bshd = driver.device_alloc<__half>(tot_bshd);
    auto dKH_bshd = driver.device_alloc<__half>(tot_bshd);
    auto dVH_bshd = driver.device_alloc<__half>(tot_bshd);
    auto dO_bshd  = driver.device_alloc<float>(tot_bshd);

    CHECK_CU(cuMemsetD16((CUdeviceptr)dQH_bhsd.ptr, 0x3800, tot_bhsd));
    CHECK_CU(cuMemsetD16((CUdeviceptr)dKH_bhsd.ptr, 0x3800, tot_bhsd));
    CHECK_CU(cuMemsetD16((CUdeviceptr)dVH_bhsd.ptr, 0x3800, tot_bhsd));
    CHECK_CU(cuMemsetD16((CUdeviceptr)dQH_bshd.ptr, 0x3800, tot_bshd));
    CHECK_CU(cuMemsetD16((CUdeviceptr)dKH_bshd.ptr, 0x3800, tot_bshd));
    CHECK_CU(cuMemsetD16((CUdeviceptr)dVH_bshd.ptr, 0x3800, tot_bshd));

    double total_flops = (double)batch * heads * seq
                       * ((double)seq * d * 2.0 + (double)seq * 5.0 + (double)seq * d * 2.0);
    int grid_x = seq / Br_block;

    auto bench = [&](CUfunction fn, CUdeviceptr qp, CUdeviceptr kp, CUdeviceptr vp,
                     CUdeviceptr op, const char *label) {
        void *args[] = { &qp, &kp, &vp, &op, &seq, &heads, &scale };
        for (int i = 0; i < 5; i++)
            CHECK_CU(cuLaunchKernel(fn, grid_x, heads, batch, 128, 1, 1, (unsigned)smem, nullptr, args, nullptr));
        CHECK_CU(cuCtxSynchronize());
        BenchTimer timer; timer.start();
        for (int i = 0; i < 50; i++)
            CHECK_CU(cuLaunchKernel(fn, grid_x, heads, batch, 128, 1, 1, (unsigned)smem, nullptr, args, nullptr));
        CHECK_CU(cuCtxSynchronize());
        float ms = timer.stop_ms() / 50.0f;
        double gflops = total_flops / (ms / 1000.0) / 1e9;
        printf("  %-42s %7.3f ms  %8.0f GFLOPS\n", label, ms, gflops);
        return ms;
    };

    float ms_br16  = bench(fn_br16,  (CUdeviceptr)dQH_bhsd.ptr, (CUdeviceptr)dKH_bhsd.ptr,
                           (CUdeviceptr)dVH_bhsd.ptr, (CUdeviceptr)dO_bhsd.ptr, "flash_attn_br16  (BHSD)");
    float ms_fused = bench(fn_fused, (CUdeviceptr)dQH_bshd.ptr, (CUdeviceptr)dKH_bshd.ptr,
                           (CUdeviceptr)dVH_bshd.ptr, (CUdeviceptr)dO_bshd.ptr, "flash_attn_fused (BSHD)");
    printf("\n  Fused vs br16: %+.1f%% (%+.3f ms)\n",
           100.0 * (ms_fused - ms_br16) / ms_br16, ms_fused - ms_br16);
    printf("  Pipeline savings: eliminates 4 transpose kernel launches.\n");

    return 0;
}
