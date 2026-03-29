/*
 * bench_fused.cu — Benchmark: flash_attn_fused (FP16 [B,S,H,D] layout)
 *                  vs flash_attn_br16 (FP16 [B,H,S,D] layout)
 *
 * Both kernels take FP16 input and produce FP32 output. The only difference
 * is the memory layout: BSHD vs BHSD. The fused variant eliminates all
 * transpose kernels from the pipeline.
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o flash_fused.sm_86.cubin flash_attn_fused.cu
 *   nvcc --cubin -arch=sm_86 -O2 -o flash_br16.sm_86.cubin flash_attn_br16.cu
 *   nvcc -arch=sm_86 -O2 -o bench_fused bench_fused.cu -lcuda -I../../phase2/common
 *
 * Usage:
 *   ./bench_fused              # seq=1024, batch=8, heads=8
 *   ./bench_fused 2048 4 8
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>
#include <cuda_fp16.h>

#include "../../phase2/common/bench.h"
#include "../../phase2/common/check.h"

// CPU reference (single head, FP32 naive)
static void cpu_attention(
    const float *Qf, const float *Kf, const float *Vf, float *Of,
    float *sbuf, int seq, int d, float scale
) {
    for (int q = 0; q < seq; q++) {
        float row_max = -3.402823466e+38f;
        for (int k = 0; k < seq; k++) {
            float dot = 0.0f;
            for (int dd = 0; dd < d; dd++) dot += Qf[q*d+dd] * Kf[k*d+dd];
            sbuf[k] = dot * scale;
            row_max = fmaxf(row_max, sbuf[k]);
        }
        float sum = 0.0f;
        for (int k = 0; k < seq; k++) { sbuf[k] = expf(sbuf[k] - row_max); sum += sbuf[k]; }
        float rcp = 1.0f / sum;
        for (int dd = 0; dd < d; dd++) Of[q*d+dd] = 0.0f;
        for (int k = 0; k < seq; k++) {
            float w = sbuf[k] * rcp;
            for (int dd = 0; dd < d; dd++) Of[q*d+dd] += w * Vf[k*d+dd];
        }
    }
}

static void fp32_to_fp16_host(const float *src, __half *dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = __float2half(src[i]);
}

int main(int argc, char **argv) {
    int seq        = (argc > 1) ? atoi(argv[1]) : 1024;
    int batch      = (argc > 2) ? atoi(argv[2]) : 8;
    int num_heads  = (argc > 3) ? atoi(argv[3]) : 8;

    const int d        = 64;
    const int Br_block = 64;

    if (seq % Br_block != 0) {
        fprintf(stderr, "seq=%d must be divisible by %d\n", seq, Br_block);
        return 1;
    }

    float scale = 1.0f / sqrtf((float)d);
    int d_model = num_heads * d;

    printf("=== Flash Attention: BSHD layout (fused) vs BHSD layout (br16) ===\n");
    printf("seq=%d  d=%d  d_model=%d  batch=%d  heads=%d\n\n", seq, d, d_model, batch, num_heads);

    CHECK_CU(cuInit(0));
    CUdevice cu_dev; CHECK_CU(cuDeviceGet(&cu_dev, 0));
    char devname[256]; CHECK_CU(cuDeviceGetName(devname, sizeof(devname), cu_dev));
    printf("Device: %s\n\n", devname);
    CUcontext ctx; CHECK_CU(cuCtxCreate(&ctx, 0, cu_dev));

    CUmodule mod_br16, mod_fused;
    CUfunction fn_br16, fn_fused;

    if (cuModuleLoad(&mod_br16, "flash_br16.sm_86.cubin") != CUDA_SUCCESS) {
        fprintf(stderr, "Cannot load flash_br16.sm_86.cubin\n"); return 1;
    }
    if (cuModuleLoad(&mod_fused, "flash_fused.sm_86.cubin") != CUDA_SUCCESS) {
        fprintf(stderr, "Cannot load flash_fused.sm_86.cubin\n"); return 1;
    }
    CHECK_CU(cuModuleGetFunction(&fn_br16,  mod_br16,  "flash_attn_br16"));
    CHECK_CU(cuModuleGetFunction(&fn_fused, mod_fused, "flash_attn_fused"));

    // Both use 48 KB smem (Q loaded from global, not smem)
    size_t smem = 2 * Br_block * d * sizeof(short)
                + Br_block * Br_block * sizeof(float)
                + Br_block * d * sizeof(float);

    CHECK_CU(cuFuncSetAttribute(fn_br16,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem));
    CHECK_CU(cuFuncSetAttribute(fn_fused,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem));

    printf("Shared memory: %zu bytes (%.1f KB) — same for both\n\n", smem, smem/1024.0);

    // =========================================================
    // Correctness: single-head (both layouts collapse to [S,D])
    // =========================================================
    {
        size_t n_elems = (size_t)seq * d;
        size_t n_bytes = n_elems * sizeof(float);

        float *hQf  = (float*)malloc(n_bytes);
        float *hKf  = (float*)malloc(n_bytes);
        float *hVf  = (float*)malloc(n_bytes);
        float *hRef = (float*)malloc(n_bytes);
        float *hOut = (float*)malloc(n_bytes);
        float *sBuf = (float*)malloc(seq * sizeof(float));

        fill_random(hQf, n_elems, 20);
        fill_random(hKf, n_elems, 21);
        fill_random(hVf, n_elems, 22);

        printf("Computing CPU reference (single head, seq=%d)...\n", seq);
        cpu_attention(hQf, hKf, hVf, hRef, sBuf, seq, d, scale);
        printf("Done.\n\n");

        __half *hQh = (__half*)malloc(n_elems * sizeof(__half));
        __half *hKh = (__half*)malloc(n_elems * sizeof(__half));
        __half *hVh = (__half*)malloc(n_elems * sizeof(__half));
        fp32_to_fp16_host(hQf, hQh, n_elems);
        fp32_to_fp16_host(hKf, hKh, n_elems);
        fp32_to_fp16_host(hVf, hVh, n_elems);

        CUdeviceptr dQh, dKh, dVh, dO;
        CHECK_CU(cuMemAlloc(&dQh, n_elems * sizeof(__half)));
        CHECK_CU(cuMemAlloc(&dKh, n_elems * sizeof(__half)));
        CHECK_CU(cuMemAlloc(&dVh, n_elems * sizeof(__half)));
        CHECK_CU(cuMemAlloc(&dO,  n_bytes));
        CHECK_CU(cuMemcpyHtoD(dQh, hQh, n_elems * sizeof(__half)));
        CHECK_CU(cuMemcpyHtoD(dKh, hKh, n_elems * sizeof(__half)));
        CHECK_CU(cuMemcpyHtoD(dVh, hVh, n_elems * sizeof(__half)));

        int n1 = 1;
        printf("Correctness (vs CPU FP32):\n");

        // br16: [B,H,S,D] — single head, [S,D] same layout
        {
            void *args[] = { &dQh, &dKh, &dVh, &dO, &seq, &n1, &scale };
            CHECK_CU(cuMemsetD32(dO, 0, n_elems));
            CHECK_CU(cuLaunchKernel(fn_br16,
                seq / Br_block, 1, 1,  128, 1, 1,
                (unsigned)smem, NULL, args, NULL));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(hOut, dO, n_bytes));
            auto r = check_fp32(hOut, hRef, n_elems, 1e-2f, 1e-0f);
            print_check_result("br16  (BHSD)", r);
        }

        // fused: [B,S,H,D] — single head, [S,D] same layout
        {
            void *args[] = { &dQh, &dKh, &dVh, &dO, &seq, &n1, &scale };
            CHECK_CU(cuMemsetD32(dO, 0, n_elems));
            CHECK_CU(cuLaunchKernel(fn_fused,
                seq / Br_block, 1, 1,  128, 1, 1,
                (unsigned)smem, NULL, args, NULL));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(hOut, dO, n_bytes));
            auto r = check_fp32(hOut, hRef, n_elems, 1e-2f, 1e-0f);
            print_check_result("fused (BSHD)", r);
        }
        printf("\n");

        cuMemFree(dQh); cuMemFree(dKh); cuMemFree(dVh); cuMemFree(dO);
        free(hQf); free(hKf); free(hVf); free(hRef); free(hOut); free(sBuf);
        free(hQh); free(hKh); free(hVh);
    }

    // =========================================================
    // Multi-head correctness: fused [B,S,H,D] layout
    // =========================================================
    {
        int test_batch = 1, test_heads = 2, test_seq = 256;
        int test_dmodel = test_heads * d;
        size_t total_elems = (size_t)test_batch * test_seq * test_dmodel;

        // Generate FP32 data in [B,S,H,D] layout
        float *h_Qf = (float*)malloc(total_elems * sizeof(float));
        float *h_Kf = (float*)malloc(total_elems * sizeof(float));
        float *h_Vf = (float*)malloc(total_elems * sizeof(float));
        fill_random(h_Qf, total_elems, 30);
        fill_random(h_Kf, total_elems, 31);
        fill_random(h_Vf, total_elems, 32);

        // Convert to FP16 (same [B,S,H,D] flat layout)
        __half *h_Qh = (__half*)malloc(total_elems * sizeof(__half));
        __half *h_Kh = (__half*)malloc(total_elems * sizeof(__half));
        __half *h_Vh = (__half*)malloc(total_elems * sizeof(__half));
        fp32_to_fp16_host(h_Qf, h_Qh, total_elems);
        fp32_to_fp16_host(h_Kf, h_Kh, total_elems);
        fp32_to_fp16_host(h_Vf, h_Vh, total_elems);

        float *h_O = (float*)malloc(total_elems * sizeof(float));

        CUdeviceptr d_Q, d_K, d_V, d_O;
        CHECK_CU(cuMemAlloc(&d_Q, total_elems * sizeof(__half)));
        CHECK_CU(cuMemAlloc(&d_K, total_elems * sizeof(__half)));
        CHECK_CU(cuMemAlloc(&d_V, total_elems * sizeof(__half)));
        CHECK_CU(cuMemAlloc(&d_O, total_elems * sizeof(float)));
        CHECK_CU(cuMemcpyHtoD(d_Q, h_Qh, total_elems * sizeof(__half)));
        CHECK_CU(cuMemcpyHtoD(d_K, h_Kh, total_elems * sizeof(__half)));
        CHECK_CU(cuMemcpyHtoD(d_V, h_Vh, total_elems * sizeof(__half)));
        CHECK_CU(cuMemsetD32(d_O, 0, total_elems));

        void *args[] = { &d_Q, &d_K, &d_V, &d_O, &test_seq, &test_heads, &scale };
        CHECK_CU(cuLaunchKernel(fn_fused,
            test_seq / Br_block, test_heads, test_batch,
            128, 1, 1, (unsigned)smem, NULL, args, NULL));
        CHECK_CU(cuCtxSynchronize());
        CHECK_CU(cuMemcpyDtoH(h_O, d_O, total_elems * sizeof(float)));

        // Compare per-head vs CPU reference
        float *h_qs = (float*)malloc(test_seq * d * sizeof(float));
        float *h_ks = (float*)malloc(test_seq * d * sizeof(float));
        float *h_vs = (float*)malloc(test_seq * d * sizeof(float));
        float *h_os = (float*)malloc(test_seq * d * sizeof(float));
        float *h_ref = (float*)malloc(test_seq * d * sizeof(float));
        float *sbuf = (float*)malloc(test_seq * sizeof(float));

        printf("Multi-head correctness (batch=%d, heads=%d, seq=%d):\n",
               test_batch, test_heads, test_seq);
        for (int h = 0; h < test_heads; h++) {
            // Extract [S, D] slice for head h from [B, S, H, D]
            for (int s = 0; s < test_seq; s++) {
                for (int dd = 0; dd < d; dd++) {
                    size_t bshd_idx = (size_t)s * test_dmodel + h * d + dd;
                    // Use the FP16-converted values (via float cast) for fair comparison
                    h_qs[s * d + dd] = __half2float(h_Qh[bshd_idx]);
                    h_ks[s * d + dd] = __half2float(h_Kh[bshd_idx]);
                    h_vs[s * d + dd] = __half2float(h_Vh[bshd_idx]);
                    h_os[s * d + dd] = h_O[bshd_idx];
                }
            }
            cpu_attention(h_qs, h_ks, h_vs, h_ref, sbuf, test_seq, d, scale);
            auto r = check_fp32(h_os, h_ref, test_seq * d, 1e-2f, 1e-0f);
            char label[64];
            snprintf(label, sizeof(label), "  head %d", h);
            print_check_result(label, r);
        }
        printf("\n");

        cuMemFree(d_Q); cuMemFree(d_K); cuMemFree(d_V); cuMemFree(d_O);
        free(h_Qf); free(h_Kf); free(h_Vf); free(h_Qh); free(h_Kh); free(h_Vh);
        free(h_O); free(h_qs); free(h_ks); free(h_vs); free(h_os); free(h_ref); free(sbuf);
    }

    // =========================================================
    // Performance (multi-head, multi-batch)
    // =========================================================
    printf("Performance (batch=%d, heads=%d, seq=%d):\n\n", batch, num_heads, seq);

    size_t tot_bhsd = (size_t)batch * num_heads * seq * d;  // BHSD element count
    size_t tot_bshd = (size_t)batch * seq * d_model;        // BSHD element count (same total)

    // BHSD buffers for br16
    CUdeviceptr dQHm_bhsd, dKHm_bhsd, dVHm_bhsd, dO_bhsd;
    CHECK_CU(cuMemAlloc(&dQHm_bhsd, tot_bhsd * sizeof(__half)));
    CHECK_CU(cuMemAlloc(&dKHm_bhsd, tot_bhsd * sizeof(__half)));
    CHECK_CU(cuMemAlloc(&dVHm_bhsd, tot_bhsd * sizeof(__half)));
    CHECK_CU(cuMemAlloc(&dO_bhsd,   tot_bhsd * sizeof(float)));
    CHECK_CU(cuMemsetD16(dQHm_bhsd, 0x3800, tot_bhsd));
    CHECK_CU(cuMemsetD16(dKHm_bhsd, 0x3800, tot_bhsd));
    CHECK_CU(cuMemsetD16(dVHm_bhsd, 0x3800, tot_bhsd));

    // BSHD buffers for fused
    CUdeviceptr dQHm_bshd, dKHm_bshd, dVHm_bshd, dO_bshd;
    CHECK_CU(cuMemAlloc(&dQHm_bshd, tot_bshd * sizeof(__half)));
    CHECK_CU(cuMemAlloc(&dKHm_bshd, tot_bshd * sizeof(__half)));
    CHECK_CU(cuMemAlloc(&dVHm_bshd, tot_bshd * sizeof(__half)));
    CHECK_CU(cuMemAlloc(&dO_bshd,   tot_bshd * sizeof(float)));
    CHECK_CU(cuMemsetD16(dQHm_bshd, 0x3800, tot_bshd));
    CHECK_CU(cuMemsetD16(dKHm_bshd, 0x3800, tot_bshd));
    CHECK_CU(cuMemsetD16(dVHm_bshd, 0x3800, tot_bshd));

    int warmup = 5, bench_n = 50;

    auto run_bench = [&](CUfunction fn, size_t fn_smem, CUdeviceptr qp, CUdeviceptr kp,
                         CUdeviceptr vp, CUdeviceptr op, const char *label) {
        void *args[] = { &qp, &kp, &vp, &op, &seq, &num_heads, &scale };
        int grid_x = seq / Br_block;
        for (int i = 0; i < warmup; i++) {
            CHECK_CU(cuLaunchKernel(fn,
                grid_x, num_heads, batch, 128, 1, 1,
                (unsigned)fn_smem, NULL, args, NULL));
        }
        CHECK_CU(cuCtxSynchronize());

        BenchTimer timer;
        timer.start();
        for (int i = 0; i < bench_n; i++) {
            CHECK_CU(cuLaunchKernel(fn,
                grid_x, num_heads, batch, 128, 1, 1,
                (unsigned)fn_smem, NULL, args, NULL));
        }
        float avg_ms = timer.stop_ms() / bench_n;

        double flops = 2.0 * batch * num_heads * (2.0 * seq * seq * d);
        double gflops = flops / (avg_ms / 1000.0) / 1e9;
        printf("  %-42s %7.3f ms  %6.0f GFLOPS\n", label, avg_ms, gflops);
        return avg_ms;
    };

    float ms_br16 = run_bench(fn_br16, smem, dQHm_bhsd, dKHm_bhsd, dVHm_bhsd, dO_bhsd,
                              "flash_attn_br16  (FP16 [B,H,S,D])");
    float ms_fused = run_bench(fn_fused, smem, dQHm_bshd, dKHm_bshd, dVHm_bshd, dO_bshd,
                               "flash_attn_fused (FP16 [B,S,H,D])");

    printf("\n  Fused vs br16: %+.1f%% (%+.3f ms)\n",
           100.0 * (ms_fused - ms_br16) / ms_br16, ms_fused - ms_br16);
    printf("  Both use FP16 inputs / 48 KB smem. Fused uses stride d_model for Q.\n");
    printf("  Pipeline savings: eliminates 4 transpose kernel launches.\n");

    cuMemFree(dQHm_bhsd); cuMemFree(dKHm_bhsd); cuMemFree(dVHm_bhsd); cuMemFree(dO_bhsd);
    cuMemFree(dQHm_bshd); cuMemFree(dKHm_bshd); cuMemFree(dVHm_bshd); cuMemFree(dO_bshd);
    cuModuleUnload(mod_br16); cuModuleUnload(mod_fused);
    cuCtxDestroy(ctx);
    return 0;
}
