/*
 * bench_pipeline.cu — Baseline vs cp.async pipelined Flash Attention (BenchDriver)
 *
 * Tests cp.async benefit at many KV tile iterations (seq_len >> Bc).
 *
 * Build:
 *   nvcc -arch=sm_86 -O2 -o bench_pipeline bench_pipeline.cu -lcuda -I../../kernels/_common
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <cuda_fp16.h>

#include "../../_common/bench_driver.h"

#define D_HEAD   64
#define Br_BLOCK 64
#define Bc       64

static void fp32_to_f16buf(const float *src, unsigned short *dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        __half h = __float2half(src[i]);
        memcpy(&dst[i], &h, 2);
    }
}

static void cpu_self_attn(const float *Q, const float *K, const float *V, float *O,
                          int seq, float scale) {
    for (int q = 0; q < seq; q++) {
        float *s = (float*)malloc(seq * sizeof(float));
        float row_max = -1e38f;
        for (int k = 0; k < seq; k++) {
            double dot = 0.0;
            for (int d = 0; d < D_HEAD; d++) dot += (double)Q[q * D_HEAD + d] * (double)K[k * D_HEAD + d];
            s[k] = (float)(dot * scale);
            row_max = fmaxf(row_max, s[k]);
        }
        float sum = 0.0f;
        for (int k = 0; k < seq; k++) { s[k] = expf(s[k] - row_max); sum += s[k]; }
        for (int d = 0; d < D_HEAD; d++) {
            double acc = 0.0;
            for (int k = 0; k < seq; k++) acc += (double)(s[k] / sum) * (double)V[k * D_HEAD + d];
            O[q * D_HEAD + d] = (float)acc;
        }
        free(s);
    }
}

int main(void) {
    struct Config { int seq, batch, heads; const char *label; }
    configs[] = {
        { 256, 8, 8, "seq=256  (4 KV)"},
        { 512, 8, 8, "seq=512  (8 KV)"},
        {1024, 8, 8, "seq=1024 (16 KV)"},
        {2048, 4, 8, "seq=2048 (32 KV)"},
    };
    int ncfg = (int)(sizeof(configs) / sizeof(configs[0]));
    float scale = 1.0f / sqrtf((float)D_HEAD);

    BenchDriver driver;
    driver.init_context();

    CUmodule mod_base, mod_pipe;
    CHECK_CU(cuModuleLoad(&mod_base, "flash_br16.sm_86.cubin"));
    CHECK_CU(cuModuleLoad(&mod_pipe, "flash_br16_pipeline.sm_86.cubin"));
    CUfunction fn_base, fn_pipe;
    CHECK_CU(cuModuleGetFunction(&fn_base, mod_base, "flash_attn_br16"));
    CHECK_CU(cuModuleGetFunction(&fn_pipe, mod_pipe, "flash_attn_br16_pipeline"));

    int smem_base = 48 * 1024;
    int smem_pipe = 64 * 1024;
    CHECK_CU(cuFuncSetAttribute(fn_base, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_base));
    CHECK_CU(cuFuncSetAttribute(fn_pipe, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_pipe));

    printf("=== Flash Attention: Baseline vs cp.async Pipelined ===\n");
    printf("  Baseline:  LDG,  48 KB smem\n");
    printf("  Pipelined: LDGSTS, 64 KB smem\n\n");

    for (int ci = 0; ci < ncfg; ci++) {
        int seq = configs[ci].seq, batch = configs[ci].batch, heads = configs[ci].heads;
        int kv = seq / Bc, gq = (seq + Br_BLOCK - 1) / Br_BLOCK;
        size_t ne  = (size_t)batch * heads * seq * D_HEAD;
        size_t nf16 = ne * 2;
        size_t nf32 = ne * sizeof(float);

        auto h_Qf = driver.host_alloc<float>(ne);
        auto h_Kf = driver.host_alloc<float>(ne);
        auto h_Vf = driver.host_alloc<float>(ne);
        auto h_Or = driver.host_alloc<float>(ne);
        auto h_Ob = driver.host_alloc<float>(ne);
        auto h_Op = driver.host_alloc<float>(ne);
        auto h_Qh = driver.host_alloc<unsigned short>(ne);
        auto h_Kh = driver.host_alloc<unsigned short>(ne);
        auto h_Vh = driver.host_alloc<unsigned short>(ne);

        srand(42 + ci);
        for (size_t i = 0; i < ne; i++) h_Qf[i] = 0.1f * ((float)rand()/RAND_MAX - 0.5f);
        for (size_t i = 0; i < ne; i++) h_Kf[i] = 0.1f * ((float)rand()/RAND_MAX - 0.5f);
        for (size_t i = 0; i < ne; i++) h_Vf[i] = 0.1f * ((float)rand()/RAND_MAX - 0.5f);
        fp32_to_f16buf(h_Qf.get(), h_Qh.get(), ne);
        fp32_to_f16buf(h_Kf.get(), h_Kh.get(), ne);
        fp32_to_f16buf(h_Vf.get(), h_Vh.get(), ne);

        bool ran_cpu = (seq <= 512);
        if (ran_cpu) cpu_self_attn(h_Qf.get(), h_Kf.get(), h_Vf.get(), h_Or.get(), seq, scale);

        auto dQ = driver.device_alloc<unsigned short>(ne);
        auto dK = driver.device_alloc<unsigned short>(ne);
        auto dV = driver.device_alloc<unsigned short>(ne);
        auto dO = driver.device_alloc<float>(ne);
        driver.copy_h2d(dQ, h_Qh, nf16);
        driver.copy_h2d(dK, h_Kh, nf16);
        driver.copy_h2d(dV, h_Vh, nf16);

        void *args[] = { &dQ.ptr, &dK.ptr, &dV.ptr, &dO.ptr, &seq, &heads, &scale };

        CHECK_CU(cuLaunchKernel(fn_base, gq, heads, batch, 128, 1, 1, smem_base, nullptr, args, nullptr));
        CHECK_CU(cuCtxSynchronize());
        driver.copy_d2h(h_Ob, dO, nf32);

        CHECK_CU(cuMemsetD32((CUdeviceptr)dO.ptr, 0, ne));
        CHECK_CU(cuLaunchKernel(fn_pipe, gq, heads, batch, 128, 1, 1, smem_pipe, nullptr, args, nullptr));
        CHECK_CU(cuCtxSynchronize());
        driver.copy_d2h(h_Op, dO, nf32);

        float max_abs = 0.0f;
        for (size_t i = 0; i < ne; i++) max_abs = fmaxf(max_abs, fabsf(h_Op[i] - h_Ob[i]));

        printf("--- %s ---\n", configs[ci].label);
        printf("  KV iters: %d\n", kv);
        printf("  vs baseline: %s (max_abs=%.2e)\n", max_abs < 1e-4f ? "PASS" : "FAIL", (double)max_abs);
        if (ran_cpu) {
            float max_ref = 0.0f;
            for (int i = 0; i < seq * D_HEAD; i++) max_ref = fmaxf(max_ref, fabsf(h_Op[i] - h_Or[i]));
            printf("  vs CPU:      max_abs=%.3e\n", (double)max_ref);
        }

        // Timing baseline
        for (int t = 0; t < 20; t++)
            CHECK_CU(cuLaunchKernel(fn_base, gq, heads, batch, 128, 1, 1, smem_base, nullptr, args, nullptr));
        CHECK_CU(cuCtxSynchronize());
        BenchTimer t1; t1.start();
        for (int t = 0; t < 200; t++)
            CHECK_CU(cuLaunchKernel(fn_base, gq, heads, batch, 128, 1, 1, smem_base, nullptr, args, nullptr));
        float ms_base = t1.stop_ms() / 200.0f;

        // Timing pipelined
        for (int t = 0; t < 20; t++)
            CHECK_CU(cuLaunchKernel(fn_pipe, gq, heads, batch, 128, 1, 1, smem_pipe, nullptr, args, nullptr));
        CHECK_CU(cuCtxSynchronize());
        BenchTimer t2; t2.start();
        for (int t = 0; t < 200; t++)
            CHECK_CU(cuLaunchKernel(fn_pipe, gq, heads, batch, 128, 1, 1, smem_pipe, nullptr, args, nullptr));
        float ms_pipe = t2.stop_ms() / 200.0f;

        float bytes = (float)ne * (3 * 2 + 4);
        double gflops_b = 4.0 * seq * seq * D_HEAD * batch * heads / (ms_base * 1e-3) / 1e9;
        double gflops_p = 4.0 * seq * seq * D_HEAD * batch * heads / (ms_pipe  * 1e-3) / 1e9;
        printf("  Baseline:  %.3f ms  %6.0f GFLOPS  %5.1f GB/s\n",
               ms_base, gflops_b, bytes / (ms_base * 1e-3f) / 1e9f);
        printf("  Pipelined: %.3f ms  %6.0f GFLOPS  %5.1f GB/s  (%.2fx speedup)\n\n",
               ms_pipe, gflops_p, bytes / (ms_pipe * 1e-3f) / 1e9f, ms_base / ms_pipe);
    }

    cuModuleUnload(mod_base);
    cuModuleUnload(mod_pipe);
    return 0;
}
