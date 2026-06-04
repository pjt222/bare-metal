/*
 * bench_bc128.cu — Bc=64 vs Bc=128 Flash Attention benchmark (BenchDriver)
 *
 * Tests hypothesis: doubling Bc halves KV iterations, increases HMMA/tile,
 * may offset 50 KB smem cliff occupancy loss.
 *
 * Build:
 *   nvcc -arch=sm_86 -O2 -o bench_bc128 bench_bc128.cu -lcuda -I../../kernels/_common
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
#define Bc64     64
#define Bc128    128

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
        { 256, 8, 8, "seq=256  (Bc64:4  Bc128:2)"},
        { 512, 8, 8, "seq=512  (Bc64:8  Bc128:4)"},
        {1024, 8, 8, "seq=1024 (Bc64:16 Bc128:8)"},
        {2048, 4, 8, "seq=2048 (Bc64:32 Bc128:16)"},
    };
    int ncfg = (int)(sizeof(configs) / sizeof(configs[0]));
    float scale = 1.0f / sqrtf((float)D_HEAD);

    BenchDriver driver;
    driver.init_context();

    CUmodule mod_base, mod_bc128;
    CHECK_CU(cuModuleLoad(&mod_base,  "flash_attn_br16.sm_86.cubin"));
    CHECK_CU(cuModuleLoad(&mod_bc128, "flash_attn_br16_bc128.sm_86.cubin"));

    CUfunction fn_base, fn_bc128;
    CHECK_CU(cuModuleGetFunction(&fn_base,  mod_base,  "flash_attn_br16"));
    CHECK_CU(cuModuleGetFunction(&fn_bc128, mod_bc128, "flash_attn_bc128"));

    int smem_base  = 48 * 1024;
    int smem_bc128 = 80 * 1024;
    CHECK_CU(cuFuncSetAttribute(fn_base,  CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_base));
    CHECK_CU(cuFuncSetAttribute(fn_bc128, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bc128));

    printf("=== Flash Attention: Bc=64 vs Bc=128 ===\n");
    printf("  Bc=64:  48 KB smem,   64 HMMA/tile\n");
    printf("  Bc=128: 80 KB smem,  128 HMMA/tile\n\n");

    for (int ci = 0; ci < ncfg; ci++) {
        int seq = configs[ci].seq, batch = configs[ci].batch, heads = configs[ci].heads;
        int kv64 = seq / Bc64, kv128 = seq / Bc128;
        int gq = (seq + Br_BLOCK - 1) / Br_BLOCK;

        size_t ne = (size_t)batch * heads * seq * D_HEAD;
        size_t nb_f16 = ne * 2;
        size_t nb_f32 = ne * sizeof(float);

        auto h_Qf = driver.host_alloc<float>(ne);
        auto h_Kf = driver.host_alloc<float>(ne);
        auto h_Vf = driver.host_alloc<float>(ne);
        auto h_Or = driver.host_alloc<float>(ne);
        auto h_Ob = driver.host_alloc<float>(ne);
        auto h_Oc = driver.host_alloc<float>(ne);
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
        driver.copy_h2d(dQ, h_Qh, nb_f16);
        driver.copy_h2d(dK, h_Kh, nb_f16);
        driver.copy_h2d(dV, h_Vh, nb_f16);

        // Run baseline
        void *args[] = { &dQ.ptr, &dK.ptr, &dV.ptr, &dO.ptr, &seq, &heads, &scale };
        CHECK_CU(cuLaunchKernel(fn_base, gq, heads, batch, 128, 1, 1, smem_base, nullptr, args, nullptr));
        CHECK_CU(cuCtxSynchronize());
        driver.copy_d2h(h_Ob, dO, nb_f32);

        // Run bc128
        CHECK_CU(cuMemsetD32((CUdeviceptr)dO.ptr, 0, ne));
        CHECK_CU(cuLaunchKernel(fn_bc128, gq, heads, batch, 128, 1, 1, smem_bc128, nullptr, args, nullptr));
        CHECK_CU(cuCtxSynchronize());
        driver.copy_d2h(h_Oc, dO, nb_f32);

        float max_abs = 0.0f;
        for (size_t i = 0; i < ne; i++) max_abs = fmaxf(max_abs, fabsf(h_Oc[i] - h_Ob[i]));

        printf("--- %s ---\n", configs[ci].label);
        printf("  KV iters: Bc64=%d  Bc128=%d\n", kv64, kv128);
        printf("  vs Bc=64: %s (max_abs=%.2e)\n", max_abs < 1e-3f ? "PASS" : "FAIL", (double)max_abs);
        if (ran_cpu) {
            float max_ref = 0.0f;
            for (int i = 0; i < seq * D_HEAD; i++) max_ref = fmaxf(max_ref, fabsf(h_Oc[i] - h_Or[i]));
            printf("  vs CPU:   max_abs=%.3e\n", (double)max_ref);
        }

        // Timing: baseline
        for (int t = 0; t < 20; t++)
            CHECK_CU(cuLaunchKernel(fn_base, gq, heads, batch, 128, 1, 1, smem_base, nullptr, args, nullptr));
        CHECK_CU(cuCtxSynchronize());
        BenchTimer t1; t1.start();
        for (int t = 0; t < 200; t++)
            CHECK_CU(cuLaunchKernel(fn_base, gq, heads, batch, 128, 1, 1, smem_base, nullptr, args, nullptr));
        float ms_base = t1.stop_ms() / 200.0f;

        // Timing: bc128
        for (int t = 0; t < 20; t++)
            CHECK_CU(cuLaunchKernel(fn_bc128, gq, heads, batch, 128, 1, 1, smem_bc128, nullptr, args, nullptr));
        CHECK_CU(cuCtxSynchronize());
        BenchTimer t2; t2.start();
        for (int t = 0; t < 200; t++)
            CHECK_CU(cuLaunchKernel(fn_bc128, gq, heads, batch, 128, 1, 1, smem_bc128, nullptr, args, nullptr));
        float ms_bc = t2.stop_ms() / 200.0f;

        float bytes = (float)ne * (3 * 2 + 4);
        double gflops_base = 4.0 * seq * seq * D_HEAD * batch * heads / (ms_base * 1e-3) / 1e9;
        double gflops_bc   = 4.0 * seq * seq * D_HEAD * batch * heads / (ms_bc   * 1e-3) / 1e9;

        printf("  Bc=64:   %.3f ms  %6.0f GFLOPS  %5.1f GB/s\n",
               ms_base, gflops_base, bytes / (ms_base * 1e-3f) / 1e9f);
        printf("  Bc=128:  %.3f ms  %6.0f GFLOPS  %5.1f GB/s  (%.2fx speedup)\n\n",
               ms_bc, gflops_bc, bytes / (ms_bc * 1e-3f) / 1e9f, ms_base / ms_bc);
    }

    cuModuleUnload(mod_base);
    cuModuleUnload(mod_bc128);
    return 0;
}
