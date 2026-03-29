/*
 * bench_pipeline.cu — flash_attn_br16 vs flash_attn_br16_pipeline
 *
 * Tests the hypothesis from the cross-attention postmortem:
 *   cp.async benefits kernels with many KV tile iterations (seq_len >> Bc)
 *   where K/V data stays cold in DRAM across block waves.
 *
 * Build:
 *   nvcc -arch=sm_86 -O2 -o bench_pipeline bench_pipeline.cu \
 *        -lcuda -I../../phase2/common
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <cuda_fp16.h>

#include "../../phase2/common/bench.h"
#include "../../phase2/common/check.h"

#define D_HEAD    64
#define Br_BLOCK  64
#define Bc        64

static void fp32_to_fp16_buf(const float *src, unsigned short *dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        __half h = __float2half(src[i]);
        memcpy(&dst[i], &h, 2);
    }
}

// CPU reference: two-pass naive self-attention (single head, single batch)
static void cpu_self_attn(
    const float *Q, const float *K, const float *V, float *O,
    int seq_len, float scale
) {
    for (int qi = 0; qi < seq_len; qi++) {
        float *scores = new float[seq_len];
        float row_max = -1e38f;
        for (int ki = 0; ki < seq_len; ki++) {
            double dot = 0.0;
            for (int d = 0; d < D_HEAD; d++)
                dot += (double)Q[qi * D_HEAD + d] * (double)K[ki * D_HEAD + d];
            scores[ki] = (float)(dot * scale);
            if (scores[ki] > row_max) row_max = scores[ki];
        }
        float row_sum = 0.0f;
        for (int ki = 0; ki < seq_len; ki++) {
            scores[ki] = expf(scores[ki] - row_max);
            row_sum += scores[ki];
        }
        for (int d = 0; d < D_HEAD; d++) {
            double acc = 0.0;
            for (int ki = 0; ki < seq_len; ki++)
                acc += (double)(scores[ki] / row_sum) * (double)V[ki * D_HEAD + d];
            O[qi * D_HEAD + d] = (float)acc;
        }
        delete[] scores;
    }
}

int main(void) {
    struct Config {
        int seq_len, batch, heads;
        const char *label;
    } configs[] = {
        {  256, 8, 8, "seq=256   (4 KV iters)   — moderate"},
        {  512, 8, 8, "seq=512   (8 KV iters)   — good"},
        { 1024, 8, 8, "seq=1024  (16 KV iters)  — target"},
        { 2048, 4, 8, "seq=2048  (32 KV iters)  — best"},
    };
    int num_configs = (int)(sizeof(configs) / sizeof(configs[0]));

    float scale = 1.0f / sqrtf((float)D_HEAD);

    CHECK_CU(cuInit(0));
    CUdevice  cu_dev; CHECK_CU(cuDeviceGet(&cu_dev, 0));
    CUcontext cu_ctx; CHECK_CU(cuCtxCreate(&cu_ctx, 0, cu_dev));

    CUmodule   mod_base, mod_pipe;
    CUfunction fn_base, fn_pipe;
    CHECK_CU(cuModuleLoad(&mod_base, "flash_br16.sm_86.cubin"));
    CHECK_CU(cuModuleLoad(&mod_pipe, "flash_br16_pipeline.sm_86.cubin"));
    CHECK_CU(cuModuleGetFunction(&fn_base, mod_base, "flash_attn_br16"));
    CHECK_CU(cuModuleGetFunction(&fn_pipe, mod_pipe, "flash_attn_br16_pipeline"));

    int smem_base = 48 * 1024;
    int smem_pipe = 64 * 1024;
    CHECK_CU(cuFuncSetAttribute(fn_base, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_base));
    CHECK_CU(cuFuncSetAttribute(fn_pipe, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_pipe));

    printf("=== Flash Attention: Baseline vs cp.async Pipelined ===\n");
    printf("  Baseline:  flash_attn_br16         (sync LDG, 48 KB smem)\n");
    printf("  Pipelined: flash_attn_br16_pipeline (LDGSTS, 64 KB smem)\n\n");

    for (int ci = 0; ci < num_configs; ci++) {
        int seq_len = configs[ci].seq_len;
        int batch   = configs[ci].batch;
        int heads   = configs[ci].heads;
        int kv_iters = seq_len / Bc;

        size_t qkvo_elems = (size_t)batch * heads * seq_len * D_HEAD;

        float          *h_Q_f32 = new float[qkvo_elems];
        float          *h_K_f32 = new float[qkvo_elems];
        float          *h_V_f32 = new float[qkvo_elems];
        float          *h_O_ref = new float[qkvo_elems];
        float          *h_O_base= new float[qkvo_elems];
        float          *h_O_pipe= new float[qkvo_elems];
        unsigned short *h_Q_f16 = new unsigned short[qkvo_elems];
        unsigned short *h_K_f16 = new unsigned short[qkvo_elems];
        unsigned short *h_V_f16 = new unsigned short[qkvo_elems];

        srand(42 + ci);
        for (size_t i = 0; i < qkvo_elems; i++) h_Q_f32[i] = 0.1f * ((float)rand()/RAND_MAX - 0.5f);
        for (size_t i = 0; i < qkvo_elems; i++) h_K_f32[i] = 0.1f * ((float)rand()/RAND_MAX - 0.5f);
        for (size_t i = 0; i < qkvo_elems; i++) h_V_f32[i] = 0.1f * ((float)rand()/RAND_MAX - 0.5f);

        fp32_to_fp16_buf(h_Q_f32, h_Q_f16, qkvo_elems);
        fp32_to_fp16_buf(h_K_f32, h_K_f16, qkvo_elems);
        fp32_to_fp16_buf(h_V_f32, h_V_f16, qkvo_elems);

        // CPU reference for first head/batch only (slow at large seq)
        bool ran_cpu = (seq_len <= 512);
        if (ran_cpu) {
            cpu_self_attn(h_Q_f32, h_K_f32, h_V_f32, h_O_ref, seq_len, scale);
        }

        CUdeviceptr d_Q, d_K, d_V, d_O;
        CHECK_CU(cuMemAlloc(&d_Q, qkvo_elems * 2));
        CHECK_CU(cuMemAlloc(&d_K, qkvo_elems * 2));
        CHECK_CU(cuMemAlloc(&d_V, qkvo_elems * 2));
        CHECK_CU(cuMemAlloc(&d_O, qkvo_elems * sizeof(float)));

        CHECK_CU(cuMemcpyHtoD(d_Q, h_Q_f16, qkvo_elems * 2));
        CHECK_CU(cuMemcpyHtoD(d_K, h_K_f16, qkvo_elems * 2));
        CHECK_CU(cuMemcpyHtoD(d_V, h_V_f16, qkvo_elems * 2));

        int grid_q = (seq_len + Br_BLOCK - 1) / Br_BLOCK;

        // ---- Correctness: baseline ----
        {
            void *args[] = { &d_Q, &d_K, &d_V, &d_O, &seq_len, &heads, &scale };
            CHECK_CU(cuLaunchKernel(fn_base, grid_q, heads, batch, 128,1,1,
                                    smem_base, 0, args, 0));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(h_O_base, d_O, qkvo_elems * sizeof(float)));
        }

        // ---- Correctness: pipelined ----
        {
            CHECK_CU(cuMemsetD32(d_O, 0, qkvo_elems));
            void *args[] = { &d_Q, &d_K, &d_V, &d_O, &seq_len, &heads, &scale };
            CHECK_CU(cuLaunchKernel(fn_pipe, grid_q, heads, batch, 128,1,1,
                                    smem_pipe, 0, args, 0));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(h_O_pipe, d_O, qkvo_elems * sizeof(float)));
        }

        float max_abs_vs_base = 0.0f;
        for (size_t i = 0; i < qkvo_elems; i++)
            max_abs_vs_base = fmaxf(max_abs_vs_base, fabsf(h_O_pipe[i] - h_O_base[i]));

        printf("--- %s ---\n", configs[ci].label);
        printf("  KV tile iterations: %d\n", kv_iters);
        printf("  vs baseline: %s (max_abs=%.2e)\n",
               max_abs_vs_base < 1e-4f ? "PASS" : "FAIL", (double)max_abs_vs_base);

        if (ran_cpu) {
            float max_abs_vs_ref = 0.0f;
            // compare first head/batch only
            int head0_elems = seq_len * D_HEAD;
            for (int i = 0; i < head0_elems; i++)
                max_abs_vs_ref = fmaxf(max_abs_vs_ref, fabsf(h_O_pipe[i] - h_O_ref[i]));
            printf("  vs CPU ref:  max_abs=%.3e\n", (double)max_abs_vs_ref);
        }

        // ---- Timing: baseline ----
        float ms_base = 0.0f;
        {
            void *args[] = { &d_Q, &d_K, &d_V, &d_O, &seq_len, &heads, &scale };
            for (int t = 0; t < 20; t++)
                CHECK_CU(cuLaunchKernel(fn_base, grid_q, heads, batch, 128,1,1, smem_base,0, args,0));
            CHECK_CU(cuCtxSynchronize());
            BenchTimer bt; bt.start();
            for (int t = 0; t < 200; t++)
                CHECK_CU(cuLaunchKernel(fn_base, grid_q, heads, batch, 128,1,1, smem_base,0, args,0));
            ms_base = bt.stop_ms() / 200.0f;
        }

        // ---- Timing: pipelined ----
        float ms_pipe = 0.0f;
        {
            void *args[] = { &d_Q, &d_K, &d_V, &d_O, &seq_len, &heads, &scale };
            for (int t = 0; t < 20; t++)
                CHECK_CU(cuLaunchKernel(fn_pipe, grid_q, heads, batch, 128,1,1, smem_pipe,0, args,0));
            CHECK_CU(cuCtxSynchronize());
            BenchTimer bt; bt.start();
            for (int t = 0; t < 200; t++)
                CHECK_CU(cuLaunchKernel(fn_pipe, grid_q, heads, batch, 128,1,1, smem_pipe,0, args,0));
            ms_pipe = bt.stop_ms() / 200.0f;
        }

        // Ideal bandwidth: Q+K+V+O read/written once
        float bytes = (float)qkvo_elems * (3 * 2 + 1 * 4);  // 3× FP16 + 1× FP32
        double gflops_base = 4.0 * seq_len * seq_len * D_HEAD * batch * heads / (ms_base*1e-3) / 1e9;
        double gflops_pipe = 4.0 * seq_len * seq_len * D_HEAD * batch * heads / (ms_pipe*1e-3) / 1e9;

        printf("  Baseline:   %.3f ms  %6.0f GFLOPS  %5.1f GB/s\n",
               ms_base, gflops_base, bytes / (ms_base * 1e-3f) / 1e9f);
        printf("  Pipelined:  %.3f ms  %6.0f GFLOPS  %5.1f GB/s  (%.2f× speedup)\n\n",
               ms_pipe, gflops_pipe, bytes / (ms_pipe * 1e-3f) / 1e9f,
               ms_base / ms_pipe);

        cuMemFree(d_Q); cuMemFree(d_K); cuMemFree(d_V); cuMemFree(d_O);
        delete[] h_Q_f32; delete[] h_K_f32; delete[] h_V_f32;
        delete[] h_O_ref; delete[] h_O_base; delete[] h_O_pipe;
        delete[] h_Q_f16; delete[] h_K_f16; delete[] h_V_f16;
    }

    cuModuleUnload(mod_base);
    cuModuleUnload(mod_pipe);
    cuCtxDestroy(cu_ctx);
    return 0;
}
