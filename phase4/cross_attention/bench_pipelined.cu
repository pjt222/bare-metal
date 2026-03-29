/*
 * bench_pipelined.cu — Pipelined (cp.async) vs baseline cross-attention
 *
 * Compares cross_attn_br16 (synchronous LDG tile loads) against
 * cross_attn_pipelined (cp.async double-buffered tile loads).
 *
 * Build:
 *   nvcc -arch=sm_86 -O2 -o bench_pipelined bench_pipelined.cu -lcuda -I../../phase2/common
 *
 * Usage:
 *   ./bench_pipelined
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <cuda_fp16.h>

#include "../../phase2/common/bench.h"
#include "../../phase2/common/check.h"

// Kernel constants (must match both kernel files)
#define D_HEAD     64
#define Br_BLOCK   64
#define Bc         64

// -----------------------------------------------------------------------
// FP32 → FP16 host conversion helpers
// -----------------------------------------------------------------------
static void fp32_to_fp16_array(const float *src, unsigned short *dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        __half h = __float2half(src[i]);
        memcpy(&dst[i], &h, 2);
    }
}

// -----------------------------------------------------------------------
// CPU reference: naive cross-attention
//   Q: [batch, heads, seq_q,  D_HEAD] FP32
//   K: [batch, heads, seq_kv, D_HEAD] FP32
//   V: [batch, heads, seq_kv, D_HEAD] FP32
//   O: [batch, heads, seq_q,  D_HEAD] FP32
// -----------------------------------------------------------------------
static void cpu_cross_attn(
    const float *Q, const float *K, const float *V, float *O,
    int batch, int heads, int seq_q, int seq_kv, float scale
) {
    for (int b = 0; b < batch; b++)
    for (int h = 0; h < heads; h++) {
        size_t q_offset  = ((size_t)b * heads + h) * seq_q  * D_HEAD;
        size_t kv_offset = ((size_t)b * heads + h) * seq_kv * D_HEAD;

        for (int qi = 0; qi < seq_q; qi++) {
            // Compute raw scores: S[qi, :] = Q[qi] · K[:]^T * scale
            float *scores = new float[seq_kv];
            float max_s   = -1e38f;
            for (int kv = 0; kv < seq_kv; kv++) {
                double dot = 0.0;
                for (int d = 0; d < D_HEAD; d++)
                    dot += (double)Q[q_offset + qi * D_HEAD + d]
                         * (double)K[kv_offset + kv * D_HEAD + d];
                scores[kv] = (float)(dot * scale);
                if (scores[kv] > max_s) max_s = scores[kv];
            }
            // Softmax
            float sum = 0.0f;
            for (int kv = 0; kv < seq_kv; kv++) {
                scores[kv] = expf(scores[kv] - max_s);
                sum += scores[kv];
            }
            for (int kv = 0; kv < seq_kv; kv++) scores[kv] /= sum;

            // Weighted sum over V
            size_t o_offset = q_offset + qi * D_HEAD;
            for (int d = 0; d < D_HEAD; d++) {
                double acc = 0.0;
                for (int kv = 0; kv < seq_kv; kv++)
                    acc += (double)scores[kv] * V[kv_offset + kv * D_HEAD + d];
                O[o_offset + d] = (float)acc;
            }
            delete[] scores;
        }
    }
}

// -----------------------------------------------------------------------
// main
// -----------------------------------------------------------------------
int main(void) {
    // ---- Test configurations ----
    struct Config {
        int seq_q, seq_kv, batch, heads;
        const char *label;
    } configs[] = {
        {  64,  77, 1, 8, "SD  8× 8 (sq=64,  skv=77)"},
        { 256,  77, 1, 8, "SD 16×16 (sq=256, skv=77)"},
        {1024,  77, 1, 8, "SD 32×32 (sq=1024,skv=77)"},
        {4096,  77, 1, 8, "SD 64×64 (sq=4096,skv=77)"},
        {1024, 512, 1, 8, "Long ctx (sq=1024,skv=512)"},
    };
    int num_configs = sizeof(configs) / sizeof(configs[0]);

    float scale = 1.0f / sqrtf((float)D_HEAD);

    // ---- CUDA setup ----
    CHECK_CU(cuInit(0));
    CUdevice  cu_dev; CHECK_CU(cuDeviceGet(&cu_dev, 0));
    CUcontext cu_ctx; CHECK_CU(cuCtxCreate(&cu_ctx, 0, cu_dev));

    // Load cubins
    CUmodule  mod_base, mod_pipe;
    CUfunction fn_base, fn_pipe;
    CHECK_CU(cuModuleLoad(&mod_base, "cross_attn.sm_86.cubin"));
    CHECK_CU(cuModuleLoad(&mod_pipe, "cross_attn_pipelined.sm_86.cubin"));
    CHECK_CU(cuModuleGetFunction(&fn_base, mod_base, "cross_attn_br16"));
    CHECK_CU(cuModuleGetFunction(&fn_pipe, mod_pipe, "cross_attn_pipelined"));

    // Baseline smem: 48 KB
    int smem_base = 48 * 1024;
    CHECK_CU(cuFuncSetAttribute(fn_base, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_base));

    // Pipelined smem: 64 KB (double K/V buffers)
    int smem_pipe = 64 * 1024;
    CHECK_CU(cuFuncSetAttribute(fn_pipe, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_pipe));

    printf("=== Pipelined vs Baseline Cross-Attention ===\n");
    printf("  Baseline:   cross_attn_br16 (synchronous LDG, 48 KB smem)\n");
    printf("  Pipelined:  cross_attn_pipelined (cp.async LDGSTS, 64 KB smem)\n\n");

    for (int ci = 0; ci < num_configs; ci++) {
        int seq_q  = configs[ci].seq_q;
        int seq_kv = configs[ci].seq_kv;
        int batch  = configs[ci].batch;
        int heads  = configs[ci].heads;
        const char *label = configs[ci].label;

        size_t Q_elems  = (size_t)batch * heads * seq_q  * D_HEAD;
        size_t KV_elems = (size_t)batch * heads * seq_kv * D_HEAD;

        // ---- Host data (FP32 for reference) ----
        float *h_Q_fp32  = new float[Q_elems];
        float *h_K_fp32  = new float[KV_elems];
        float *h_V_fp32  = new float[KV_elems];
        float *h_O_ref   = new float[Q_elems];
        float *h_O_base  = new float[Q_elems];
        float *h_O_pipe  = new float[Q_elems];

        srand(42 + ci);
        for (size_t i = 0; i < Q_elems;  i++) h_Q_fp32[i] = 0.1f * ((float)rand()/RAND_MAX - 0.5f);
        for (size_t i = 0; i < KV_elems; i++) h_K_fp32[i] = 0.1f * ((float)rand()/RAND_MAX - 0.5f);
        for (size_t i = 0; i < KV_elems; i++) h_V_fp32[i] = 0.1f * ((float)rand()/RAND_MAX - 0.5f);

        // CPU reference (skip for large configs to save time)
        bool ran_cpu = false;
        if (seq_q * seq_kv < 1024 * 512) {
            cpu_cross_attn(h_Q_fp32, h_K_fp32, h_V_fp32, h_O_ref,
                           batch, heads, seq_q, seq_kv, scale);
            ran_cpu = true;
        }

        // ---- FP16 device arrays (Q, K, V) ----
        unsigned short *h_Q_fp16  = new unsigned short[Q_elems];
        unsigned short *h_K_fp16  = new unsigned short[KV_elems];
        unsigned short *h_V_fp16  = new unsigned short[KV_elems];
        fp32_to_fp16_array(h_Q_fp32, h_Q_fp16, Q_elems);
        fp32_to_fp16_array(h_K_fp32, h_K_fp16, KV_elems);
        fp32_to_fp16_array(h_V_fp32, h_V_fp16, KV_elems);

        CUdeviceptr d_Q, d_K, d_V, d_O;
        CHECK_CU(cuMemAlloc(&d_Q, Q_elems  * 2));
        CHECK_CU(cuMemAlloc(&d_K, KV_elems * 2));
        CHECK_CU(cuMemAlloc(&d_V, KV_elems * 2));
        CHECK_CU(cuMemAlloc(&d_O, Q_elems  * sizeof(float)));

        CHECK_CU(cuMemcpyHtoD(d_Q, h_Q_fp16, Q_elems  * 2));
        CHECK_CU(cuMemcpyHtoD(d_K, h_K_fp16, KV_elems * 2));
        CHECK_CU(cuMemcpyHtoD(d_V, h_V_fp16, KV_elems * 2));

        // Grid for both kernels: (ceil(seq_q/64), heads, batch)
        int grid_q = (seq_q + Br_BLOCK - 1) / Br_BLOCK;

        // ---- Correctness check: baseline ----
        {
            void *args[] = { &d_Q, &d_K, &d_V, &d_O, &seq_q, &seq_kv, &heads, &scale };
            CHECK_CU(cuLaunchKernel(fn_base, grid_q, heads, batch, 128, 1, 1,
                                    smem_base, 0, args, 0));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(h_O_base, d_O, Q_elems * sizeof(float)));
        }

        // ---- Correctness check: pipelined ----
        {
            CHECK_CU(cuMemsetD32(d_O, 0, Q_elems));
            void *args[] = { &d_Q, &d_K, &d_V, &d_O, &seq_q, &seq_kv, &heads, &scale };
            CHECK_CU(cuLaunchKernel(fn_pipe, grid_q, heads, batch, 128, 1, 1,
                                    smem_pipe, 0, args, 0));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(h_O_pipe, d_O, Q_elems * sizeof(float)));
        }

        // ---- Verify pipelined matches baseline (primary check) ----
        float max_abs_vs_base = 0.0f;
        for (size_t i = 0; i < Q_elems; i++) {
            float diff = fabsf(h_O_pipe[i] - h_O_base[i]);
            if (diff > max_abs_vs_base) max_abs_vs_base = diff;
        }

        float max_abs_vs_ref = -1.0f;
        if (ran_cpu) {
            max_abs_vs_ref = 0.0f;
            for (size_t i = 0; i < Q_elems; i++) {
                float diff = fabsf(h_O_pipe[i] - h_O_ref[i]);
                if (diff > max_abs_vs_ref) max_abs_vs_ref = diff;
            }
        }

        printf("--- %s ---\n", label);
        bool match_base = (max_abs_vs_base < 1e-4f);
        printf("  vs baseline: %s (max_abs=%.2e)\n",
               match_base ? "PASS" : "FAIL", (double)max_abs_vs_base);
        if (ran_cpu)
            printf("  vs CPU ref:  max_abs=%.3e\n", (double)max_abs_vs_ref);

        // ---- Timing: baseline ----
        float ms_base = 0.0f;
        {
            void *args[] = { &d_Q, &d_K, &d_V, &d_O, &seq_q, &seq_kv, &heads, &scale };
            for (int t = 0; t < 20; t++)
                CHECK_CU(cuLaunchKernel(fn_base, grid_q, heads, batch, 128,1,1, smem_base,0, args,0));
            CHECK_CU(cuCtxSynchronize());
            BenchTimer base_timer;
            base_timer.start();
            for (int t = 0; t < 200; t++)
                CHECK_CU(cuLaunchKernel(fn_base, grid_q, heads, batch, 128,1,1, smem_base,0, args,0));
            ms_base = base_timer.stop_ms() / 200.0f;
        }

        // ---- Timing: pipelined ----
        float ms_pipe = 0.0f;
        {
            void *args[] = { &d_Q, &d_K, &d_V, &d_O, &seq_q, &seq_kv, &heads, &scale };
            for (int t = 0; t < 20; t++)
                CHECK_CU(cuLaunchKernel(fn_pipe, grid_q, heads, batch, 128,1,1, smem_pipe,0, args,0));
            CHECK_CU(cuCtxSynchronize());
            BenchTimer pipe_timer;
            pipe_timer.start();
            for (int t = 0; t < 200; t++)
                CHECK_CU(cuLaunchKernel(fn_pipe, grid_q, heads, batch, 128,1,1, smem_pipe,0, args,0));
            ms_pipe = pipe_timer.stop_ms() / 200.0f;
        }

        // GFLOPS: 2 × seq_q × seq_kv × D_HEAD × 2 (QK^T + PV)
        double total_flops = 2.0 * seq_q * seq_kv * D_HEAD * 2 * batch * heads;
        double gflops_base = total_flops / (ms_base * 1e-3) / 1e9;
        double gflops_pipe = total_flops / (ms_pipe * 1e-3) / 1e9;

        printf("  Baseline:   %.3f ms  → %6.0f GFLOPS\n", ms_base, gflops_base);
        printf("  Pipelined:  %.3f ms  → %6.0f GFLOPS  (%.2f× speedup)\n",
               ms_pipe, gflops_pipe, ms_base / ms_pipe);
        printf("\n");

        // Cleanup
        cuMemFree(d_Q); cuMemFree(d_K); cuMemFree(d_V); cuMemFree(d_O);
        delete[] h_Q_fp32; delete[] h_K_fp32; delete[] h_V_fp32;
        delete[] h_O_ref; delete[] h_O_base; delete[] h_O_pipe;
        delete[] h_Q_fp16; delete[] h_K_fp16; delete[] h_V_fp16;
    }

    cuModuleUnload(mod_base);
    cuModuleUnload(mod_pipe);
    cuCtxDestroy(cu_ctx);
    return 0;
}
