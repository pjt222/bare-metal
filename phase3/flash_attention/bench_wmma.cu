/*
 * bench_wmma.cu — Benchmark flash_attn_4warp vs flash_attn_1warp
 *
 * Tests correctness and measures throughput improvement from sharing K/V tiles
 * across 4 warps (4× fewer K/V global memory reads per block).
 *
 * Build:
 *   nvcc -arch=sm_86 -O2 -o bench_wmma bench_wmma.cu -lcuda -I../../phase2/common
 *
 * Usage:
 *   ./bench_wmma              # default: seq_len=1024, batch=8, heads=8
 *   ./bench_wmma 2048 4 8
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda.h>

#include "../../phase2/common/bench.h"
#include "../../phase2/common/check.h"

// -----------------------------------------------------------------------
// CPU reference — same as bench.cu
// -----------------------------------------------------------------------
static void cpu_attention(
    const float *Q, const float *K, const float *V, float *O,
    float *score_buf, int seq_len, int d_head, float scale
) {
    for (int q = 0; q < seq_len; q++) {
        const float *q_row = Q + (size_t)q * d_head;
        float row_max = -3.402823466e+38f;
        for (int k = 0; k < seq_len; k++) {
            const float *k_row = K + (size_t)k * d_head;
            float dot = 0.0f;
            for (int d = 0; d < d_head; d++) dot += q_row[d] * k_row[d];
            score_buf[k] = dot * scale;
            row_max = fmaxf(row_max, score_buf[k]);
        }
        float exp_sum = 0.0f;
        for (int k = 0; k < seq_len; k++) {
            score_buf[k] = expf(score_buf[k] - row_max);
            exp_sum += score_buf[k];
        }
        float *o_row = O + (size_t)q * d_head;
        for (int d = 0; d < d_head; d++) o_row[d] = 0.0f;
        float rcp = 1.0f / exp_sum;
        for (int k = 0; k < seq_len; k++) {
            const float *v_row = V + (size_t)k * d_head;
            float w = score_buf[k] * rcp;
            for (int d = 0; d < d_head; d++) o_row[d] += w * v_row[d];
        }
    }
}

int main(int argc, char **argv) {
    int seq_len    = (argc > 1) ? atoi(argv[1]) : 1024;
    int batch_size = (argc > 2) ? atoi(argv[2]) : 8;
    int num_heads  = (argc > 3) ? atoi(argv[3]) : 8;

    const int d_head      = 64;
    const int block_kv_v1 = 32;   // scalar version
    const int block_kv_v2 = 64;   // 4-warp version (must match flash_attn_wmma.cu)
    const int num_warps   = 4;

    if (seq_len % block_kv_v1 != 0 || seq_len % num_warps != 0) {
        fprintf(stderr, "seq_len=%d must be divisible by %d (BLOCK_KV_v1) and %d (NUM_WARPS)\n",
                seq_len, block_kv_v1, num_warps);
        return 1;
    }

    float scale = 1.0f / sqrtf((float)d_head);

    printf("=== Flash Attention: 4-Warp KV Tile Sharing vs 1-Warp Scalar ===\n");
    printf("seq=%d  d=%d  batch=%d  heads=%d\n\n", seq_len, d_head, batch_size, num_heads);

    CHECK_CU(cuInit(0));
    CUdevice cu_device; CHECK_CU(cuDeviceGet(&cu_device, 0));
    char name[256]; CHECK_CU(cuDeviceGetName(name, sizeof(name), cu_device));
    printf("Device: %s\n\n", name);

    CUcontext cu_context; CHECK_CU(cuCtxCreate(&cu_context, 0, cu_device));

    // Load cubins
    CUmodule   mod_v1, mod_v2;
    CUfunction fn_multihead, fn_4warp;

    {
        CUresult r = cuModuleLoad(&mod_v1, "flash_attn.sm_86.cubin");
        if (r != CUDA_SUCCESS) {
            fprintf(stderr, "Cannot load flash_attn.sm_86.cubin — build it first\n");
            return 1;
        }
    }
    {
        CUresult r = cuModuleLoad(&mod_v2, "flash_wmma.sm_86.cubin");
        if (r != CUDA_SUCCESS) {
            fprintf(stderr, "Cannot load flash_wmma.sm_86.cubin — build it first\n");
            return 1;
        }
    }

    CHECK_CU(cuModuleGetFunction(&fn_multihead, mod_v1, "flash_attn_multihead"));
    CHECK_CU(cuModuleGetFunction(&fn_4warp,     mod_v2, "flash_attn_4warp"));
    printf("Kernels loaded.\n\n");

    // =========================================================
    // Correctness test — single head, seq_len
    // =========================================================
    size_t sh_elems = (size_t)seq_len * d_head;
    size_t sh_bytes = sh_elems * sizeof(float);

    float *hQ    = (float*)malloc(sh_bytes);
    float *hK    = (float*)malloc(sh_bytes);
    float *hV    = (float*)malloc(sh_bytes);
    float *hO_v1 = (float*)malloc(sh_bytes);
    float *hO_v2 = (float*)malloc(sh_bytes);
    float *hRef  = (float*)malloc(sh_bytes);
    float *sBuf  = (float*)malloc(seq_len * sizeof(float));

    fill_random(hQ, sh_elems, 10);
    fill_random(hK, sh_elems, 11);
    fill_random(hV, sh_elems, 12);

    printf("Computing CPU reference...\n");
    cpu_attention(hQ, hK, hV, hRef, sBuf, seq_len, d_head, scale);
    printf("Done.\n\n");

    CUdeviceptr dQ, dK, dV, dO;
    CHECK_CU(cuMemAlloc(&dQ, sh_bytes));
    CHECK_CU(cuMemAlloc(&dK, sh_bytes));
    CHECK_CU(cuMemAlloc(&dV, sh_bytes));
    CHECK_CU(cuMemAlloc(&dO, sh_bytes));
    CHECK_CU(cuMemcpyHtoD(dQ, hQ, sh_bytes));
    CHECK_CU(cuMemcpyHtoD(dK, hK, sh_bytes));
    CHECK_CU(cuMemcpyHtoD(dV, hV, sh_bytes));

    // v1: flash_attn_multihead — grid=(seq_len, 1, 1), block=(32,1,1)
    int n1 = 1;
    void *args_v1[] = { &dQ, &dK, &dV, &dO, &seq_len, &n1, &scale };
    CHECK_CU(cuMemsetD32(dO, 0, sh_elems));
    CHECK_CU(cuLaunchKernel(fn_multihead,
        seq_len, 1, 1,   32, 1, 1,   0, NULL, args_v1, NULL));
    CHECK_CU(cuCtxSynchronize());
    CHECK_CU(cuMemcpyDtoH(hO_v1, dO, sh_bytes));

    // v2: flash_attn_4warp — grid=(seq_len/NUM_WARPS, 1, 1), block=(128,1,1)
    void *args_v2[] = { &dQ, &dK, &dV, &dO, &seq_len, &n1, &scale };
    CHECK_CU(cuMemsetD32(dO, 0, sh_elems));
    CHECK_CU(cuLaunchKernel(fn_4warp,
        seq_len / num_warps, 1, 1,   128, 1, 1,   0, NULL, args_v2, NULL));
    CHECK_CU(cuCtxSynchronize());
    CHECK_CU(cuMemcpyDtoH(hO_v2, dO, sh_bytes));

    printf("Correctness (vs CPU naive):\n");
    auto r1 = check_fp32(hO_v1, hRef, sh_elems, 1e-3f, 1e-1f);
    auto r2 = check_fp32(hO_v2, hRef, sh_elems, 1e-3f, 1e-1f);
    print_check_result("flash_attn_multihead (v1 scalar)", r1);
    print_check_result("flash_attn_4warp      (v2 4-warp)", r2);
    printf("\n");

    cuMemFree(dQ); cuMemFree(dK); cuMemFree(dV); cuMemFree(dO);
    free(hQ); free(hK); free(hV); free(hO_v1); free(hO_v2); free(hRef); free(sBuf);

    // =========================================================
    // Performance benchmark — multi-head/batch
    // =========================================================
    printf("Performance (batch=%d, heads=%d, seq_len=%d):\n\n", batch_size, num_heads, seq_len);

    size_t total_elems = (size_t)batch_size * num_heads * seq_len * d_head;
    size_t total_bytes = total_elems * sizeof(float);

    CUdeviceptr dQm, dKm, dVm, dOm;
    CHECK_CU(cuMemAlloc(&dQm, total_bytes));
    CHECK_CU(cuMemAlloc(&dKm, total_bytes));
    CHECK_CU(cuMemAlloc(&dVm, total_bytes));
    CHECK_CU(cuMemAlloc(&dOm, total_bytes));

    // Fill K/V/Q with pattern (don't bother with exact values for perf test)
    CHECK_CU(cuMemsetD32(dQm, 0x3f000000, total_elems));  // 0.5f
    CHECK_CU(cuMemsetD32(dKm, 0x3f000000, total_elems));
    CHECK_CU(cuMemsetD32(dVm, 0x3f000000, total_elems));

    int warmup_iters = 5;
    int bench_iters  = 50;

    auto run_bench = [&](CUfunction fn, int grid_x, int block_x,
                         const char *label, int kv_iters) {
        void *args[] = { &dQm, &dKm, &dVm, &dOm, &seq_len, &num_heads, &scale };

        for (int i = 0; i < warmup_iters; i++) {
            CHECK_CU(cuLaunchKernel(fn,
                grid_x, num_heads, batch_size,
                block_x, 1, 1,
                0, NULL, args, NULL));
        }
        CHECK_CU(cuCtxSynchronize());

        float avg_ms;
        {
            BenchTimer timer;
            timer.start();
            for (int i = 0; i < bench_iters; i++) {
                CHECK_CU(cuLaunchKernel(fn,
                    grid_x, num_heads, batch_size,
                    block_x, 1, 1,
                    0, NULL, args, NULL));
            }
            avg_ms = timer.stop_ms() / bench_iters;
        }

        // Effective bandwidth: Q (1 read) + K × kv_iters + V × kv_iters + O (1 write)
        double bytes_total = ((double)1 + 2.0 * kv_iters + 1.0) * total_bytes;
        double bw_gb_s = (bytes_total / 1e9) / (avg_ms / 1000.0);

        // Ideal bandwidth (Q+K+V+O each read once)
        double ideal_bytes = 4.0 * total_bytes;
        double ideal_bw    = (ideal_bytes / 1e9) / (avg_ms / 1000.0);

        printf("  %-40s %7.3f ms   %6.1f GB/s  (ideal: %5.1f GB/s)\n",
               label, avg_ms, bw_gb_s, ideal_bw);
    };

    int kv_iters_v1 = seq_len / block_kv_v1;  // = seq_len/32
    int kv_iters_v2 = seq_len / block_kv_v2;  // = seq_len/64

    run_bench(fn_multihead, seq_len,           32, "flash_attn_multihead (1-warp, BKV=32)", kv_iters_v1);
    run_bench(fn_4warp,     seq_len/num_warps, 128,"flash_attn_4warp     (4-warp, BKV=64)", kv_iters_v2);

    printf("\nKey insight: ideal BW shows effective bandwidth if K/V tiles were L2-resident.\n");
    printf("4-warp version: 4× fewer K/V tile loads + 2× larger tiles = 8× less K/V traffic.\n");
    printf("If both are K/V-load-bound, expect 4-8× speedup.\n");

    cuMemFree(dQm); cuMemFree(dKm); cuMemFree(dVm); cuMemFree(dOm);
    cuModuleUnload(mod_v1); cuModuleUnload(mod_v2);
    cuCtxDestroy(cu_context);
    return 0;
}
