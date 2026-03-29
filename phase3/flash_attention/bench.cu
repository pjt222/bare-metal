/*
 * bench.cu — Flash Attention benchmark: correctness vs CPU + throughput
 *
 * Tests:
 *   1. Correctness: flash_attn_1warp output vs CPU numerically stable naive attention
 *   2. Performance: flash_attn_multihead throughput in GB/s at multiple seq_lens
 *
 * Build:
 *   nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../../phase2/common
 *
 * Usage:
 *   ./bench                     # default: single-head, seq_len=512
 *   ./bench 1024                # seq_len=1024
 *   ./bench 2048 8 8            # seq_len=2048, batch=8, num_heads=8
 *
 * Expected SASS (check with cuobjdump -sass flash_attn.sm_86.cubin):
 *   SHFL.BFLY  — warp dot product reduction (5 instructions, offsets 16,8,4,2,1)
 *   MUFU.EX2   — exp2f for attention weights (online softmax)
 *   MUFU.RCP   — __frcp_rn for final 1/sum normalization
 *   FMAX       — running max comparison across KV tiles
 *   FFMA       — weighted V accumulation (output_reg += weight * V_tile)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda.h>

#include "../../phase2/common/bench.h"
#include "../../phase2/common/check.h"

// -----------------------------------------------------------------------
// CPU reference: numerically stable scaled dot-product attention
//   O[q] = sum_k softmax(scale * Q[q] · K[k])_k * V[k]
// Uses standard two-pass softmax (max subtraction then normalize).
// -----------------------------------------------------------------------
static void cpu_attention(
    const float *Q,         // [seq_len × d_head]
    const float *K,         // [seq_len × d_head]
    const float *V,         // [seq_len × d_head]
    float       *O,         // [seq_len × d_head]  output
    float       *score_buf, // [seq_len] scratch buffer
    int seq_len,
    int d_head,
    float scale
) {
    for (int q_idx = 0; q_idx < seq_len; q_idx++) {
        const float *q_row = Q + (size_t)q_idx * d_head;

        // --- Pass 1: compute scores and find max ---
        float row_max = -3.402823466e+38f;
        for (int k_idx = 0; k_idx < seq_len; k_idx++) {
            const float *k_row = K + (size_t)k_idx * d_head;
            float dot = 0.0f;
            for (int d = 0; d < d_head; d++) {
                dot += q_row[d] * k_row[d];
            }
            score_buf[k_idx] = dot * scale;
            row_max = fmaxf(row_max, score_buf[k_idx]);
        }

        // --- Pass 2: exp(score - max) and sum ---
        float exp_sum = 0.0f;
        for (int k_idx = 0; k_idx < seq_len; k_idx++) {
            score_buf[k_idx] = expf(score_buf[k_idx] - row_max);
            exp_sum += score_buf[k_idx];
        }

        // --- Pass 3: normalize and accumulate weighted V ---
        float *o_row = O + (size_t)q_idx * d_head;
        for (int d = 0; d < d_head; d++) o_row[d] = 0.0f;

        float rcp_sum = 1.0f / exp_sum;
        for (int k_idx = 0; k_idx < seq_len; k_idx++) {
            float softmax_weight = score_buf[k_idx] * rcp_sum;
            const float *v_row = V + (size_t)k_idx * d_head;
            for (int d = 0; d < d_head; d++) {
                o_row[d] += softmax_weight * v_row[d];
            }
        }
    }
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------
int main(int argc, char **argv) {
    int seq_len   = (argc > 1) ? atoi(argv[1]) : 512;
    int batch_size = (argc > 2) ? atoi(argv[2]) : 1;
    int num_heads  = (argc > 3) ? atoi(argv[3]) : 1;

    const int d_head = 64;   // must match D_HEAD in flash_attn.cu
    const int block_kv = 32; // must match BLOCK_KV in flash_attn.cu

    // seq_len must be multiple of block_kv for this kernel
    if (seq_len % block_kv != 0) {
        fprintf(stderr, "ERROR: seq_len=%d must be a multiple of BLOCK_KV=%d\n",
                seq_len, block_kv);
        return EXIT_FAILURE;
    }

    float scale = 1.0f / sqrtf((float)d_head);

    printf("=== Flash Attention Benchmark — Online Softmax ===\n");
    printf("seq_len=%d  d_head=%d  batch=%d  heads=%d  scale=%.4f\n\n",
           seq_len, d_head, batch_size, num_heads, scale);

    CHECK_CU(cuInit(0));
    CUdevice  cu_device;
    CHECK_CU(cuDeviceGet(&cu_device, 0));

    char device_name[256];
    CHECK_CU(cuDeviceGetName(device_name, sizeof(device_name), cu_device));
    printf("Device: %s\n\n", device_name);

    CUcontext cu_context;
    CHECK_CU(cuCtxCreate(&cu_context, 0, cu_device));

    // --- Load kernel cubin ---
    CUmodule   flash_module;
    CUfunction single_head_func, multihead_func;

    CUresult load_result = cuModuleLoad(&flash_module, "flash_attn.sm_86.cubin");
    if (load_result != CUDA_SUCCESS) {
        const char *err_str = nullptr;
        cuGetErrorString(load_result, &err_str);
        fprintf(stderr, "Cannot load flash_attn.sm_86.cubin: %s\n", err_str);
        fprintf(stderr, "Build with: nvcc --cubin -arch=sm_86 -O2 -o flash_attn.sm_86.cubin flash_attn.cu\n");
        cuCtxDestroy(cu_context);
        return EXIT_FAILURE;
    }
    CHECK_CU(cuModuleGetFunction(&single_head_func, flash_module, "flash_attn_1warp"));
    CHECK_CU(cuModuleGetFunction(&multihead_func,   flash_module, "flash_attn_multihead"));
    printf("Flash Attention kernels loaded.\n\n");

    // =========================================================
    // Part 1: Correctness test (single-head, seq_len)
    // =========================================================
    size_t single_head_elements = (size_t)seq_len * d_head;
    size_t single_head_bytes    = single_head_elements * sizeof(float);

    float *host_Q     = (float *)malloc(single_head_bytes);
    float *host_K     = (float *)malloc(single_head_bytes);
    float *host_V     = (float *)malloc(single_head_bytes);
    float *host_O_gpu = (float *)malloc(single_head_bytes);
    float *host_O_ref = (float *)malloc(single_head_bytes);
    float *score_buf  = (float *)malloc(seq_len * sizeof(float));

    fill_random(host_Q, single_head_elements, 42);
    fill_random(host_K, single_head_elements, 43);
    fill_random(host_V, single_head_elements, 44);

    // CPU reference (numerically stable naive attention)
    printf("Computing CPU reference...\n");
    cpu_attention(host_Q, host_K, host_V, host_O_ref, score_buf,
                  seq_len, d_head, scale);
    printf("CPU reference done.\n\n");

    // Allocate GPU buffers
    CUdeviceptr dev_Q, dev_K, dev_V, dev_O;
    CHECK_CU(cuMemAlloc(&dev_Q, single_head_bytes));
    CHECK_CU(cuMemAlloc(&dev_K, single_head_bytes));
    CHECK_CU(cuMemAlloc(&dev_V, single_head_bytes));
    CHECK_CU(cuMemAlloc(&dev_O, single_head_bytes));

    CHECK_CU(cuMemcpyHtoD(dev_Q, host_Q, single_head_bytes));
    CHECK_CU(cuMemcpyHtoD(dev_K, host_K, single_head_bytes));
    CHECK_CU(cuMemcpyHtoD(dev_V, host_V, single_head_bytes));

    // Launch flash_attn_1warp: one warp per query row
    //   Grid:  (seq_len, 1, 1)
    //   Block: (WARP_SIZE=32, 1, 1)
    {
        void *args[] = { &dev_Q, &dev_K, &dev_V, &dev_O, &seq_len, &scale };
        CHECK_CU(cuLaunchKernel(single_head_func,
            seq_len, 1, 1,  // grid
            32, 1, 1,        // block: exactly one warp
            0, NULL, args, NULL));
        CHECK_CU(cuCtxSynchronize());
    }
    CHECK_CU(cuMemcpyDtoH(host_O_gpu, dev_O, single_head_bytes));

    printf("Correctness (flash_attn_1warp vs CPU naive):\n");
    auto correctness = check_fp32(host_O_gpu, host_O_ref, single_head_elements, 1e-3f, 1e-3f);
    print_check_result("flash_attn_1warp", correctness);
    printf("\n");

    // =========================================================
    // Part 2: Performance benchmark (multihead)
    // =========================================================
    printf("Performance benchmark (flash_attn_multihead):\n");
    printf("  batch=%d  heads=%d  seq_len=%d  d_head=%d\n\n",
           batch_size, num_heads, seq_len, d_head);

    size_t multi_elements = (size_t)batch_size * num_heads * seq_len * d_head;
    size_t multi_bytes    = multi_elements * sizeof(float);

    CUdeviceptr dev_Q_multi, dev_K_multi, dev_V_multi, dev_O_multi;
    CHECK_CU(cuMemAlloc(&dev_Q_multi, multi_bytes));
    CHECK_CU(cuMemAlloc(&dev_K_multi, multi_bytes));
    CHECK_CU(cuMemAlloc(&dev_V_multi, multi_bytes));
    CHECK_CU(cuMemAlloc(&dev_O_multi, multi_bytes));

    // Fill with random data (reuse Q/K/V host arrays for pattern)
    float *host_multi = (float *)malloc(multi_bytes);
    fill_random(host_multi, multi_elements, 99);
    CHECK_CU(cuMemcpyHtoD(dev_Q_multi, host_multi, multi_bytes));
    CHECK_CU(cuMemcpyHtoD(dev_K_multi, host_multi, multi_bytes));
    CHECK_CU(cuMemcpyHtoD(dev_V_multi, host_multi, multi_bytes));
    free(host_multi);

    // Grid: (seq_len, num_heads, batch_size)
    // Block: (32, 1, 1)
    void *multi_args[] = {
        &dev_Q_multi, &dev_K_multi, &dev_V_multi, &dev_O_multi,
        &seq_len, &num_heads, &scale
    };

    int warmup_iters = 5;
    int bench_iters  = 50;

    // Warmup
    for (int iter = 0; iter < warmup_iters; iter++) {
        CHECK_CU(cuLaunchKernel(multihead_func,
            seq_len, num_heads, batch_size,
            32, 1, 1,
            0, NULL, multi_args, NULL));
    }
    CHECK_CU(cuCtxSynchronize());

    // Benchmark
    float avg_ms;
    {
        BenchTimer timer;
        timer.start();
        for (int iter = 0; iter < bench_iters; iter++) {
            CHECK_CU(cuLaunchKernel(multihead_func,
                seq_len, num_heads, batch_size,
                32, 1, 1,
                0, NULL, multi_args, NULL));
        }
        avg_ms = timer.stop_ms() / bench_iters;
    }

    // Bandwidth: read 3 tensors (Q, K, V) + write 1 (O)
    // Flash attention reads Q once, K and V multiple times (seq_len/BLOCK_KV iterations each)
    // True reads: Q=1 pass, K=seq_len/BLOCK_KV passes, V=seq_len/BLOCK_KV passes
    // For seq_len=512: 16 KV iterations → K read 16×, V read 16×
    int kv_iterations = seq_len / block_kv;
    double bytes_read  = (double)multi_bytes         // Q: read once
                       + (double)multi_bytes * kv_iterations  // K: read kv_iterations times
                       + (double)multi_bytes * kv_iterations; // V: read kv_iterations times
    double bytes_write = (double)multi_bytes;         // O: write once
    double total_bytes = bytes_read + bytes_write;
    double bandwidth_gb_s = (total_bytes / 1e9) / (avg_ms / 1000.0);

    // Effective compute: for each query, seq_len dot products of d_head dims (2 ops each)
    // plus softmax (exp + weighted sum)
    double flops = (double)batch_size * num_heads * seq_len * (
        (double)seq_len * d_head * 2 +    // QK^T: seq_len dot products
        (double)seq_len * 5 +             // softmax per position: exp + rescale + sum
        (double)seq_len * d_head * 2      // PV: seq_len weighted accumulations
    );
    double gflops = flops / (avg_ms / 1000.0) / 1e9;

    printf("  %-30s  %7.3f ms   %7.1f GB/s   %7.1f GFLOPS\n",
           "flash_attn_multihead", avg_ms, bandwidth_gb_s, gflops);

    // Also show the "idealized" bandwidth if we only counted input/output once
    double ideal_bytes = 4.0 * multi_bytes;  // Q + K + V + O
    double ideal_bw = (ideal_bytes / 1e9) / (avg_ms / 1000.0);
    printf("  (ideal BW if K/V cached: %.1f GB/s — memory ceiling: 608 GB/s)\n", ideal_bw);

    printf("\nTo inspect SASS:\n");
    printf("  cuobjdump -sass flash_attn.sm_86.cubin | grep -E 'SHFL|MUFU|FMAX|FFMA'\n");
    printf("  SHFL.BFLY  — 5× per score (offsets 16,8,4,2,1) × BLOCK_KV=%d scores = %d SHFL per Q row\n",
           block_kv, 5 * block_kv);
    printf("  MUFU.EX2   — 1 per weight + 1 for rescale factor\n");
    printf("  MUFU.RCP   — 1 per Q row (final normalization)\n");

    // --- Cleanup ---
    cuMemFree(dev_Q);
    cuMemFree(dev_K);
    cuMemFree(dev_V);
    cuMemFree(dev_O);
    cuMemFree(dev_Q_multi);
    cuMemFree(dev_K_multi);
    cuMemFree(dev_V_multi);
    cuMemFree(dev_O_multi);
    cuModuleUnload(flash_module);
    cuCtxDestroy(cu_context);

    free(host_Q); free(host_K); free(host_V);
    free(host_O_gpu); free(host_O_ref); free(score_buf);

    return 0;
}
