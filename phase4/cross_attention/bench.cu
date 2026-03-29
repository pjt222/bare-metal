/*
 * bench.cu — Cross-Attention benchmark: correctness + throughput
 *
 * Tests cross_attn_br16 across typical Stable Diffusion configurations:
 *
 *   SD 1.x/2.x cross-attention resolutions:
 *     8×8   feature map  (deepest UNet layer): seq_q =   64, seq_kv = 77 (CLIP)
 *     16×16 feature map:                       seq_q =  256, seq_kv = 77
 *     32×32 feature map:                       seq_q = 1024, seq_kv = 77
 *     64×64 feature map:                       seq_q = 4096, seq_kv = 77
 *
 *   SD-XL and long-context attention:
 *     32×32 with longer context:               seq_q = 1024, seq_kv = 512
 *
 * Build:
 *   nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../../phase2/common
 *
 * Usage:
 *   ./bench                      # default: seq_q=256, seq_kv=77, heads=8
 *   ./bench 1024 77 8            # SD 32×32 feature map, CLIP context
 *   ./bench 256 512 8            # long context (SD-XL style)
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>
#include <cuda_fp16.h>

#include "../../phase2/common/bench.h"
#include "../../phase2/common/check.h"

// -----------------------------------------------------------------------
// CPU reference: naive cross-attention (FP32)
// For a single head: A = softmax(Q @ K^T / sqrt(d)) @ V
// -----------------------------------------------------------------------
static void cpu_cross_attn(
    const float *Q, const float *K, const float *V, float *O,
    float *score_buf,
    int seq_q, int seq_kv, int d_head, float scale
) {
    for (int q = 0; q < seq_q; q++) {
        // Dot Q[q] with all K[k] → raw scores
        float row_max = -3.402823466e+38f;
        for (int k = 0; k < seq_kv; k++) {
            float dot = 0.0f;
            for (int d = 0; d < d_head; d++) dot += Q[q*d_head+d] * K[k*d_head+d];
            score_buf[k] = dot * scale;
            row_max = fmaxf(row_max, score_buf[k]);
        }

        // Softmax
        float sum = 0.0f;
        for (int k = 0; k < seq_kv; k++) {
            score_buf[k] = expf(score_buf[k] - row_max);
            sum += score_buf[k];
        }
        float rcp = 1.0f / sum;

        // Weighted sum of V
        for (int d = 0; d < d_head; d++) O[q*d_head+d] = 0.0f;
        for (int k = 0; k < seq_kv; k++) {
            float weight = score_buf[k] * rcp;
            for (int d = 0; d < d_head; d++) O[q*d_head+d] += weight * V[k*d_head+d];
        }
    }
}

// FP32 → FP16 host conversion
static void fp32_to_fp16(const float *src, __half *dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = __float2half(src[i]);
}

int main(int argc, char **argv) {
    int seq_q   = (argc > 1) ? atoi(argv[1]) : 256;   // image tokens (H*W)
    int seq_kv  = (argc > 2) ? atoi(argv[2]) : 77;    // text tokens (CLIP)
    int num_heads = (argc > 3) ? atoi(argv[3]) : 8;
    int batch   = 1;

    const int d_head    = 64;
    const int Br_block  = 64;   // must match cross_attn.cu

    if (seq_q % Br_block != 0) {
        // Pad seq_q to next multiple of Br_block for grid calculation
        int padded = ((seq_q + Br_block - 1) / Br_block) * Br_block;
        printf("Note: seq_q=%d padded to %d for kernel grid alignment\n", seq_q, padded);
        seq_q = padded;
    }

    float scale = 1.0f / sqrtf((float)d_head);

    printf("=== Cross-Attention (Flash, HMMA Br=16) — Q×K^T + Softmax + ×V ===\n");
    printf("seq_q=%d (image)  seq_kv=%d (text)  heads=%d  d_head=%d\n\n",
           seq_q, seq_kv, num_heads, d_head);

    CHECK_CU(cuInit(0));
    CUdevice cu_dev; CHECK_CU(cuDeviceGet(&cu_dev, 0));
    char devname[256]; CHECK_CU(cuDeviceGetName(devname, sizeof(devname), cu_dev));
    printf("Device: %s\n\n", devname);

    CUcontext ctx; CHECK_CU(cuCtxCreate(&ctx, 0, cu_dev));

    CUmodule mod;
    CUfunction fn;
    if (cuModuleLoad(&mod, "cross_attn.sm_86.cubin") != CUDA_SUCCESS) {
        fprintf(stderr, "Cannot load cross_attn.sm_86.cubin\n"); return 1;
    }
    CHECK_CU(cuModuleGetFunction(&fn, mod, "cross_attn_br16"));

    // Shared memory: same 48 KB calculation as flash_attn_br16
    size_t smem_bytes = 2 * Br_block * d_head * sizeof(short)      // K+V tiles FP16
                      + Br_block * Br_block * sizeof(float)         // smem_work
                      + Br_block * d_head   * sizeof(float);        // smem_pv
    CHECK_CU(cuFuncSetAttribute(fn,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem_bytes));
    printf("Shared memory per block: %zu bytes (%.1f KB)\n\n",
           smem_bytes, smem_bytes / 1024.0f);

    // =========================================================
    // Correctness test (single head, FP32 CPU reference)
    // =========================================================
    size_t q_elems  = (size_t)seq_q  * d_head;
    size_t kv_elems = (size_t)seq_kv * d_head;

    float *hQf  = (float*)malloc(q_elems  * sizeof(float));
    float *hKf  = (float*)malloc(kv_elems * sizeof(float));
    float *hVf  = (float*)malloc(kv_elems * sizeof(float));
    float *hRef = (float*)malloc(q_elems  * sizeof(float));
    float *hOut = (float*)malloc(q_elems  * sizeof(float));
    float *sBuf = (float*)malloc(seq_kv   * sizeof(float));

    fill_random(hQf, q_elems,  10);
    fill_random(hKf, kv_elems, 11);
    fill_random(hVf, kv_elems, 12);

    printf("Computing CPU reference (single head)...\n");
    cpu_cross_attn(hQf, hKf, hVf, hRef, sBuf, seq_q, seq_kv, d_head, scale);
    printf("Done.\n\n");

    // Convert to FP16 for GPU kernel
    __half *hQh = (__half*)malloc(q_elems  * sizeof(__half));
    __half *hKh = (__half*)malloc(kv_elems * sizeof(__half));
    __half *hVh = (__half*)malloc(kv_elems * sizeof(__half));
    fp32_to_fp16(hQf, hQh, q_elems);
    fp32_to_fp16(hKf, hKh, kv_elems);
    fp32_to_fp16(hVf, hVh, kv_elems);

    // Device buffers (single head, batch=1)
    CUdeviceptr dQ, dK, dV, dO;
    CHECK_CU(cuMemAlloc(&dQ, q_elems  * sizeof(__half)));
    CHECK_CU(cuMemAlloc(&dK, kv_elems * sizeof(__half)));
    CHECK_CU(cuMemAlloc(&dV, kv_elems * sizeof(__half)));
    CHECK_CU(cuMemAlloc(&dO, q_elems  * sizeof(float)));

    CHECK_CU(cuMemcpyHtoD(dQ, hQh, q_elems  * sizeof(__half)));
    CHECK_CU(cuMemcpyHtoD(dK, hKh, kv_elems * sizeof(__half)));
    CHECK_CU(cuMemcpyHtoD(dV, hVh, kv_elems * sizeof(__half)));

    int nh1 = 1;  // heads=1 for correctness test
    void *args_single[] = { &dQ, &dK, &dV, &dO, &seq_q, &seq_kv, &nh1, &scale };
    CHECK_CU(cuMemsetD32(dO, 0, q_elems));
    CHECK_CU(cuLaunchKernel(fn,
        seq_q / Br_block, 1, 1,
        128, 1, 1,
        (unsigned)smem_bytes, NULL, args_single, NULL));
    CHECK_CU(cuCtxSynchronize());
    CHECK_CU(cuMemcpyDtoH(hOut, dO, q_elems * sizeof(float)));

    printf("Correctness (vs CPU FP32 naive, single head):\n");
    // FP16 quantization introduces ~2e-3 max_abs error (same as flash_attn_br16)
    auto correctness = check_fp32(hOut, hRef, q_elems, 1e-2f, 1.0f);
    print_check_result("cross_attn_br16 (FP16 HMMA)", correctness);
    printf("\n");

    cuMemFree(dQ); cuMemFree(dK); cuMemFree(dV); cuMemFree(dO);
    free(hQf); free(hKf); free(hVf); free(hRef); free(hOut); free(sBuf);
    free(hQh); free(hKh); free(hVh);

    // =========================================================
    // Performance benchmark (multi-head, batch)
    // =========================================================
    printf("Performance (batch=%d, heads=%d):\n\n", batch, num_heads);

    size_t tot_q  = (size_t)batch * num_heads * seq_q  * d_head;
    size_t tot_kv = (size_t)batch * num_heads * seq_kv * d_head;

    CUdeviceptr dQm, dKm, dVm, dOm;
    CHECK_CU(cuMemAlloc(&dQm, tot_q  * sizeof(__half)));
    CHECK_CU(cuMemAlloc(&dKm, tot_kv * sizeof(__half)));
    CHECK_CU(cuMemAlloc(&dVm, tot_kv * sizeof(__half)));
    CHECK_CU(cuMemAlloc(&dOm, tot_q  * sizeof(float)));

    // Initialize device buffers with small values (0.5 in FP16 = 0x3800)
    CHECK_CU(cuMemsetD16(dQm, 0x3800, tot_q));
    CHECK_CU(cuMemsetD16(dKm, 0x3800, tot_kv));
    CHECK_CU(cuMemsetD16(dVm, 0x3800, tot_kv));
    CHECK_CU(cuMemsetD32(dOm, 0, tot_q));

    void *args_multi[] = { &dQm, &dKm, &dVm, &dOm,
                           &seq_q, &seq_kv, &num_heads, &scale };

    int warmup = 5, bench_iters = 100;
    for (int i = 0; i < warmup; i++) {
        CHECK_CU(cuLaunchKernel(fn,
            seq_q / Br_block, num_heads, batch,
            128, 1, 1,
            (unsigned)smem_bytes, NULL, args_multi, NULL));
    }
    CHECK_CU(cuCtxSynchronize());

    float avg_ms;
    {
        BenchTimer timer;
        timer.start();
        for (int i = 0; i < bench_iters; i++) {
            CHECK_CU(cuLaunchKernel(fn,
                seq_q / Br_block, num_heads, batch,
                128, 1, 1,
                (unsigned)smem_bytes, NULL, args_multi, NULL));
        }
        avg_ms = timer.stop_ms() / bench_iters;
    }

    // FLOPs: Q @ K^T = 2 * seq_q * seq_kv * d_head per head per batch
    //         attn @ V = 2 * seq_q * seq_kv * d_head per head per batch
    double flops = 2.0 * 2.0 * batch * num_heads * seq_q * seq_kv * d_head;
    double gflops = flops / 1e9 / (avg_ms / 1000.0);

    // Ideal bandwidth: Q read once + K,V read seq_q/Bc times each + O written once
    // Counting Q + K + V + O (each once for ideal):
    double ideal_bytes = sizeof(__half) * (double)(tot_q + 2*tot_kv)
                       + sizeof(float)  * (double)tot_q;
    double ideal_gb_s  = ideal_bytes / 1e9 / (avg_ms / 1000.0);

    printf("  %-45s %7.3f ms\n", "cross_attn_br16", avg_ms);
    printf("  FLOPS:  %.2f GFLOPS (%.1f%% of %.0f TFLOPS FP16 tensor peak)\n",
           gflops, gflops / 174000.0 * 100.0, 174.0);
    printf("  Ideal BW (Q+K+V+O once): %.1f GB/s (peak: 608 GB/s)\n", ideal_gb_s);

    // Compare against naive (theoretical)
    printf("\nKey asymmetry vs self-attention at same seq_q:\n");
    printf("  seq_q=%d tokens (image), seq_kv=%d tokens (text)\n", seq_q, seq_kv);
    printf("  KV tile iterations: ceil(%d / 64) = %d   (vs %d for self-attn)\n",
           seq_kv, (seq_kv + 63) / 64, (seq_q + 63) / 64);
    printf("  When seq_kv << seq_q: cross-attn is FASTER than self-attn!\n");
    printf("    (e.g., seq_q=1024, seq_kv=77 → 2 KV iters vs 16 for self-attn)\n");

    printf("\nSASS inspection:\n");
    printf("  cuobjdump -sass cross_attn.sm_86.cubin | grep HMMA | wc -l → 64\n");
    printf("  cuobjdump -sass cross_attn.sm_86.cubin | grep SHFL.BFLY    → 5 rounds per row\n");
    printf("  cuobjdump -sass cross_attn.sm_86.cubin | grep MUFU.EX2     → weights + rescale\n");

    cuMemFree(dQm); cuMemFree(dKm); cuMemFree(dVm); cuMemFree(dOm);
    cuModuleUnload(mod); cuCtxDestroy(ctx);
    return 0;
}
