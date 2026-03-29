/*
 * bench_split_q.cu — Benchmark: flash_attn_split_q vs flash_attn_br16
 *
 * Tests correctness of split-Q against CPU reference, then benchmarks
 * split-Q at multiple split counts versus the br16 baseline.
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o flash_br16.sm_86.cubin flash_attn_br16.cu
 *   nvcc --cubin -arch=sm_86 -O2 -o flash_split_q.sm_86.cubin flash_attn_split_q.cu
 *   nvcc -arch=sm_86 -O2 -o bench_split_q bench_split_q.cu -lcuda -I../../phase2/common
 *
 * Usage:
 *   ./bench_split_q                   # seq=1024, batch=8, heads=8
 *   ./bench_split_q 2048 4 8          # seq=2048, batch=4, heads=8
 *   ./bench_split_q 1024 8 8 4        # seq=1024, batch=8, heads=8, num_splits=4 (single test)
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <cuda_fp16.h>

#include "../../phase2/common/bench.h"
#include "../../phase2/common/check.h"

// -----------------------------------------------------------------------
// CPU reference: numerically stable scaled dot-product attention (FP32)
// -----------------------------------------------------------------------
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

static void fp32_to_fp16(const float *src, __half *dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = __float2half(src[i]);
}

int main(int argc, char **argv) {
    int seq       = (argc > 1) ? atoi(argv[1]) : 1024;
    int batch     = (argc > 2) ? atoi(argv[2]) : 8;
    int num_heads = (argc > 3) ? atoi(argv[3]) : 8;
    int force_splits = (argc > 4) ? atoi(argv[4]) : 0;  // 0 = sweep

    const int d        = 64;
    const int Br_block = 64;
    const int Bc_val   = 64;

    if (seq % Br_block != 0) {
        fprintf(stderr, "seq=%d must be divisible by Br_block=%d\n", seq, Br_block);
        return 1;
    }

    float scale = 1.0f / sqrtf((float)d);
    int num_kv_tiles = seq / Bc_val;

    printf("=== Split-Q Flash Attention vs br16 Baseline ===\n");
    printf("seq=%d  d=%d  batch=%d  heads=%d  kv_tiles=%d\n\n", seq, d, batch, num_heads, num_kv_tiles);

    CHECK_CU(cuInit(0));
    CUdevice cu_dev; CHECK_CU(cuDeviceGet(&cu_dev, 0));
    char devname[256]; CHECK_CU(cuDeviceGetName(devname, sizeof(devname), cu_dev));
    printf("Device: %s\n\n", devname);

    CUcontext ctx; CHECK_CU(cuCtxCreate(&ctx, 0, cu_dev));

    // ---- Load cubins ----
    CUmodule mod_br16, mod_split;
    CUfunction fn_br16, fn_split_q, fn_reduce;

    if (cuModuleLoad(&mod_br16, "flash_br16.sm_86.cubin") != CUDA_SUCCESS) {
        fprintf(stderr, "Cannot load flash_br16.sm_86.cubin\n");
        fprintf(stderr, "Build: nvcc --cubin -arch=sm_86 -O2 -o flash_br16.sm_86.cubin flash_attn_br16.cu\n");
        return 1;
    }
    if (cuModuleLoad(&mod_split, "flash_split_q.sm_86.cubin") != CUDA_SUCCESS) {
        fprintf(stderr, "Cannot load flash_split_q.sm_86.cubin\n");
        fprintf(stderr, "Build: nvcc --cubin -arch=sm_86 -O2 -o flash_split_q.sm_86.cubin flash_attn_split_q.cu\n");
        return 1;
    }
    CHECK_CU(cuModuleGetFunction(&fn_br16,    mod_br16,  "flash_attn_br16"));
    CHECK_CU(cuModuleGetFunction(&fn_split_q, mod_split, "flash_attn_split_q"));
    CHECK_CU(cuModuleGetFunction(&fn_reduce,  mod_split, "flash_attn_split_q_reduce"));

    // Set shared memory for br16 and split-Q (both use 48 KB)
    size_t smem_size = 2 * Bc_val * d * sizeof(short)       // K+V tiles FP16
                     + Br_block * Bc_val * sizeof(float)     // smem_work FP32
                     + Br_block * d * sizeof(float);         // smem_pv FP32
    CHECK_CU(cuFuncSetAttribute(fn_br16,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem_size));
    CHECK_CU(cuFuncSetAttribute(fn_split_q,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem_size));
    printf("Shared memory per block: %zu bytes (%.1f KB)\n\n", smem_size, smem_size/1024.0);

    // =========================================================
    // Correctness test (single head, single batch)
    // =========================================================
    {
        size_t n_elems = (size_t)seq * d;
        size_t n_bytes_f32 = n_elems * sizeof(float);
        size_t n_bytes_f16 = n_elems * sizeof(__half);

        float *hQf  = (float*)malloc(n_bytes_f32);
        float *hKf  = (float*)malloc(n_bytes_f32);
        float *hVf  = (float*)malloc(n_bytes_f32);
        float *hRef = (float*)malloc(n_bytes_f32);
        float *hOut = (float*)malloc(n_bytes_f32);
        float *sBuf = (float*)malloc(seq * sizeof(float));

        fill_random(hQf, n_elems, 30);
        fill_random(hKf, n_elems, 31);
        fill_random(hVf, n_elems, 32);

        printf("Computing CPU reference (seq=%d, single head)...\n", seq);
        cpu_attention(hQf, hKf, hVf, hRef, sBuf, seq, d, scale);
        printf("Done.\n\n");

        __half *hQh = (__half*)malloc(n_bytes_f16);
        __half *hKh = (__half*)malloc(n_bytes_f16);
        __half *hVh = (__half*)malloc(n_bytes_f16);
        fp32_to_fp16(hQf, hQh, n_elems);
        fp32_to_fp16(hKf, hKh, n_elems);
        fp32_to_fp16(hVf, hVh, n_elems);

        CUdeviceptr dQh, dKh, dVh, dO;
        CHECK_CU(cuMemAlloc(&dQh, n_bytes_f16));
        CHECK_CU(cuMemAlloc(&dKh, n_bytes_f16));
        CHECK_CU(cuMemAlloc(&dVh, n_bytes_f16));
        CHECK_CU(cuMemAlloc(&dO,  n_bytes_f32));

        CHECK_CU(cuMemcpyHtoD(dQh, hQh, n_bytes_f16));
        CHECK_CU(cuMemcpyHtoD(dKh, hKh, n_bytes_f16));
        CHECK_CU(cuMemcpyHtoD(dVh, hVh, n_bytes_f16));

        int one_head = 1;

        // Test br16
        {
            CHECK_CU(cuMemsetD32(dO, 0, n_elems));
            void *args[] = { &dQh, &dKh, &dVh, &dO, &seq, &one_head, &scale };
            CHECK_CU(cuLaunchKernel(fn_br16,
                seq / Br_block, 1, 1,   128, 1, 1,
                (unsigned)smem_size, NULL, args, NULL));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(hOut, dO, n_bytes_f32));

            printf("Correctness (vs CPU FP32 naive):\n");
            auto r = check_fp32(hOut, hRef, n_elems, 1e-2f, 1e-0f);
            print_check_result("  br16 baseline (FP16 HMMA)  ", r);
        }

        // Test split-Q at various split counts
        int test_splits[] = {2, 4, num_kv_tiles};
        int num_test_configs = 3;

        for (int ti = 0; ti < num_test_configs; ti++) {
            int ns = test_splits[ti];
            if (ns > num_kv_tiles) ns = num_kv_tiles;

            // Allocate partial buffers
            size_t partial_ml_elems = (size_t)ns * 1 * 1 * seq;  // single batch, single head
            size_t partial_o_elems  = partial_ml_elems * d;

            CUdeviceptr dPm, dPl, dPo;
            CHECK_CU(cuMemAlloc(&dPm, partial_ml_elems * sizeof(float)));
            CHECK_CU(cuMemAlloc(&dPl, partial_ml_elems * sizeof(float)));
            CHECK_CU(cuMemAlloc(&dPo, partial_o_elems  * sizeof(float)));

            // Launch split-Q kernel
            void *split_args[] = { &dQh, &dKh, &dVh, &dPo, &dPm, &dPl, &seq, &one_head, &ns, &scale };
            CHECK_CU(cuLaunchKernel(fn_split_q,
                ns, 1, 1,   128, 1, 1,
                (unsigned)smem_size, NULL, split_args, NULL));
            CHECK_CU(cuCtxSynchronize());

            // Launch reduce kernel
            CHECK_CU(cuMemsetD32(dO, 0, n_elems));
            void *reduce_args[] = { &dPo, &dPm, &dPl, &dO, &seq, &one_head, &ns };
            CHECK_CU(cuLaunchKernel(fn_reduce,
                seq / Br_block, 1, 1,   128, 1, 1,
                0, NULL, reduce_args, NULL));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(hOut, dO, n_bytes_f32));

            char label[64];
            snprintf(label, sizeof(label), "  split-Q (splits=%d)", ns);
            auto r = check_fp32(hOut, hRef, n_elems, 1e-2f, 1e-0f);
            print_check_result(label, r);

            cuMemFree(dPm); cuMemFree(dPl); cuMemFree(dPo);
        }
        printf("\n");

        cuMemFree(dQh); cuMemFree(dKh); cuMemFree(dVh); cuMemFree(dO);
        free(hQf); free(hKf); free(hVf); free(hRef); free(hOut); free(sBuf);
        free(hQh); free(hKh); free(hVh);
    }

    // =========================================================
    // Performance benchmark (multi-head, multi-batch)
    // =========================================================
    printf("Performance (batch=%d, heads=%d, seq=%d):\n\n", batch, num_heads, seq);

    size_t total_elems_f16 = (size_t)batch * num_heads * seq * d;
    size_t total_elems_f32 = total_elems_f16;

    CUdeviceptr dQm, dKm, dVm, dOm;
    CHECK_CU(cuMemAlloc(&dQm, total_elems_f16 * sizeof(__half)));
    CHECK_CU(cuMemAlloc(&dKm, total_elems_f16 * sizeof(__half)));
    CHECK_CU(cuMemAlloc(&dVm, total_elems_f16 * sizeof(__half)));
    CHECK_CU(cuMemAlloc(&dOm, total_elems_f32 * sizeof(float)));

    // Fill with 0.5 in FP16
    CHECK_CU(cuMemsetD16(dQm, 0x3800, total_elems_f16));
    CHECK_CU(cuMemsetD16(dKm, 0x3800, total_elems_f16));
    CHECK_CU(cuMemsetD16(dVm, 0x3800, total_elems_f16));

    int warmup_iters = 5;
    int bench_iters  = 50;

    // FLOPS formula: batch * heads * seq * (2*seq*d [QK^T] + 5*seq [softmax] + 2*seq*d [PV])
    double total_flops = (double)batch * num_heads * seq * (
        (double)seq * d * 2 +
        (double)seq * 5 +
        (double)seq * d * 2
    );

    // ---- Benchmark br16 baseline ----
    {
        void *args[] = { &dQm, &dKm, &dVm, &dOm, &seq, &num_heads, &scale };

        for (int i = 0; i < warmup_iters; i++)
            CHECK_CU(cuLaunchKernel(fn_br16,
                seq / Br_block, num_heads, batch,   128, 1, 1,
                (unsigned)smem_size, NULL, args, NULL));
        CHECK_CU(cuCtxSynchronize());

        BenchTimer timer;
        timer.start();
        for (int i = 0; i < bench_iters; i++)
            CHECK_CU(cuLaunchKernel(fn_br16,
                seq / Br_block, num_heads, batch,   128, 1, 1,
                (unsigned)smem_size, NULL, args, NULL));
        float br16_ms = timer.stop_ms() / bench_iters;
        double br16_gflops = total_flops / (br16_ms / 1000.0) / 1e9;

        printf("  %-45s %7.3f ms  %7.1f GFLOPS\n", "br16 baseline", br16_ms, br16_gflops);

        // ---- Benchmark split-Q at various split counts ----
        int sweep_splits[5];
        int num_sweep = 0;

        if (force_splits > 0) {
            sweep_splits[0] = force_splits;
            num_sweep = 1;
        } else {
            // Sweep: 2, 4, 8, num_kv_tiles (if unique)
            sweep_splits[num_sweep++] = 2;
            if (num_kv_tiles >= 4)  sweep_splits[num_sweep++] = 4;
            if (num_kv_tiles >= 8)  sweep_splits[num_sweep++] = 8;
            if (num_kv_tiles > 8)   sweep_splits[num_sweep++] = num_kv_tiles;
        }

        for (int si = 0; si < num_sweep; si++) {
            int ns = sweep_splits[si];
            if (ns > num_kv_tiles) ns = num_kv_tiles;

            size_t partial_ml_elems = (size_t)ns * batch * num_heads * seq;
            size_t partial_o_elems  = partial_ml_elems * d;
            size_t partial_ml_bytes = partial_ml_elems * sizeof(float);
            size_t partial_o_bytes  = partial_o_elems  * sizeof(float);

            // Check VRAM availability
            size_t total_partial = 2 * partial_ml_bytes + partial_o_bytes;
            size_t free_mem = 0, total_mem = 0;
            cuMemGetInfo(&free_mem, &total_mem);
            if (total_partial > free_mem * 0.8) {
                printf("  split-Q (splits=%-3d) — skipped (%.0f MB partial > %.0f MB free)\n",
                       ns, total_partial / 1e6, free_mem / 1e6);
                continue;
            }

            CUdeviceptr dPm, dPl, dPo;
            CHECK_CU(cuMemAlloc(&dPm, partial_ml_bytes));
            CHECK_CU(cuMemAlloc(&dPl, partial_ml_bytes));
            CHECK_CU(cuMemAlloc(&dPo, partial_o_bytes));

            void *split_args[] = { &dQm, &dKm, &dVm, &dPo, &dPm, &dPl, &seq, &num_heads, &ns, &scale };
            void *reduce_args[] = { &dPo, &dPm, &dPl, &dOm, &seq, &num_heads, &ns };

            // Warmup
            for (int i = 0; i < warmup_iters; i++) {
                CHECK_CU(cuLaunchKernel(fn_split_q,
                    ns, num_heads, batch,   128, 1, 1,
                    (unsigned)smem_size, NULL, split_args, NULL));
                CHECK_CU(cuLaunchKernel(fn_reduce,
                    seq / Br_block, num_heads, batch,   128, 1, 1,
                    0, NULL, reduce_args, NULL));
            }
            CHECK_CU(cuCtxSynchronize());

            // Benchmark (both kernels together)
            BenchTimer timer2;
            timer2.start();
            for (int i = 0; i < bench_iters; i++) {
                CHECK_CU(cuLaunchKernel(fn_split_q,
                    ns, num_heads, batch,   128, 1, 1,
                    (unsigned)smem_size, NULL, split_args, NULL));
                CHECK_CU(cuLaunchKernel(fn_reduce,
                    seq / Br_block, num_heads, batch,   128, 1, 1,
                    0, NULL, reduce_args, NULL));
            }
            float split_ms = timer2.stop_ms() / bench_iters;
            double split_gflops = total_flops / (split_ms / 1000.0) / 1e9;
            float speedup = br16_ms / split_ms;

            char label[64];
            snprintf(label, sizeof(label), "split-Q (splits=%-3d, %.0f MB partial)",
                     ns, total_partial / 1e6);
            printf("  %-45s %7.3f ms  %7.1f GFLOPS  %.2fx\n",
                   label, split_ms, split_gflops, speedup);

            cuMemFree(dPm); cuMemFree(dPl); cuMemFree(dPo);
        }
    }

    printf("\nExpected SASS:\n");
    printf("  cuobjdump -sass flash_split_q.sm_86.cubin | grep HMMA\n");

    cuMemFree(dQm); cuMemFree(dKm); cuMemFree(dVm); cuMemFree(dOm);
    cuModuleUnload(mod_br16); cuModuleUnload(mod_split);
    cuCtxDestroy(ctx);

    return 0;
}
