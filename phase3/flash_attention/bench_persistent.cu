/*
 * bench_persistent.cu — Persistent grid Flash Attention vs standard br16
 *
 * Tests SM utilization improvement at small batch sizes where standard grid
 * leaves many SMs idle.
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o flash_br16.sm_86.cubin flash_attn_br16.cu
 *   nvcc --cubin -arch=sm_86 -O2 -o flash_persistent.sm_86.cubin flash_attn_persistent.cu
 *   nvcc -arch=sm_86 -O2 -o bench_persistent bench_persistent.cu -lcuda -I../../phase2/common
 *
 * Usage:
 *   ./bench_persistent                  # default: batch=1 heads=8
 *   ./bench_persistent 4 16            # batch=4 heads=16
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>
#include <cuda_fp16.h>

#include "../../phase2/common/bench.h"
#include "../../phase2/common/check.h"

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
    int batch     = (argc > 1) ? atoi(argv[1]) : 1;
    int num_heads = (argc > 2) ? atoi(argv[2]) : 8;

    const int d        = 64;
    const int Br_block = 64;
    const int Bc       = 64;

    float scale = 1.0f / sqrtf((float)d);

    printf("=== Persistent Grid Flash Attention Benchmark ===\n");
    printf("batch=%d  heads=%d  d=%d\n\n", batch, num_heads, d);

    CHECK_CU(cuInit(0));
    CUdevice cu_dev; CHECK_CU(cuDeviceGet(&cu_dev, 0));
    char devname[256]; CHECK_CU(cuDeviceGetName(devname, sizeof(devname), cu_dev));

    int num_sms = 0;
    CHECK_CU(cuDeviceGetAttribute(&num_sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cu_dev));
    printf("Device: %s (%d SMs)\n\n", devname, num_sms);

    CUcontext ctx; CHECK_CU(cuCtxCreate(&ctx, 0, cu_dev));

    // Load cubins
    CUmodule mod_br16, mod_persist;
    CUfunction fn_br16, fn_persist;

    if (cuModuleLoad(&mod_br16, "flash_br16.sm_86.cubin") != CUDA_SUCCESS) {
        fprintf(stderr, "Cannot load flash_br16.sm_86.cubin\n");
        return 1;
    }
    if (cuModuleLoad(&mod_persist, "flash_persistent.sm_86.cubin") != CUDA_SUCCESS) {
        fprintf(stderr, "Cannot load flash_persistent.sm_86.cubin\n");
        return 1;
    }
    CHECK_CU(cuModuleGetFunction(&fn_br16,   mod_br16,    "flash_attn_br16"));
    CHECK_CU(cuModuleGetFunction(&fn_persist, mod_persist, "flash_attn_persistent"));

    size_t smem_bytes = 2 * Bc * d * sizeof(short)
                      + Br_block * Bc * sizeof(float)
                      + Br_block * d  * sizeof(float);
    CHECK_CU(cuFuncSetAttribute(fn_br16,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem_bytes));
    CHECK_CU(cuFuncSetAttribute(fn_persist,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem_bytes));
    printf("Shared memory: %zu bytes (%.1f KB)\n", smem_bytes, smem_bytes/1024.0);

    int persistent_grid = num_sms * 2;  // 2 blocks/SM for 8 warps
    printf("Persistent grid: %d blocks (%d SMs × 2)\n\n", persistent_grid, num_sms);

    // Tile counter for persistent kernel
    CUdeviceptr dev_tile_counter;
    CHECK_CU(cuMemAlloc(&dev_tile_counter, sizeof(int)));

    // =========================================================
    // Correctness (single head, seq=256)
    // =========================================================
    {
        int seq = 256;
        size_t n_elems = (size_t)seq * d;

        float *hQf  = (float*)malloc(n_elems * sizeof(float));
        float *hKf  = (float*)malloc(n_elems * sizeof(float));
        float *hVf  = (float*)malloc(n_elems * sizeof(float));
        float *hRef = (float*)malloc(n_elems * sizeof(float));
        float *hOut = (float*)malloc(n_elems * sizeof(float));
        float *sBuf = (float*)malloc(seq * sizeof(float));

        fill_random(hQf, n_elems, 20);
        fill_random(hKf, n_elems, 21);
        fill_random(hVf, n_elems, 22);

        printf("Correctness (seq=%d, single head):\n", seq);
        cpu_attention(hQf, hKf, hVf, hRef, sBuf, seq, d, scale);

        __half *hQh = (__half*)malloc(n_elems * sizeof(__half));
        __half *hKh = (__half*)malloc(n_elems * sizeof(__half));
        __half *hVh = (__half*)malloc(n_elems * sizeof(__half));
        fp32_to_fp16(hQf, hQh, n_elems);
        fp32_to_fp16(hKf, hKh, n_elems);
        fp32_to_fp16(hVf, hVh, n_elems);

        CUdeviceptr dQh, dKh, dVh, dO;
        CHECK_CU(cuMemAlloc(&dQh, n_elems * sizeof(__half)));
        CHECK_CU(cuMemAlloc(&dKh, n_elems * sizeof(__half)));
        CHECK_CU(cuMemAlloc(&dVh, n_elems * sizeof(__half)));
        CHECK_CU(cuMemAlloc(&dO,  n_elems * sizeof(float)));

        CHECK_CU(cuMemcpyHtoD(dQh, hQh, n_elems * sizeof(__half)));
        CHECK_CU(cuMemcpyHtoD(dKh, hKh, n_elems * sizeof(__half)));
        CHECK_CU(cuMemcpyHtoD(dVh, hVh, n_elems * sizeof(__half)));

        // br16
        int n1 = 1;
        void *args_br[] = { &dQh, &dKh, &dVh, &dO, &seq, &n1, &scale };
        CHECK_CU(cuMemsetD32(dO, 0, n_elems));
        CHECK_CU(cuLaunchKernel(fn_br16,
            seq / Br_block, 1, 1,  128, 1, 1,
            (unsigned)smem_bytes, NULL, args_br, NULL));
        CHECK_CU(cuCtxSynchronize());
        CHECK_CU(cuMemcpyDtoH(hOut, dO, n_elems * sizeof(float)));
        auto r1 = check_fp32(hOut, hRef, n_elems, 1e-2f, 1e0f);
        print_check_result("flash_attn_br16       ", r1);

        // persistent
        int batch_1 = 1;
        int total_tiles = seq / Br_block;
        int q_tiles_per_seq = seq / Br_block;
        CHECK_CU(cuMemsetD32(dO, 0, n_elems));
        CHECK_CU(cuMemsetD32(dev_tile_counter, 0, 1));
        void *args_p[] = { &dQh, &dKh, &dVh, &dO, &seq, &n1, &batch_1, &scale,
                           &total_tiles, &q_tiles_per_seq, &dev_tile_counter };
        CHECK_CU(cuLaunchKernel(fn_persist,
            persistent_grid, 1, 1,  128, 1, 1,
            (unsigned)smem_bytes, NULL, args_p, NULL));
        CHECK_CU(cuCtxSynchronize());
        CHECK_CU(cuMemcpyDtoH(hOut, dO, n_elems * sizeof(float)));
        auto r2 = check_fp32(hOut, hRef, n_elems, 1e-2f, 1e0f);
        print_check_result("flash_attn_persistent ", r2);

        cuMemFree(dQh); cuMemFree(dKh); cuMemFree(dVh); cuMemFree(dO);
        free(hQf); free(hKf); free(hVf); free(hRef); free(hOut); free(sBuf);
        free(hQh); free(hKh); free(hVh);
    }

    // =========================================================
    // Performance sweep: seq = 256, 512, 1024
    // =========================================================
    int seqs[] = { 256, 512, 1024 };
    int warmup = 5, bench_n = 50;

    printf("\nPerformance (batch=%d, heads=%d):\n", batch, num_heads);
    printf("%-8s  %-6s  %-12s  %-12s  %-10s  %-10s\n",
           "seq", "tiles", "br16 (ms)", "persist (ms)", "speedup", "note");
    printf("------  ------  ----------  -----------  --------  ---------\n");

    for (int si = 0; si < 3; si++) {
        int seq = seqs[si];
        int q_tiles = seq / Br_block;
        int total_tiles = q_tiles * num_heads * batch;

        size_t tot_elems = (size_t)batch * num_heads * seq * d;

        CUdeviceptr dQm, dKm, dVm, dOm;
        CHECK_CU(cuMemAlloc(&dQm, tot_elems * sizeof(__half)));
        CHECK_CU(cuMemAlloc(&dKm, tot_elems * sizeof(__half)));
        CHECK_CU(cuMemAlloc(&dVm, tot_elems * sizeof(__half)));
        CHECK_CU(cuMemAlloc(&dOm, tot_elems * sizeof(float)));
        CHECK_CU(cuMemsetD16(dQm, 0x3800, tot_elems));
        CHECK_CU(cuMemsetD16(dKm, 0x3800, tot_elems));
        CHECK_CU(cuMemsetD16(dVm, 0x3800, tot_elems));

        // --- br16 benchmark ---
        void *args_br[] = { &dQm, &dKm, &dVm, &dOm, &seq, &num_heads, &scale };
        for (int i = 0; i < warmup; i++) {
            CHECK_CU(cuLaunchKernel(fn_br16,
                q_tiles, num_heads, batch, 128, 1, 1,
                (unsigned)smem_bytes, NULL, args_br, NULL));
        }
        CHECK_CU(cuCtxSynchronize());

        float ms_br16;
        {
            BenchTimer timer;
            timer.start();
            for (int i = 0; i < bench_n; i++) {
                CHECK_CU(cuLaunchKernel(fn_br16,
                    q_tiles, num_heads, batch, 128, 1, 1,
                    (unsigned)smem_bytes, NULL, args_br, NULL));
            }
            ms_br16 = timer.stop_ms() / bench_n;
        }

        // --- persistent benchmark ---
        int q_tiles_per_seq = q_tiles;
        int grid_p = persistent_grid;

        void *args_p[] = { &dQm, &dKm, &dVm, &dOm, &seq, &num_heads, &batch,
                           &scale, &total_tiles, &q_tiles_per_seq, &dev_tile_counter };

        for (int i = 0; i < warmup; i++) {
            CHECK_CU(cuMemsetD32(dev_tile_counter, 0, 1));
            CHECK_CU(cuLaunchKernel(fn_persist,
                grid_p, 1, 1, 128, 1, 1,
                (unsigned)smem_bytes, NULL, args_p, NULL));
        }
        CHECK_CU(cuCtxSynchronize());

        float ms_persist;
        {
            BenchTimer timer;
            timer.start();
            for (int i = 0; i < bench_n; i++) {
                CHECK_CU(cuMemsetD32(dev_tile_counter, 0, 1));
                CHECK_CU(cuLaunchKernel(fn_persist,
                    grid_p, 1, 1, 128, 1, 1,
                    (unsigned)smem_bytes, NULL, args_p, NULL));
            }
            ms_persist = timer.stop_ms() / bench_n;
        }

        float sm_util = (float)total_tiles / (float)(num_sms * 2) * 100.0f;
        if (sm_util > 100.0f) sm_util = 100.0f;

        const char *note = "";
        if (total_tiles < num_sms) note = "SMs idle";
        else if (total_tiles < num_sms * 2) note = "1 blk/SM";

        printf("%-8d  %-6d  %-12.3f  %-12.3f  %-10.4f  %s\n",
               seq, total_tiles, ms_br16, ms_persist,
               ms_br16 / ms_persist, note);

        cuMemFree(dQm); cuMemFree(dKm); cuMemFree(dVm); cuMemFree(dOm);
    }

    printf("\nSM slot utilization: %d SMs × 2 blocks/SM = %d slots\n",
           num_sms, num_sms * 2);

    cuMemFree(dev_tile_counter);
    cuModuleUnload(mod_br16);
    cuModuleUnload(mod_persist);
    cuCtxDestroy(ctx);
    return 0;
}
