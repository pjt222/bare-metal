/*
 * bench_br16.cu — Benchmark: flash_attn_br16 (HMMA) vs flash_attn_4warp (scalar)
 *
 * Build:
 *   nvcc -arch=sm_86 -O2 -o bench_br16 bench_br16.cu -lcuda -I../../phase2/common
 *
 * Usage:
 *   ./bench_br16              # seq=1024, batch=8, heads=8
 *   ./bench_br16 2048 4 8
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>
#include <cuda_fp16.h>

#include "../../phase2/common/bench.h"
#include "../../phase2/common/check.h"

// CPU reference (FP32 naive — same as before)
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

// Convert FP32 host array to FP16 for br16 kernel
static void fp32_to_fp16(const float *src, __half *dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = __float2half(src[i]);
}

int main(int argc, char **argv) {
    int seq        = (argc > 1) ? atoi(argv[1]) : 1024;
    int batch      = (argc > 2) ? atoi(argv[2]) : 8;
    int num_heads  = (argc > 3) ? atoi(argv[3]) : 8;

    const int d        = 64;
    const int Br_block = 64;   // must match flash_attn_br16.cu
    const int num_warps = 4;
    const int Bc_4warp  = 64;

    if (seq % Br_block != 0) {
        fprintf(stderr, "seq=%d must be divisible by Br_block=%d\n", seq, Br_block);
        return 1;
    }

    float scale = 1.0f / sqrtf((float)d);

    printf("=== Flash Attention WMMA (Br=16 HMMA) vs 4-Warp Scalar ===\n");
    printf("seq=%d  d=%d  batch=%d  heads=%d\n\n", seq, d, batch, num_heads);

    CHECK_CU(cuInit(0));
    CUdevice cu_dev; CHECK_CU(cuDeviceGet(&cu_dev, 0));
    char devname[256]; CHECK_CU(cuDeviceGetName(devname, sizeof(devname), cu_dev));
    printf("Device: %s\n\n", devname);

    CUcontext ctx; CHECK_CU(cuCtxCreate(&ctx, 0, cu_dev));

    // Load cubins
    CUmodule mod_4w, mod_br16;
    CUfunction fn_4warp, fn_br16;

    if (cuModuleLoad(&mod_4w, "flash_wmma.sm_86.cubin") != CUDA_SUCCESS ||
        cuModuleLoad(&mod_br16, "flash_br16.sm_86.cubin") != CUDA_SUCCESS) {
        fprintf(stderr, "Cannot load cubins. Build both first.\n");
        return 1;
    }
    CHECK_CU(cuModuleGetFunction(&fn_4warp, mod_4w,   "flash_attn_4warp"));
    CHECK_CU(cuModuleGetFunction(&fn_br16,  mod_br16, "flash_attn_br16"));

    // Set required shared memory for br16 (48 KB)
    size_t smem_br16 = 2 * Bc_4warp * d * sizeof(short)      // K+V tiles FP16
                     + Br_block * Bc_4warp * sizeof(float)    // smem_work FP32
                     + Br_block * d        * sizeof(float);   // smem_pv FP32
    CHECK_CU(cuFuncSetAttribute(fn_br16,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem_br16));
    printf("br16 shared memory: %zu bytes (%.1f KB)\n\n", smem_br16, smem_br16/1024.0);

    // =========================================================
    // Correctness test (single head)
    // =========================================================
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

    printf("Computing CPU reference...\n");
    cpu_attention(hQf, hKf, hVf, hRef, sBuf, seq, d, scale);
    printf("Done.\n\n");

    // FP16 versions for br16 kernel
    __half *hQh = (__half*)malloc(n_elems * sizeof(__half));
    __half *hKh = (__half*)malloc(n_elems * sizeof(__half));
    __half *hVh = (__half*)malloc(n_elems * sizeof(__half));
    fp32_to_fp16(hQf, hQh, n_elems);
    fp32_to_fp16(hKf, hKh, n_elems);
    fp32_to_fp16(hVf, hVh, n_elems);

    // Device buffers
    CUdeviceptr dQf, dKf, dVf, dO;        // FP32 for 4-warp
    CUdeviceptr dQh, dKh, dVh;             // FP16 for br16

    CHECK_CU(cuMemAlloc(&dQf, n_bytes));
    CHECK_CU(cuMemAlloc(&dKf, n_bytes));
    CHECK_CU(cuMemAlloc(&dVf, n_bytes));
    CHECK_CU(cuMemAlloc(&dO,  n_bytes));
    CHECK_CU(cuMemAlloc(&dQh, n_elems * sizeof(__half)));
    CHECK_CU(cuMemAlloc(&dKh, n_elems * sizeof(__half)));
    CHECK_CU(cuMemAlloc(&dVh, n_elems * sizeof(__half)));

    CHECK_CU(cuMemcpyHtoD(dQf, hQf, n_bytes));
    CHECK_CU(cuMemcpyHtoD(dKf, hKf, n_bytes));
    CHECK_CU(cuMemcpyHtoD(dVf, hVf, n_bytes));
    CHECK_CU(cuMemcpyHtoD(dQh, hQh, n_elems * sizeof(__half)));
    CHECK_CU(cuMemcpyHtoD(dKh, hKh, n_elems * sizeof(__half)));
    CHECK_CU(cuMemcpyHtoD(dVh, hVh, n_elems * sizeof(__half)));

    int n1 = 1;

    // Test 4-warp (FP32 in/out)
    void *args_4w[] = { &dQf, &dKf, &dVf, &dO, &seq, &n1, &scale };
    CHECK_CU(cuMemsetD32(dO, 0, n_elems));
    CHECK_CU(cuLaunchKernel(fn_4warp,
        seq / num_warps, 1, 1,   128, 1, 1,   0, NULL, args_4w, NULL));
    CHECK_CU(cuCtxSynchronize());
    CHECK_CU(cuMemcpyDtoH(hOut, dO, n_bytes));

    printf("Correctness (vs CPU FP32 naive):\n");
    auto r4 = check_fp32(hOut, hRef, n_elems, 1e-3f, 1e-1f);
    print_check_result("flash_attn_4warp (FP32 K/V)", r4);

    // Test br16 (FP16 in, FP32 out)
    void *args_br[] = { &dQh, &dKh, &dVh, &dO, &seq, &n1, &scale };
    CHECK_CU(cuMemsetD32(dO, 0, n_elems));
    CHECK_CU(cuLaunchKernel(fn_br16,
        seq / Br_block, 1, 1,   128, 1, 1,   (unsigned)smem_br16, NULL, args_br, NULL));
    CHECK_CU(cuCtxSynchronize());
    CHECK_CU(cuMemcpyDtoH(hOut, dO, n_bytes));

    auto rbr = check_fp32(hOut, hRef, n_elems, 1e-2f, 1e-0f);
    print_check_result("flash_attn_br16  (FP16 HMMA)", rbr);
    printf("\n");

    cuMemFree(dQf); cuMemFree(dKf); cuMemFree(dVf);
    cuMemFree(dQh); cuMemFree(dKh); cuMemFree(dVh); cuMemFree(dO);

    // =========================================================
    // Performance benchmark (multi-head)
    // =========================================================
    printf("Performance (batch=%d, heads=%d, seq=%d):\n\n", batch, num_heads, seq);

    size_t tot_f32 = (size_t)batch * num_heads * seq * d;
    size_t tot_f16 = tot_f32;  // same element count

    CUdeviceptr dQFm, dKFm, dVFm, dOMf;    // FP32 multi
    CUdeviceptr dQHm, dKHm, dVHm;           // FP16 multi

    CHECK_CU(cuMemAlloc(&dQFm, tot_f32 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&dKFm, tot_f32 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&dVFm, tot_f32 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&dOMf, tot_f32 * sizeof(float)));
    CHECK_CU(cuMemAlloc(&dQHm, tot_f16 * sizeof(__half)));
    CHECK_CU(cuMemAlloc(&dKHm, tot_f16 * sizeof(__half)));
    CHECK_CU(cuMemAlloc(&dVHm, tot_f16 * sizeof(__half)));

    CHECK_CU(cuMemsetD32(dQFm, 0x3f000000, tot_f32));
    CHECK_CU(cuMemsetD32(dKFm, 0x3f000000, tot_f32));
    CHECK_CU(cuMemsetD32(dVFm, 0x3f000000, tot_f32));
    // Initialize FP16 buffers (0x3800 ≈ 0.5 in FP16)
    CHECK_CU(cuMemsetD16(dQHm, 0x3800, tot_f16));
    CHECK_CU(cuMemsetD16(dKHm, 0x3800, tot_f16));
    CHECK_CU(cuMemsetD16(dVHm, 0x3800, tot_f16));

    int warmup = 5, bench_n = 50;

    auto run_bench_fn = [&](CUfunction fn, int grid_x, int block_x,
                             size_t smem, void **args_ptr, const char *label) {
        for (int i = 0; i < warmup; i++) {
            CHECK_CU(cuLaunchKernel(fn,
                grid_x, num_heads, batch,
                block_x, 1, 1,
                (unsigned)smem, NULL, args_ptr, NULL));
        }
        CHECK_CU(cuCtxSynchronize());
        float avg_ms;
        {
            BenchTimer timer;
            timer.start();
            for (int i = 0; i < bench_n; i++) {
                CHECK_CU(cuLaunchKernel(fn,
                    grid_x, num_heads, batch,
                    block_x, 1, 1,
                    (unsigned)smem, NULL, args_ptr, NULL));
            }
            avg_ms = timer.stop_ms() / bench_n;
        }
        double bw_elem = (double)tot_f32 * (1 + 2.0*(seq/64) + 1);  // Q+K×iter+V×iter+O
        double bw_gb   = bw_elem * sizeof(float) / 1e9 / (avg_ms / 1000.0);
        double ideal   = 4.0 * tot_f32 * sizeof(float) / 1e9 / (avg_ms / 1000.0);
        printf("  %-42s %7.3f ms  %6.1f GB/s  (ideal: %5.1f GB/s)\n",
               label, avg_ms, bw_gb, ideal);
    };

    void *args_4wm[] = { &dQFm, &dKFm, &dVFm, &dOMf, &seq, &num_heads, &scale };
    void *args_brm[] = { &dQHm, &dKHm, &dVHm, &dOMf, &seq, &num_heads, &scale };

    run_bench_fn(fn_4warp, seq/num_warps,  128, 0,         args_4wm, "flash_attn_4warp (FP32, BKV=64)");
    run_bench_fn(fn_br16,  seq/Br_block,   128, smem_br16, args_brm, "flash_attn_br16  (FP16, HMMA)  ");

    printf("\nNote: br16 uses FP16 inputs (half the bandwidth for K/V loads)\n");
    printf("Expected SASS: cuobjdump -sass flash_br16.sm_86.cubin | grep HMMA\n");

    cuMemFree(dQFm); cuMemFree(dKFm); cuMemFree(dVFm); cuMemFree(dOMf);
    cuMemFree(dQHm); cuMemFree(dKHm); cuMemFree(dVHm);
    cuModuleUnload(mod_4w); cuModuleUnload(mod_br16);
    cuCtxDestroy(ctx);

    free(hQf); free(hKf); free(hVf); free(hRef); free(hOut); free(sBuf);
    free(hQh); free(hKh); free(hVh);
    return 0;
}
