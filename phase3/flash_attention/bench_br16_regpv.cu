/*
 * bench_br16_regpv.cu — Benchmark: flash_attn_br16_regpv vs flash_attn_br16 (baseline)
 *
 * Tests the register-resident PV accumulator optimization:
 *   Baseline (br16):  48 KB smem → 2 blocks/SM = 8 warps/SM
 *   New (regpv):      32 KB smem → 3 blocks/SM = 12 warps/SM
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o flash_br16.sm_86.cubin flash_attn_br16.cu
 *   nvcc --cubin -arch=sm_86 -O2 -o flash_br16_regpv.sm_86.cubin flash_attn_br16_regpv.cu
 *   nvcc -arch=sm_86 -O2 -o bench_br16_regpv bench_br16_regpv.cu -lcuda -I../../phase2/common
 *
 * Usage:
 *   ./bench_br16_regpv              # seq=1024, batch=8, heads=8
 *   ./bench_br16_regpv 2048 4 8
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>
#include <cuda_fp16.h>

#include "../../phase2/common/bench.h"
#include "../../phase2/common/check.h"

// CPU reference (FP32 naive)
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
    int seq        = (argc > 1) ? atoi(argv[1]) : 1024;
    int batch      = (argc > 2) ? atoi(argv[2]) : 8;
    int num_heads  = (argc > 3) ? atoi(argv[3]) : 8;

    const int d         = 64;
    const int Br_block  = 64;
    const int Bc_val    = 64;

    if (seq % Br_block != 0) {
        fprintf(stderr, "seq=%d must be divisible by Br_block=%d\n", seq, Br_block);
        return 1;
    }

    float scale = 1.0f / sqrtf((float)d);

    printf("=== Flash Attention: Register-PV (32 KB) vs Baseline (48 KB) ===\n");
    printf("seq=%d  d=%d  batch=%d  heads=%d\n\n", seq, d, batch, num_heads);

    CHECK_CU(cuInit(0));
    CUdevice cu_dev; CHECK_CU(cuDeviceGet(&cu_dev, 0));
    char devname[256]; CHECK_CU(cuDeviceGetName(devname, sizeof(devname), cu_dev));
    printf("Device: %s\n\n", devname);

    CUcontext ctx; CHECK_CU(cuCtxCreate(&ctx, 0, cu_dev));

    // Load cubins
    CUmodule mod_base, mod_regpv;
    CUfunction fn_base, fn_regpv;

    if (cuModuleLoad(&mod_base, "flash_br16.sm_86.cubin") != CUDA_SUCCESS) {
        fprintf(stderr, "Cannot load flash_br16.sm_86.cubin. Build baseline first.\n");
        return 1;
    }
    if (cuModuleLoad(&mod_regpv, "flash_br16_regpv.sm_86.cubin") != CUDA_SUCCESS) {
        fprintf(stderr, "Cannot load flash_br16_regpv.sm_86.cubin. Build regpv first.\n");
        return 1;
    }
    CHECK_CU(cuModuleGetFunction(&fn_base,  mod_base,  "flash_attn_br16"));
    CHECK_CU(cuModuleGetFunction(&fn_regpv, mod_regpv, "flash_attn_br16_regpv"));

    // Shared memory sizes
    size_t smem_base  = 2 * Bc_val * d * sizeof(short)     // K+V tiles FP16
                      + Br_block * Bc_val * sizeof(float)   // smem_work FP32
                      + Br_block * d      * sizeof(float);  // smem_pv FP32
    size_t smem_regpv = 2 * Bc_val * d * sizeof(short)     // K+V tiles FP16
                      + Br_block * Bc_val * sizeof(float);  // smem_work FP32 (no smem_pv)

    CHECK_CU(cuFuncSetAttribute(fn_base,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem_base));
    CHECK_CU(cuFuncSetAttribute(fn_regpv,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem_regpv));

    printf("Baseline shared memory: %zu bytes (%.1f KB)\n", smem_base, smem_base/1024.0);
    printf("RegPV    shared memory: %zu bytes (%.1f KB)\n\n", smem_regpv, smem_regpv/1024.0);

    // Query register counts
    int regs_base = 0, regs_regpv = 0;
    cuFuncGetAttribute(&regs_base, CU_FUNC_ATTRIBUTE_NUM_REGS, fn_base);
    cuFuncGetAttribute(&regs_regpv, CU_FUNC_ATTRIBUTE_NUM_REGS, fn_regpv);
    printf("Registers per thread:  baseline=%d  regpv=%d\n", regs_base, regs_regpv);

    // Compute occupancy
    int max_blocks_base = 0, max_blocks_regpv = 0;
    cuOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_base, fn_base, 128, smem_base);
    cuOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_regpv, fn_regpv, 128, smem_regpv);
    printf("Blocks per SM:         baseline=%d  regpv=%d\n", max_blocks_base, max_blocks_regpv);
    printf("Warps per SM:          baseline=%d  regpv=%d\n\n",
           max_blocks_base * 4, max_blocks_regpv * 4);

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
    CHECK_CU(cuMemAlloc(&dO,  n_bytes));

    CHECK_CU(cuMemcpyHtoD(dQh, hQh, n_elems * sizeof(__half)));
    CHECK_CU(cuMemcpyHtoD(dKh, hKh, n_elems * sizeof(__half)));
    CHECK_CU(cuMemcpyHtoD(dVh, hVh, n_elems * sizeof(__half)));

    int n1 = 1;
    void *args[] = { &dQh, &dKh, &dVh, &dO, &seq, &n1, &scale };

    printf("Correctness (vs CPU FP32 naive):\n");

    // Test baseline
    CHECK_CU(cuMemsetD32(dO, 0, n_elems));
    CHECK_CU(cuLaunchKernel(fn_base,
        seq / Br_block, 1, 1,   128, 1, 1,   (unsigned)smem_base, NULL, args, NULL));
    CHECK_CU(cuCtxSynchronize());
    CHECK_CU(cuMemcpyDtoH(hOut, dO, n_bytes));
    auto r_base = check_fp32(hOut, hRef, n_elems, 1e-2f, 1e-0f);
    print_check_result("flash_attn_br16          (baseline, 48 KB)", r_base);

    // Test regpv
    CHECK_CU(cuMemsetD32(dO, 0, n_elems));
    CHECK_CU(cuLaunchKernel(fn_regpv,
        seq / Br_block, 1, 1,   128, 1, 1,   (unsigned)smem_regpv, NULL, args, NULL));
    CHECK_CU(cuCtxSynchronize());
    CHECK_CU(cuMemcpyDtoH(hOut, dO, n_bytes));
    auto r_regpv = check_fp32(hOut, hRef, n_elems, 1e-2f, 1e-0f);
    print_check_result("flash_attn_br16_regpv    (reg PV,   32 KB)", r_regpv);
    printf("\n");

    cuMemFree(dQh); cuMemFree(dKh); cuMemFree(dVh); cuMemFree(dO);

    // =========================================================
    // Performance benchmark (multi-head)
    // =========================================================
    printf("Performance (batch=%d, heads=%d, seq=%d):\n\n", batch, num_heads, seq);

    size_t tot_elems = (size_t)batch * num_heads * seq * d;

    CUdeviceptr dQm, dKm, dVm, dOm;
    CHECK_CU(cuMemAlloc(&dQm, tot_elems * sizeof(__half)));
    CHECK_CU(cuMemAlloc(&dKm, tot_elems * sizeof(__half)));
    CHECK_CU(cuMemAlloc(&dVm, tot_elems * sizeof(__half)));
    CHECK_CU(cuMemAlloc(&dOm, tot_elems * sizeof(float)));

    CHECK_CU(cuMemsetD16(dQm, 0x3800, tot_elems));
    CHECK_CU(cuMemsetD16(dKm, 0x3800, tot_elems));
    CHECK_CU(cuMemsetD16(dVm, 0x3800, tot_elems));

    int warmup = 5, bench_n = 50;

    // GFLOPS: QK^T (2*S*S*D) + softmax (~5*S*S) + PV (2*S*S*D) per head per batch
    double total_flops = (double)batch * num_heads * seq
                       * ((double)seq * d * 2.0   // QK^T
                        + (double)seq * 5.0        // softmax
                        + (double)seq * d * 2.0);  // PV

    auto run_bench = [&](CUfunction fn, size_t smem, const char *label) {
        void *bench_args[] = { &dQm, &dKm, &dVm, &dOm, &seq, &num_heads, &scale };
        int grid_x = seq / Br_block;

        for (int i = 0; i < warmup; i++) {
            CHECK_CU(cuLaunchKernel(fn,
                grid_x, num_heads, batch,
                128, 1, 1,
                (unsigned)smem, NULL, bench_args, NULL));
        }
        CHECK_CU(cuCtxSynchronize());

        float avg_ms;
        {
            BenchTimer timer;
            timer.start();
            for (int i = 0; i < bench_n; i++) {
                CHECK_CU(cuLaunchKernel(fn,
                    grid_x, num_heads, batch,
                    128, 1, 1,
                    (unsigned)smem, NULL, bench_args, NULL));
            }
            avg_ms = timer.stop_ms() / bench_n;
        }

        double gflops = total_flops / (avg_ms / 1000.0) / 1e9;
        printf("  %-45s %7.3f ms  %8.0f GFLOPS\n", label, avg_ms, gflops);
        return avg_ms;
    };

    float ms_base  = run_bench(fn_base,  smem_base,  "flash_attn_br16       (baseline, 48 KB)");
    float ms_regpv = run_bench(fn_regpv, smem_regpv, "flash_attn_br16_regpv (reg PV,   32 KB)");

    printf("\n  Speedup: %.2fx (%.1f%% %s)\n",
           ms_base / ms_regpv,
           fabsf(ms_base - ms_regpv) / ms_base * 100.0f,
           ms_regpv < ms_base ? "faster" : "slower");

    printf("\nSASS inspection:\n");
    printf("  cuobjdump -sass flash_br16_regpv.sm_86.cubin | grep -c HMMA\n");
    printf("  cuobjdump -sass flash_br16_regpv.sm_86.cubin | grep -c 'LDS\\|STS'\n");

    cuMemFree(dQm); cuMemFree(dKm); cuMemFree(dVm); cuMemFree(dOm);
    cuModuleUnload(mod_base); cuModuleUnload(mod_regpv);
    cuCtxDestroy(ctx);

    free(hQf); free(hKf); free(hVf); free(hRef); free(hOut); free(sBuf);
    free(hQh); free(hKh); free(hVh);
    return 0;
}
