/*
 * bench_igemm_sparse.cu — Benchmark and correctness test for sparse INT8 GEMM
 *
 * Tests igemm_sparse_tiled: 16-warp 128×128, mma.sp.m16n8k32.s32.s8.s8.s32
 * Uses sparse_meta_int8.h for host-side 2:4 compression and CPU reference.
 *
 * Correctness tolerance: INT8 accumulates into INT32, then scaled to FP32.
 * Use abs=0.5, rel=0.1 (same as dense IGEMM).
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o igemm_sparse_tiled.sm_86.cubin igemm_sparse_tiled.cu
 *   nvcc -arch=sm_86 -O2 -o bench_igemm_sparse bench_igemm_sparse.cu -lcuda -I../common
 *
 * Usage:
 *   ./bench_igemm_sparse              # default M=N=K=256
 *   ./bench_igemm_sparse 512 512 512
 *   ./bench_igemm_sparse 2048 2048 2048
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda.h>

#include "../common/bench.h"
#include "../common/check.h"
#include "sparse_meta_int8.h"

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 256;
    int N = (argc > 2) ? atoi(argv[2]) : M;
    int K = (argc > 3) ? atoi(argv[3]) : M;

    if (M % 16 != 0 || N % 16 != 0 || K % 32 != 0) {
        fprintf(stderr, "M, N must be multiples of 16; K must be multiple of 32\n");
        return 1;
    }

    printf("=== Sparse INT8 GEMM (mma.sp.m16n8k32) ===\n");
    printf("M=%d  N=%d  K=%d\n\n", M, N, K);

    CHECK_CU(cuInit(0));
    CUdevice cu_dev; CHECK_CU(cuDeviceGet(&cu_dev, 0));
    char devname[256]; CHECK_CU(cuDeviceGetName(devname, sizeof(devname), cu_dev));
    printf("Device: %s\n\n", devname);

    CUcontext ctx; CHECK_CU(cuCtxCreate(&ctx, 0, cu_dev));

    // =========================================================
    // Load cubin
    // =========================================================
    CUmodule mod;
    CUfunction fn;
    if (cuModuleLoad(&mod, "igemm_sparse_tiled.sm_86.cubin") != CUDA_SUCCESS) {
        fprintf(stderr, "Cannot load igemm_sparse_tiled.sm_86.cubin. Build first.\n");
        return 1;
    }
    CHECK_CU(cuModuleGetFunction(&fn, mod, "igemm_sparse_tiled"));

    int num_regs = 0;
    cuFuncGetAttribute(&num_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn);
    printf("Kernel registers: %d\n", num_regs);
    printf("(SASS: cuobjdump -sass igemm_sparse_tiled.sm_86.cubin | grep -E 'IMMA|LDGSTS')\n\n");

    // =========================================================
    // Allocate host data
    // =========================================================
    size_t elems_a      = (size_t)M * K;
    size_t elems_b      = (size_t)K * N;
    size_t elems_c      = (size_t)M * N;
    size_t elems_a_comp = (size_t)M * (K / 2);  // compressed A
    size_t meta_count   = sparse_meta_count_int8(M, K);

    int8_t   *host_a_dense = (int8_t*)  malloc(elems_a      * sizeof(int8_t));
    int8_t   *host_a_comp  = (int8_t*)  malloc(elems_a_comp * sizeof(int8_t));
    int8_t   *host_b       = (int8_t*)  malloc(elems_b      * sizeof(int8_t));
    uint32_t *host_meta    = (uint32_t*)malloc(meta_count   * sizeof(uint32_t));
    int32_t  *host_ref_i32 = (int32_t*) malloc(elems_c      * sizeof(int32_t));
    float    *host_ref     = (float*)   malloc(elems_c      * sizeof(float));
    float    *host_out     = (float*)   malloc(elems_c      * sizeof(float));

    // Scale factors (symmetric quantization)
    float scale_a = 1.0f / 127.0f;
    float scale_b = 1.0f / 127.0f;

    // Generate random 2:4 sparse A and dense B
    gen_random_sparse_2_4_int8(host_a_dense, M, K, /*seed=*/42);
    compress_2_4_int8(host_a_dense, M, K, host_a_comp, host_meta);

    // Random dense B: values in -127..127
    srand(137);
    for (size_t i = 0; i < elems_b; i++) {
        int8_t v;
        do { v = (int8_t)((rand() % 255) - 127); } while (v == 0);
        host_b[i] = v;
    }

    // CPU reference (INT32 accumulation on sparse A)
    printf("Computing CPU reference...\n");
    cpu_sparse_gemm_int8(host_a_dense, host_b, host_ref_i32, M, N, K);
    float dequant = scale_a * scale_b;
    for (size_t i = 0; i < elems_c; i++)
        host_ref[i] = (float)host_ref_i32[i] * dequant;
    printf("Done.\n\n");

    // =========================================================
    // GPU: allocate and copy
    // =========================================================
    CUdeviceptr d_a_comp, d_b, d_c, d_meta;
    CHECK_CU(cuMemAlloc(&d_a_comp, elems_a_comp * sizeof(int8_t)));
    CHECK_CU(cuMemAlloc(&d_b,      elems_b      * sizeof(int8_t)));
    CHECK_CU(cuMemAlloc(&d_c,      elems_c      * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_meta,   meta_count   * sizeof(uint32_t)));

    CHECK_CU(cuMemcpyHtoD(d_a_comp, host_a_comp, elems_a_comp * sizeof(int8_t)));
    CHECK_CU(cuMemcpyHtoD(d_b,      host_b,      elems_b      * sizeof(int8_t)));
    CHECK_CU(cuMemcpyHtoD(d_meta,   host_meta,   meta_count   * sizeof(uint32_t)));

    // =========================================================
    // Correctness test
    // =========================================================
    printf("=== Correctness ===\n");
    CHECK_CU(cuMemsetD32(d_c, 0, elems_c));

    int grid_x = (N + 127) / 128;
    int grid_y = (M + 127) / 128;
    void *args[] = { &d_a_comp, &d_b, &d_c, &d_meta, &M, &N, &K, &scale_a, &scale_b };

    CHECK_CU(cuLaunchKernel(fn,
        grid_x, grid_y, 1,
        512, 1, 1,
        0, NULL, args, NULL));
    CHECK_CU(cuCtxSynchronize());
    CHECK_CU(cuMemcpyDtoH(host_out, d_c, elems_c * sizeof(float)));

    auto result = check_fp32(host_out, host_ref, elems_c, 0.5f, 0.1f);
    print_check_result("igemm_sparse_tiled (mma.sp.m16n8k32, INT8)", result);

    if (result.num_errors > 0 && result.num_errors <= 20) {
        printf("\n  First errors:\n");
        int shown = 0;
        for (size_t i = 0; i < elems_c && shown < 10; i++) {
            float abs_err = fabsf(host_out[i] - host_ref[i]);
            float ref_abs = fabsf(host_ref[i]);
            float rel_err = (ref_abs > 1e-8f) ? (abs_err / ref_abs) : abs_err;
            if (abs_err > 0.5f && rel_err > 0.1f) {
                int r = (int)(i / N), c = (int)(i % N);
                printf("    [%zu] row=%d col=%d GPU=%.4f REF=%.4f abs=%.2e rel=%.2e\n",
                       i, r, c, host_out[i], host_ref[i], abs_err, rel_err);
                shown++;
            }
        }
    }
    printf("\n");

    // =========================================================
    // Performance benchmark (M >= 512)
    // =========================================================
    if (M >= 512) {
        printf("=== Performance (M=%d, N=%d, K=%d) ===\n", M, N, K);

        int warmup_iters = 5, bench_iters = 50;

        // Sparse GFLOPS: effective ops = 2 × M × N × (K/2) = K/2 mults per output element
        double eff_flops      = 2.0 * M * N * (K / 2);
        double dense_eq_flops = 2.0 * M * N * K;

        for (int i = 0; i < warmup_iters; i++) {
            CHECK_CU(cuLaunchKernel(fn,
                grid_x, grid_y, 1, 512, 1, 1, 0, NULL, args, NULL));
        }
        CHECK_CU(cuCtxSynchronize());

        float avg_ms;
        {
            BenchTimer timer;
            timer.start();
            for (int i = 0; i < bench_iters; i++) {
                CHECK_CU(cuLaunchKernel(fn,
                    grid_x, grid_y, 1, 512, 1, 1, 0, NULL, args, NULL));
            }
            avg_ms = timer.stop_ms() / bench_iters;
        }

        double eff_gflops      = eff_flops      / (avg_ms / 1000.0) / 1e9;
        double dense_eq_gflops = dense_eq_flops / (avg_ms / 1000.0) / 1e9;

        printf("  %-36s %9.3f ms  %8.0f eff GFLOPS  %8.0f dense-equiv GFLOPS\n",
               "igemm_sparse_tiled", avg_ms, eff_gflops, dense_eq_gflops);
        printf("\n");
    }

    // Cleanup
    cuMemFree(d_a_comp);
    cuMemFree(d_b);
    cuMemFree(d_c);
    cuMemFree(d_meta);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);

    free(host_a_dense); free(host_a_comp); free(host_b);
    free(host_meta); free(host_ref_i32); free(host_ref); free(host_out);

    return 0;
}
