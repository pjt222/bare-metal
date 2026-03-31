/*
 * bench.cu — Benchmark: sparse HGEMM (mma.sp) vs dense HGEMM (WMMA)
 *
 * Tests 2:4 structured sparsity on GA104 (RTX 3070 Ti, sm_86).
 * Fixed 2:4 pattern: positions {0,1} per group of 4 along K are nonzero.
 *
 * Build:
 *   # In phase2/hgemm_sparse/
 *   nvcc --cubin -arch=sm_86 -O2 -o hgemm_sparse_naive.sm_86.cubin hgemm_sparse_naive.cu
 *   nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../common
 *
 * Usage:
 *   ./bench              # M=N=K=256 (small for correctness)
 *   ./bench 4096 4096 4096
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>
#include <cuda_fp16.h>

#include "../common/bench.h"
#include "../common/check.h"

// -----------------------------------------------------------------------
// Host-side 2:4 sparse compression
//
// Pattern: for every group of 4 consecutive K-elements, keep positions 0,1.
// A_dense [M × K] → A_compressed [M × K/2]
// -----------------------------------------------------------------------
static void compress_2_4(
    const float *A_dense, __half *A_compressed,
    int M, int K
) {
    int K_stored = K / 2;
    for (int row = 0; row < M; row++) {
        int stored_col = 0;
        for (int k = 0; k < K; k += 4) {
            // Keep positions 0, 1 of each group of 4
            A_compressed[row * K_stored + stored_col + 0] =
                __float2half(A_dense[row * K + k + 0]);
            A_compressed[row * K_stored + stored_col + 1] =
                __float2half(A_dense[row * K + k + 1]);
            stored_col += 2;
        }
    }
}

// -----------------------------------------------------------------------
// CPU reference: sparse GEMM with the same 2:4 pattern
// C = A_sparse × B where A_sparse has zeros at positions {2,3} per group of 4
// -----------------------------------------------------------------------
static void cpu_sparse_gemm(
    const float *A_dense,  // [M × K] — positions 2,3 per group are ignored
    const float *B,        // [K × N]
    float *C,              // [M × N]
    int M, int N, int K
) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                // Only use positions 0,1 per group of 4
                if ((k % 4) < 2) {
                    acc += A_dense[row * K + k] * B[k * N + col];
                }
            }
            C[row * N + col] = acc;
        }
    }
}

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 256;
    int N = (argc > 2) ? atoi(argv[2]) : M;
    int K = (argc > 3) ? atoi(argv[3]) : M;

    if (K % 16 != 0 || M % 16 != 0 || N % 16 != 0) {
        fprintf(stderr, "M, N, K must be multiples of 16\n");
        return 1;
    }

    printf("=== Sparse HGEMM (2:4 mma.sp) — Proof of Concept ===\n");
    printf("M=%d  N=%d  K=%d\n\n", M, N, K);

    CHECK_CU(cuInit(0));
    CUdevice cu_dev; CHECK_CU(cuDeviceGet(&cu_dev, 0));
    char devname[256]; CHECK_CU(cuDeviceGetName(devname, sizeof(devname), cu_dev));
    printf("Device: %s\n\n", devname);

    CUcontext ctx; CHECK_CU(cuCtxCreate(&ctx, 0, cu_dev));

    // Load cubin
    CUmodule mod_sparse;
    CUfunction fn_sparse;
    if (cuModuleLoad(&mod_sparse, "hgemm_sparse_naive.sm_86.cubin") != CUDA_SUCCESS) {
        fprintf(stderr, "Cannot load hgemm_sparse_naive.sm_86.cubin. Build first.\n");
        return 1;
    }
    CHECK_CU(cuModuleGetFunction(&fn_sparse, mod_sparse, "hgemm_sparse_naive"));

    // Query kernel info
    int regs = 0;
    cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn_sparse);
    printf("Sparse kernel registers: %d\n", regs);

    // Check SASS for sparse HMMA
    printf("(Inspect SASS: cuobjdump -sass hgemm_sparse_naive.sm_86.cubin | grep HMMA)\n\n");

    // =========================================================
    // Allocate and initialize host data
    // =========================================================
    size_t elems_a = (size_t)M * K;
    size_t elems_b = (size_t)K * N;
    size_t elems_c = (size_t)M * N;
    int K_stored = K / 2;
    size_t elems_a_compressed = (size_t)M * K_stored;

    float *host_a = (float*)malloc(elems_a * sizeof(float));
    float *host_b = (float*)malloc(elems_b * sizeof(float));
    float *host_ref = (float*)malloc(elems_c * sizeof(float));
    float *host_out = (float*)malloc(elems_c * sizeof(float));
    __half *host_a_compressed = (__half*)malloc(elems_a_compressed * sizeof(__half));
    __half *host_b_fp16 = (__half*)malloc(elems_b * sizeof(__half));

    // Fill A and B with random FP32
    fill_random(host_a, elems_a, 42);
    fill_random(host_b, elems_b, 137);

    // Compress A to 2:4 format (keep positions 0,1 per group)
    compress_2_4(host_a, host_a_compressed, M, K);

    // Convert B to FP16
    for (size_t i = 0; i < elems_b; i++)
        host_b_fp16[i] = __float2half(host_b[i]);

    // CPU reference (sparse multiply)
    printf("Computing CPU reference (sparse GEMM)...\n");
    cpu_sparse_gemm(host_a, host_b, host_ref, M, N, K);
    printf("Done.\n\n");

    // =========================================================
    // GPU: allocate and copy
    // =========================================================
    CUdeviceptr d_a_compressed, d_b, d_c;
    CHECK_CU(cuMemAlloc(&d_a_compressed, elems_a_compressed * sizeof(__half)));
    CHECK_CU(cuMemAlloc(&d_b, elems_b * sizeof(__half)));
    CHECK_CU(cuMemAlloc(&d_c, elems_c * sizeof(float)));

    CHECK_CU(cuMemcpyHtoD(d_a_compressed, host_a_compressed,
                           elems_a_compressed * sizeof(__half)));
    CHECK_CU(cuMemcpyHtoD(d_b, host_b_fp16, elems_b * sizeof(__half)));

    // =========================================================
    // Correctness test
    // =========================================================
    printf("Correctness test:\n");
    CHECK_CU(cuMemsetD32(d_c, 0, elems_c));

    void *args[] = { &d_a_compressed, &d_b, &d_c, &M, &N, &K };
    int grid_x = N / 16;
    int grid_y = M / 16;

    CHECK_CU(cuLaunchKernel(fn_sparse,
        grid_x, grid_y, 1,
        32, 1, 1,
        0, NULL, args, NULL));
    CHECK_CU(cuCtxSynchronize());
    CHECK_CU(cuMemcpyDtoH(host_out, d_c, elems_c * sizeof(float)));

    // FP16 accumulation: use relaxed tolerance
    auto result = check_fp32(host_out, host_ref, elems_c, 1e-1f, 1e-1f);
    print_check_result("hgemm_sparse_naive (mma.sp, FP16)", result);

    if (result.num_errors > 0 && result.num_errors <= 20) {
        printf("\n  First errors:\n");
        int shown = 0;
        for (size_t i = 0; i < elems_c && shown < 10; i++) {
            float abs_err = fabsf(host_out[i] - host_ref[i]);
            float ref_abs = fabsf(host_ref[i]);
            float rel_err = (ref_abs > 1e-8f) ? (abs_err / ref_abs) : abs_err;
            if (abs_err > 1e-1f && rel_err > 1e-1f) {
                int r = i / N, c = i % N;
                printf("    [%zu] row=%d col=%d GPU=%.4f REF=%.4f abs=%.2e\n",
                       i, r, c, host_out[i], host_ref[i], abs_err);
                shown++;
            }
        }
    }
    printf("\n");

    // =========================================================
    // Performance benchmark (if M >= 512)
    // =========================================================
    if (M >= 512) {
        printf("Performance (M=%d, N=%d, K=%d):\n", M, N, K);

        int warmup = 5, bench_n = 50;

        // GFLOPS: count only the non-zero operations (K/2 effective mults per output element)
        double effective_flops = 2.0 * M * N * (K / 2);  // sparse: half the K dimension
        // Also report "equivalent dense GFLOPS" for comparison
        double dense_equiv_flops = 2.0 * M * N * K;

        for (int i = 0; i < warmup; i++) {
            CHECK_CU(cuLaunchKernel(fn_sparse,
                grid_x, grid_y, 1, 32, 1, 1, 0, NULL, args, NULL));
        }
        CHECK_CU(cuCtxSynchronize());

        float avg_ms;
        {
            BenchTimer timer;
            timer.start();
            for (int i = 0; i < bench_n; i++) {
                CHECK_CU(cuLaunchKernel(fn_sparse,
                    grid_x, grid_y, 1, 32, 1, 1, 0, NULL, args, NULL));
            }
            avg_ms = timer.stop_ms() / bench_n;
        }

        double sparse_gflops = effective_flops / (avg_ms / 1000.0) / 1e9;
        double equiv_gflops = dense_equiv_flops / (avg_ms / 1000.0) / 1e9;
        printf("  %-40s %7.3f ms  %8.0f GFLOPS (effective)\n",
               "hgemm_sparse_naive", avg_ms, sparse_gflops);
        printf("  %-40s %7s     %8.0f GFLOPS (dense-equivalent)\n",
               "", "", equiv_gflops);
        printf("\n  Dense HGEMM baseline (32,197 GFLOPS at 4096^3) for comparison.\n");
        printf("  This naive kernel has no smem tiling — expect low utilization.\n");
    }

    // Cleanup
    cuMemFree(d_a_compressed);
    cuMemFree(d_b);
    cuMemFree(d_c);
    cuModuleUnload(mod_sparse);
    cuCtxDestroy(ctx);

    free(host_a); free(host_b); free(host_ref); free(host_out);
    free(host_a_compressed); free(host_b_fp16);

    return 0;
}
