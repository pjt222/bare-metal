/*
 * bench.cu — Benchmark: sparse HGEMM (mma.sp) naive vs tiled
 *
 * Tests 2:4 structured sparsity on GA104 (RTX 3070 Ti, sm_86).
 *
 * Two test sections:
 *   1. Fixed pattern {0,1}: tests naive + tiled kernels (baseline, all prior results)
 *   2. Arbitrary random 2:4 patterns: tests tiled kernel with dynamic metadata
 *      (see sparse_meta.h for metadata format and constraint documentation)
 *
 * Kernels:
 *   - hgemm_sparse_naive:  1 warp, no shared memory tiling, hardcoded metadata
 *   - hgemm_sparse_tiled:  16 warps, 128×128 tiles, dynamic metadata parameter
 *
 * Build:
 *   # In phase2/hgemm_sparse/
 *   nvcc --cubin -arch=sm_86 -O2 -o hgemm_sparse_naive.sm_86.cubin hgemm_sparse_naive.cu
 *   nvcc --cubin -arch=sm_86 -O2 -o hgemm_sparse_tiled.sm_86.cubin hgemm_sparse_tiled.cu
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
#include "sparse_meta.h"

// -----------------------------------------------------------------------
// compress_2_4_fixed: fixed-pattern compression (positions {0,1} always)
// Used for the naive kernel correctness/perf tests (backward compat).
// -----------------------------------------------------------------------
static void compress_2_4_fixed(
    const float *A_dense, __half *A_compressed,
    int M, int K
) {
    int K_stored = K / 2;
    for (int row = 0; row < M; row++) {
        int stored_col = 0;
        for (int k = 0; k < K; k += 4) {
            A_compressed[row * K_stored + stored_col + 0] =
                __float2half(A_dense[row * K + k + 0]);
            A_compressed[row * K_stored + stored_col + 1] =
                __float2half(A_dense[row * K + k + 1]);
            stored_col += 2;
        }
    }
}

// -----------------------------------------------------------------------
// cpu_sparse_gemm_fixed: CPU reference for fixed {0,1} pattern
// Loop order: row × k × col — cache-friendly (B row, C row both contiguous)
// -----------------------------------------------------------------------
static void cpu_sparse_gemm_fixed(
    const float *A_dense,
    const float *B,
    float *C,
    int M, int N, int K
) {
    memset(C, 0, (size_t)M * N * sizeof(float));
    for (int row = 0; row < M; row++) {
        for (int k = 0; k < K; k++) {
            if ((k % 4) < 2) {
                float a_val = A_dense[(size_t)row * K + k];
                const float *b_row = &B[(size_t)k * N];
                float       *c_row = &C[(size_t)row * N];
                for (int col = 0; col < N; col++) {
                    c_row[col] += a_val * b_row[col];
                }
            }
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

    // =========================================================
    // Load naive cubin
    // =========================================================
    CUmodule mod_naive;
    CUfunction fn_naive;
    if (cuModuleLoad(&mod_naive, "hgemm_sparse_naive.sm_86.cubin") != CUDA_SUCCESS) {
        fprintf(stderr, "Cannot load hgemm_sparse_naive.sm_86.cubin. Build first.\n");
        return 1;
    }
    CHECK_CU(cuModuleGetFunction(&fn_naive, mod_naive, "hgemm_sparse_naive"));

    int naive_regs = 0;
    cuFuncGetAttribute(&naive_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn_naive);
    printf("Naive  kernel registers: %d\n", naive_regs);

    // =========================================================
    // Load tiled cubin (optional — skip if not built yet)
    // =========================================================
    CUmodule mod_tiled;
    CUfunction fn_tiled;
    bool have_tiled = false;
    int tiled_regs = 0;

    CUresult tiled_load_result = cuModuleLoad(&mod_tiled, "hgemm_sparse_tiled.sm_86.cubin");
    if (tiled_load_result == CUDA_SUCCESS) {
        CHECK_CU(cuModuleGetFunction(&fn_tiled, mod_tiled, "hgemm_sparse_tiled"));
        cuFuncGetAttribute(&tiled_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn_tiled);
        printf("Tiled  kernel registers: %d\n", tiled_regs);
        have_tiled = true;
    } else {
        printf("WARNING: hgemm_sparse_tiled.sm_86.cubin not found — skipping tiled kernel.\n");
    }

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

    // Section 1: fixed {0,1} pattern (naive + tiled baseline)
    compress_2_4_fixed(host_a, host_a_compressed, M, K);

    // Convert B to FP16
    for (size_t i = 0; i < elems_b; i++)
        host_b_fp16[i] = __float2half(host_b[i]);

    // CPU reference for fixed pattern
    printf("Computing CPU reference (fixed {0,1} pattern)...\n");
    cpu_sparse_gemm_fixed(host_a, host_b, host_ref, M, N, K);
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

    // Fixed-pattern metadata: all 0x44444444 (positions {0,1})
    // Used for naive kernel (which has hardcoded meta) and as tiled baseline.
    size_t meta_count = sparse_meta_count(M, K);
    uint32_t *host_meta_fixed = (uint32_t*)malloc(meta_count * sizeof(uint32_t));
    for (size_t i = 0; i < meta_count; i++) host_meta_fixed[i] = 0x44444444u;

    CUdeviceptr d_meta;
    CHECK_CU(cuMemAlloc(&d_meta, meta_count * sizeof(uint32_t)));
    CHECK_CU(cuMemcpyHtoD(d_meta, host_meta_fixed,
                           meta_count * sizeof(uint32_t)));

    // =========================================================
    // Section 1: Fixed {0,1} pattern — correctness tests
    // =========================================================
    printf("=== Section 1: Fixed pattern {0,1} ===\n");
    printf("Correctness tests:\n");
    CHECK_CU(cuMemsetD32(d_c, 0, elems_c));

    void *args_naive[] = { &d_a_compressed, &d_b, &d_c, &M, &N, &K };
    int naive_grid_x = N / 16;
    int naive_grid_y = M / 16;

    CHECK_CU(cuLaunchKernel(fn_naive,
        naive_grid_x, naive_grid_y, 1,
        32, 1, 1,
        0, NULL, args_naive, NULL));
    CHECK_CU(cuCtxSynchronize());
    CHECK_CU(cuMemcpyDtoH(host_out, d_c, elems_c * sizeof(float)));

    auto naive_result = check_fp32(host_out, host_ref, elems_c, 1e-1f, 1e-1f);
    print_check_result("hgemm_sparse_naive (mma.sp, FP16, fixed {0,1})", naive_result);

    if (naive_result.num_errors > 0 && naive_result.num_errors <= 20) {
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

    if (have_tiled) {
        CHECK_CU(cuMemsetD32(d_c, 0, elems_c));

        int tiled_grid_x = (N + 127) / 128;
        int tiled_grid_y = (M + 127) / 128;
        // Tiled kernel now takes metadata as 4th arg; fixed meta = all 0x44444444
        void *args_tiled[] = { &d_a_compressed, &d_b, &d_c, &d_meta, &M, &N, &K };

        CHECK_CU(cuLaunchKernel(fn_tiled,
            tiled_grid_x, tiled_grid_y, 1,
            512, 1, 1,
            0, NULL, args_tiled, NULL));
        CHECK_CU(cuCtxSynchronize());
        CHECK_CU(cuMemcpyDtoH(host_out, d_c, elems_c * sizeof(float)));

        auto tiled_result = check_fp32(host_out, host_ref, elems_c, 1e-1f, 1e-1f);
        print_check_result("hgemm_sparse_tiled (mma.sp, FP16, fixed {0,1})", tiled_result);

        if (tiled_result.num_errors > 0 && tiled_result.num_errors <= 20) {
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
    }
    printf("\n");

    // =========================================================
    // Section 2: Arbitrary random 2:4 pattern — tiled kernel only
    //
    // Uses gen_random_sparse_2_4 + compress_2_4_arbitrary from sparse_meta.h.
    // The naive kernel is NOT tested here — it has hardcoded 0x44444444 and
    // will produce wrong results for non-{0,1} patterns.
    //
    // Validates the dynamic metadata implementation (issue #34).
    // IMPORTANT: metadata bit layout is derived from first principles; this
    // test provides the first hardware validation of arbitrary-pattern support.
    // =========================================================
    if (have_tiled) {
        printf("=== Section 2: Arbitrary random 2:4 pattern (tiled kernel) ===\n");

        // Generate random sparse A (rows within gid-pair share same pattern)
        float    *host_a_rand   = (float*)malloc(elems_a * sizeof(float));
        __half   *host_a_rand_c = (__half*)malloc(elems_a_compressed * sizeof(__half));
        uint32_t *host_meta_rand = (uint32_t*)malloc(meta_count * sizeof(uint32_t));
        float    *host_ref_rand  = (float*)malloc(elems_c * sizeof(float));

        gen_random_sparse_2_4(host_a_rand, M, K, /*seed=*/999);
        compress_2_4_arbitrary(host_a_rand, M, K, host_a_rand_c, host_meta_rand);

        // CPU reference using the actual zero pattern in A_rand
        printf("Computing CPU reference (arbitrary pattern)...\n");
        cpu_sparse_gemm_arbitrary(host_a_rand, host_b, host_ref_rand, M, N, K);
        printf("Done.\n\n");

        // Upload random compressed A and metadata
        CUdeviceptr d_a_rand, d_meta_rand;
        CHECK_CU(cuMemAlloc(&d_a_rand,    elems_a_compressed * sizeof(__half)));
        CHECK_CU(cuMemAlloc(&d_meta_rand, meta_count          * sizeof(uint32_t)));
        CHECK_CU(cuMemcpyHtoD(d_a_rand,    host_a_rand_c,
                               elems_a_compressed * sizeof(__half)));
        CHECK_CU(cuMemcpyHtoD(d_meta_rand, host_meta_rand,
                               meta_count * sizeof(uint32_t)));

        CHECK_CU(cuMemsetD32(d_c, 0, elems_c));
        int tiled_grid_x = (N + 127) / 128;
        int tiled_grid_y = (M + 127) / 128;
        void *args_rand[] = { &d_a_rand, &d_b, &d_c, &d_meta_rand, &M, &N, &K };

        CHECK_CU(cuLaunchKernel(fn_tiled,
            tiled_grid_x, tiled_grid_y, 1,
            512, 1, 1,
            0, NULL, args_rand, NULL));
        CHECK_CU(cuCtxSynchronize());
        CHECK_CU(cuMemcpyDtoH(host_out, d_c, elems_c * sizeof(float)));

        auto rand_result = check_fp32(host_out, host_ref_rand, elems_c, 1e-1f, 1e-1f);
        print_check_result("hgemm_sparse_tiled (mma.sp, FP16, arbitrary pattern)", rand_result);

        if (rand_result.num_errors > 0 && rand_result.num_errors <= 20) {
            printf("\n  First errors:\n");
            int shown = 0;
            for (size_t i = 0; i < elems_c && shown < 10; i++) {
                float abs_err = fabsf(host_out[i] - host_ref_rand[i]);
                float ref_abs = fabsf(host_ref_rand[i]);
                float rel_err = (ref_abs > 1e-8f) ? (abs_err / ref_abs) : abs_err;
                if (abs_err > 1e-1f && rel_err > 1e-1f) {
                    int r = i / N, c = i % N;
                    printf("    [%zu] row=%d col=%d GPU=%.4f REF=%.4f abs=%.2e\n",
                           i, r, c, host_out[i], host_ref_rand[i], abs_err);
                    shown++;
                }
            }
        } else if (rand_result.num_errors == 0) {
            printf("  PASS — dynamic metadata layout confirmed on hardware.\n");
        }
        printf("\n");

        cuMemFree(d_a_rand);
        cuMemFree(d_meta_rand);
        free(host_a_rand); free(host_a_rand_c);
        free(host_meta_rand); free(host_ref_rand);
    }

    // =========================================================
    // Performance benchmark (if M >= 512)
    // =========================================================
    if (M >= 512) {
        printf("Performance (M=%d, N=%d, K=%d):\n", M, N, K);
        printf("  %-32s %9s  %11s  %15s\n", "Kernel", "Time(ms)", "Eff.GFLOPS", "Dense-eq GFLOPS");
        printf("  %-32s %9s  %11s  %15s\n", "------", "--------", "----------", "---------------");

        int warmup_iters = 5, bench_iters = 50;

        // GFLOPS: count only the non-zero operations (K/2 effective mults per output element)
        double effective_flops = 2.0 * M * N * (K / 2);  // sparse: half the K dimension
        // Also report "equivalent dense GFLOPS" for comparison
        double dense_equiv_flops = 2.0 * M * N * K;

        // --- Benchmark naive kernel ---
        for (int i = 0; i < warmup_iters; i++) {
            CHECK_CU(cuLaunchKernel(fn_naive,
                naive_grid_x, naive_grid_y, 1, 32, 1, 1, 0, NULL, args_naive, NULL));
        }
        CHECK_CU(cuCtxSynchronize());

        float naive_avg_ms;
        {
            BenchTimer timer;
            timer.start();
            for (int i = 0; i < bench_iters; i++) {
                CHECK_CU(cuLaunchKernel(fn_naive,
                    naive_grid_x, naive_grid_y, 1, 32, 1, 1, 0, NULL, args_naive, NULL));
            }
            naive_avg_ms = timer.stop_ms() / bench_iters;
        }

        double naive_eff_gflops = effective_flops / (naive_avg_ms / 1000.0) / 1e9;
        double naive_equiv_gflops = dense_equiv_flops / (naive_avg_ms / 1000.0) / 1e9;
        printf("  %-32s %9.3f  %11.0f  %15.0f\n",
               "hgemm_sparse_naive", naive_avg_ms, naive_eff_gflops, naive_equiv_gflops);

        // --- Benchmark tiled kernel ---
        float tiled_avg_ms = 0.0f;
        double tiled_eff_gflops = 0.0;
        double tiled_equiv_gflops = 0.0;

        if (have_tiled) {
            int tiled_grid_x = (N + 127) / 128;
            int tiled_grid_y = (M + 127) / 128;
            void *args_tiled_bench[] = { &d_a_compressed, &d_b, &d_c,
                                         &d_meta, &M, &N, &K };

            for (int i = 0; i < warmup_iters; i++) {
                CHECK_CU(cuLaunchKernel(fn_tiled,
                    tiled_grid_x, tiled_grid_y, 1, 512, 1, 1,
                    0, NULL, args_tiled_bench, NULL));
            }
            CHECK_CU(cuCtxSynchronize());

            {
                BenchTimer timer;
                timer.start();
                for (int i = 0; i < bench_iters; i++) {
                    CHECK_CU(cuLaunchKernel(fn_tiled,
                        tiled_grid_x, tiled_grid_y, 1, 512, 1, 1,
                        0, NULL, args_tiled_bench, NULL));
                }
                tiled_avg_ms = timer.stop_ms() / bench_iters;
            }

            tiled_eff_gflops = effective_flops / (tiled_avg_ms / 1000.0) / 1e9;
            tiled_equiv_gflops = dense_equiv_flops / (tiled_avg_ms / 1000.0) / 1e9;
            printf("  %-32s %9.3f  %11.0f  %15.0f\n",
                   "hgemm_sparse_tiled", tiled_avg_ms, tiled_eff_gflops, tiled_equiv_gflops);
        }

        // --- Summary ---
        printf("  %-32s %9s  %11s  %15.0f\n",
               "Dense baseline (ref)", "", "", 32197.0);

        if (have_tiled && tiled_avg_ms > 0.0f) {
            printf("\n  Speedup (tiled/naive):          %.1fx\n",
                   naive_avg_ms / tiled_avg_ms);
        }
    }

    // Cleanup
    cuMemFree(d_a_compressed);
    cuMemFree(d_b);
    cuMemFree(d_c);
    cuMemFree(d_meta);
    cuModuleUnload(mod_naive);
    if (have_tiled) cuModuleUnload(mod_tiled);
    cuCtxDestroy(ctx);

    free(host_a); free(host_b); free(host_ref); free(host_out);
    free(host_a_compressed); free(host_b_fp16);
    free(host_meta_fixed);

    return 0;
}
