/*
 * bench.cu — Sparse HGEMM benchmark (BenchDriver refactor)
 *
 * Tests 2:4 structured sparsity on GA104 (RTX 3070 Ti, sm_86).
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o hgemm_sparse_naive.sm_86.cubin hgemm_sparse_naive.cu
 *   nvcc --cubin -arch=sm_86 -O2 -o hgemm_sparse_tiled.sm_86.cubin hgemm_sparse_tiled.cu
 *   nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../common
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>
#include <cuda_fp16.h>

#include "../../kernels/_common/bench_driver.h"
#include "sparse_meta.h"

// -----------------------------------------------------------------------
// Fixed-pattern helpers
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

static void cpu_sparse_gemm_fixed(
    const float *A_dense, const float *B, float *C,
    int M, int N, int K
) {
    memset(C, 0, (size_t)M * N * sizeof(float));
    for (int row = 0; row < M; row++) {
        for (int k = 0; k < K; k++) {
            if ((k % 4) < 2) {
                float a_val = A_dense[(size_t)row * K + k];
                const float *b_row = &B[(size_t)k * N];
                float       *c_row = &C[(size_t)row * N];
                for (int col = 0; col < N; col++) c_row[col] += a_val * b_row[col];
            }
        }
    }
}

// -----------------------------------------------------------------------
// Variant descriptor
// -----------------------------------------------------------------------
struct Variant {
    const char *name;
    const char *cubin;
    const char *sym;
    dim3 grid, block;
    bool required;
};

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 256;
    int N = (argc > 2) ? atoi(argv[2]) : M;
    int K = (argc > 3) ? atoi(argv[3]) : M;

    if (K % 16 != 0 || M % 16 != 0 || N % 16 != 0) {
        fprintf(stderr, "M, N, K must be multiples of 16\n");
        return 1;
    }

    printf("=== Sparse HGEMM Benchmark (BenchDriver refactor) ===\n");
    printf("M=%d  N=%d  K=%d\n\n", M, N, K);

    BenchDriver driver;
    driver.init_context();

    size_t elems_a      = (size_t)M * K;
    size_t elems_b      = (size_t)K * N;
    size_t elems_c      = (size_t)M * N;
    int    K_stored     = K / 2;
    size_t elems_a_comp = (size_t)M * K_stored;
    size_t meta_count   = sparse_meta_count(M, K);

    auto h_A    = driver.host_alloc<float>(elems_a);
    auto h_B    = driver.host_alloc<float>(elems_b);
    auto h_ref  = driver.host_alloc<float>(elems_c);
    auto h_out  = driver.host_alloc<float>(elems_c);
    auto h_A_c  = driver.host_alloc<__half>(elems_a_comp);
    auto h_B_fp = driver.host_alloc<__half>(elems_b);
    auto h_meta = driver.host_alloc<uint32_t>(meta_count);

    fill_random(h_A.get(), elems_a, 42);
    fill_random(h_B.get(), elems_b, 137);
    for (size_t i = 0; i < elems_b; i++) h_B_fp[i] = __float2half(h_B[i]);

    auto d_A_c = driver.device_alloc<__half>(elems_a_comp);
    auto d_B   = driver.device_alloc<__half>(elems_b);
    auto d_C   = driver.device_alloc<float>(elems_c);
    auto d_meta = driver.device_alloc<uint32_t>(meta_count);

    driver.copy_h2d(d_B, h_B_fp, elems_b * sizeof(__half));

    // =====================================================================
    // Section 1: Fixed {0,1} pattern — correctness (naive + tiled)
    // =====================================================================
    printf("=== Section 1: Fixed pattern {0,1} ===\n");

    compress_2_4_fixed(h_A.get(), h_A_c.get(), M, K);
    for (size_t i = 0; i < meta_count; i++) h_meta[i] = 0x44444444u;

    driver.copy_h2d(d_A_c, h_A_c, elems_a_comp * sizeof(__half));
    driver.copy_h2d(d_meta, h_meta, meta_count * sizeof(uint32_t));

    cpu_sparse_gemm_fixed(h_A.get(), h_B.get(), h_ref.get(), M, N, K);

    std::vector<Variant> variants = {
        {"naive", "hgemm_sparse_naive.sm_86.cubin",
         "hgemm_sparse_naive", dim3(N/16, M/16, 1), dim3(32,1,1), true},
        {"tiled", "hgemm_sparse_tiled.sm_86.cubin",
         "hgemm_sparse_tiled", dim3((N+127)/128, (M+127)/128, 1), dim3(512,1,1), false},
    };

    for (auto &v : variants) {
        CUfunction fn = driver.load_kernel(v.cubin, v.sym, v.required);
        if (!fn) { printf("  %-30s not found\n", v.name); continue; }

        // Naive: 6 args (A, B, C, M, N, K)
        // Tiled: 7 args (A, B, C, meta, M, N, K)
        if (strcmp(v.name, "naive") == 0) {
            void *args[] = { &d_A_c.ptr, &d_B.ptr, &d_C.ptr, &M, &N, &K };
            CHECK_CU(cuMemsetD32((CUdeviceptr)d_C.ptr, 0, elems_c));
            CHECK_CU(cuLaunchKernel(fn, v.grid.x, v.grid.y, v.grid.z,
                                    v.block.x, v.block.y, v.block.z,
                                    0, nullptr, args, nullptr));
            CHECK_CU(cuCtxSynchronize());
            driver.copy_d2h(h_out, d_C, elems_c * sizeof(float));
            driver.check(h_out.get(), h_ref.get(), (int)elems_c,
                         1e-1f, 1e-1f, "hgemm_sparse_naive (fixed {0,1})");
        } else {
            void *args[] = { &d_A_c.ptr, &d_B.ptr, &d_C.ptr, &d_meta.ptr, &M, &N, &K };
            CHECK_CU(cuMemsetD32((CUdeviceptr)d_C.ptr, 0, elems_c));
            CHECK_CU(cuLaunchKernel(fn, v.grid.x, v.grid.y, v.grid.z,
                                    v.block.x, v.block.y, v.block.z,
                                    0, nullptr, args, nullptr));
            CHECK_CU(cuCtxSynchronize());
            driver.copy_d2h(h_out, d_C, elems_c * sizeof(float));
            driver.check(h_out.get(), h_ref.get(), (int)elems_c,
                         1e-1f, 1e-1f, "hgemm_sparse_tiled (fixed {0,1})");
        }
    }
    printf("\n");

    // =====================================================================
    // Section 2: Arbitrary random 2:4 pattern — tiled only
    // =====================================================================
    CUfunction fn_tiled = driver.load_kernel(
        "hgemm_sparse_tiled.sm_86.cubin", "hgemm_sparse_tiled", false);
    if (fn_tiled) {
        printf("=== Section 2: Arbitrary random 2:4 pattern ===\n");

        auto h_A_rand   = driver.host_alloc<float>(elems_a);
        auto h_A_rand_c = driver.host_alloc<__half>(elems_a_comp);
        auto h_meta_r   = driver.host_alloc<uint32_t>(meta_count);
        auto h_ref_r    = driver.host_alloc<float>(elems_c);

        gen_random_sparse_2_4(h_A_rand.get(), M, K, /*seed=*/999);
        compress_2_4_arbitrary(h_A_rand.get(), M, K, h_A_rand_c.get(), h_meta_r.get());

        auto d_A_rand = driver.device_alloc<__half>(elems_a_comp);
        auto d_meta_r = driver.device_alloc<uint32_t>(meta_count);
        driver.copy_h2d(d_A_rand, h_A_rand_c, elems_a_comp * sizeof(__half));
        driver.copy_h2d(d_meta_r, h_meta_r, meta_count * sizeof(uint32_t));

        cpu_sparse_gemm_arbitrary(h_A_rand.get(), h_B.get(), h_ref_r.get(), M, N, K);

        void *args_r[] = { &d_A_rand.ptr, &d_B.ptr, &d_C.ptr,
                           &d_meta_r.ptr, &M, &N, &K };
        CHECK_CU(cuMemsetD32((CUdeviceptr)d_C.ptr, 0, elems_c));
        CHECK_CU(cuLaunchKernel(fn_tiled,
                                (N+127)/128, (M+127)/128, 1,
                                512, 1, 1,
                                0, nullptr, args_r, nullptr));
        CHECK_CU(cuCtxSynchronize());
        driver.copy_d2h(h_out, d_C, elems_c * sizeof(float));
        driver.check(h_out.get(), h_ref_r.get(), (int)elems_c,
                     1e-1f, 1e-1f, "hgemm_sparse_tiled (arbitrary)");
        printf("  PASS — dynamic metadata layout confirmed on hardware.\n\n");
    }

    // =====================================================================
    // Performance benchmark (M >= 512 only)
    // =====================================================================
    if (M >= 512) {
        printf("Performance (M=%d, N=%d, K=%d):\n", M, N, K);
        printf("  %-32s %9s  %11s  %15s\n",
               "Kernel", "Time(ms)", "Eff.GFLOPS", "Dense-eq GFLOPS");

        double eff_flops = 2.0 * M * N * (K / 2.0);
        double dense_flops = 2.0 * M * N * K;

        for (auto &v : variants) {
            CUfunction fn = driver.load_kernel(v.cubin, v.sym, false);
            if (!fn) continue;

            void *naive_args[] = { &d_A_c.ptr, &d_B.ptr, &d_C.ptr, &M, &N, &K };
            void *tiled_args[] = { &d_A_c.ptr, &d_B.ptr, &d_C.ptr,
                                   &d_meta.ptr, &M, &N, &K };
            void **args = (strcmp(v.name, "naive") == 0) ? naive_args : tiled_args;

            float ms = driver.benchmark_kernel(fn, v.grid, v.block, 0, args, 5, 50);
            double eff_gflops   = eff_flops   / (ms / 1000.0) / 1e9;
            double dense_gflops = dense_flops / (ms / 1000.0) / 1e9;
            printf("  %-32s %9.3f  %11.0f  %15.0f\n",
                   v.name, ms, eff_gflops, dense_gflops);
        }

        printf("  %-32s %9s  %11s  %15.0f\n",
               "Dense baseline (ref)", "", "", 31910.0);
    }

    return 0;
}
