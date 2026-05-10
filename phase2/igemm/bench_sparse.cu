/*
 * bench_sparse.cu — Sparse INT8 GEMM benchmark (BenchDriver refactor)
 *
 * Tests igemm_sparse_tiled: 16-warp 128×128, mma.sp.m16n8k32.s32.s8.s8.s32
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o igemm_sparse_tiled.sm_86.cubin igemm_sparse_tiled.cu
 *   nvcc -arch=sm_86 -O2 -o bench_sparse bench_sparse.cu -lcuda -I../common
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>

#include "../common/bench_driver.h"
#include "sparse_meta_int8.h"

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 256;
    int N = (argc > 2) ? atoi(argv[2]) : M;
    int K = (argc > 3) ? atoi(argv[3]) : M;

    if (M % 16 != 0 || N % 16 != 0 || K % 32 != 0) {
        fprintf(stderr, "M, N must be multiples of 16; K must be multiple of 32\n");
        return 1;
    }

    printf("=== Sparse INT8 GEMM Benchmark (BenchDriver refactor) ===\n");
    printf("M=%d  N=%d  K=%d\n\n", M, N, K);

    BenchDriver driver;
    driver.init_context();

    size_t elems_a      = (size_t)M * K;
    size_t elems_b      = (size_t)K * N;
    size_t elems_c      = (size_t)M * N;
    size_t elems_a_comp = (size_t)M * (K / 2);
    size_t meta_count   = sparse_meta_count_int8(M, K);

    auto h_A_dense = driver.host_alloc<int8_t>(elems_a);
    auto h_A_comp  = driver.host_alloc<int8_t>(elems_a_comp);
    auto h_B       = driver.host_alloc<int8_t>(elems_b);
    auto h_meta    = driver.host_alloc<uint32_t>(meta_count);
    auto h_ref_i32 = driver.host_alloc<int32_t>(elems_c);
    auto h_ref     = driver.host_alloc<float>(elems_c);
    auto h_out     = driver.host_alloc<float>(elems_c);

    float scale_a = 1.0f / 127.0f;
    float scale_b = 1.0f / 127.0f;

    gen_random_sparse_2_4_int8(h_A_dense.get(), M, K, /*seed=*/42);
    compress_2_4_int8(h_A_dense.get(), M, K, h_A_comp.get(), h_meta.get());

    srand(137);
    for (size_t i = 0; i < elems_b; i++) {
        int8_t v;
        do { v = (int8_t)((rand() % 255) - 127); } while (v == 0);
        h_B[i] = v;
    }

    cpu_sparse_gemm_int8(h_A_dense.get(), h_B.get(), h_ref_i32.get(), M, N, K);
    float dequant = scale_a * scale_b;
    for (size_t i = 0; i < elems_c; i++) h_ref[i] = (float)h_ref_i32[i] * dequant;

    auto d_A_comp = driver.device_alloc<int8_t>(elems_a_comp);
    auto d_B      = driver.device_alloc<int8_t>(elems_b);
    auto d_C      = driver.device_alloc<float>(elems_c);
    auto d_meta   = driver.device_alloc<uint32_t>(meta_count);

    driver.copy_h2d(d_A_comp, h_A_comp, elems_a_comp * sizeof(int8_t));
    driver.copy_h2d(d_B, h_B, elems_b * sizeof(int8_t));
    driver.copy_h2d(d_meta, h_meta, meta_count * sizeof(uint32_t));

    CUfunction fn = driver.load_kernel("igemm_sparse_tiled.sm_86.cubin",
                                        "igemm_sparse_tiled");

    dim3 grid((N + 127) / 128, (M + 127) / 128, 1);
    dim3 block(512, 1, 1);
    void *args[] = { &d_A_comp.ptr, &d_B.ptr, &d_C.ptr, &d_meta.ptr,
                     &M, &N, &K, &scale_a, &scale_b };

    CHECK_CU(cuMemsetD32((CUdeviceptr)d_C.ptr, 0, elems_c));
    CHECK_CU(cuLaunchKernel(fn, grid.x, grid.y, grid.z,
                            block.x, block.y, block.z,
                            0, nullptr, args, nullptr));
    CHECK_CU(cuCtxSynchronize());
    driver.copy_d2h(h_out, d_C, elems_c * sizeof(float));
    driver.check(h_out.get(), h_ref.get(), (int)elems_c,
                 0.5f, 0.1f, "igemm_sparse_tiled (INT8)");

    if (M >= 512) {
        double eff_flops = 2.0 * M * N * (K / 2.0);
        double dense_flops = 2.0 * M * N * K;

        float ms = driver.benchmark_kernel(fn, grid, block, 0, args, 5, 50);
        double eff_gflops    = eff_flops    / (ms / 1000.0) / 1e9;
        double dense_gflops  = dense_flops  / (ms / 1000.0) / 1e9;
        printf("  %-34s %9.3f ms  %8.0f eff GFLOPS  %8.0f dense-equiv GFLOPS\n",
               "igemm_sparse_tiled", ms, eff_gflops, dense_gflops);
    }
    return 0;
}
