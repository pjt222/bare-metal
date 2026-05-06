/*
 * bench.cu — SGEMM benchmark (BenchDriver refactor)
 *
 * Build: nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../common
 */

#include <cuda.h>
#include <cstdio>
#include "../common/bench_driver.h"

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int K = (argc > 3) ? atoi(argv[3]) : 1024;

    printf("=== SGEMM Benchmark (BenchDriver refactor) ===\n");
    printf("C[%dx%d] = A[%dx%d] * B[%dx%d]\n\n", M, N, M, K, K, N);

    BenchDriver driver;
    driver.init_context();

    size_t a_bytes = (size_t)M * K * sizeof(float);
    size_t b_bytes = (size_t)K * N * sizeof(float);
    size_t c_bytes = (size_t)M * N * sizeof(float);

    auto d_A = driver.device_alloc<float>(M * K);
    auto d_B = driver.device_alloc<float>(K * N);
    auto d_C = driver.device_alloc<float>(M * N);

    auto h_A = driver.host_alloc<float>(M * K);
    auto h_B = driver.host_alloc<float>(K * N);
    auto h_C = driver.host_alloc<float>(M * N);
    auto h_ref = driver.host_alloc<float>(M * N);

    fill_random(h_A.get(), M * K, 42);
    fill_random(h_B.get(), K * N, 99);
    driver.copy_h2d(d_A, h_A, a_bytes);
    driver.copy_h2d(d_B, h_B, b_bytes);

    // CPU reference (small sizes only)
    bool have_ref = (M <= 512);
    if (have_ref) {
        printf("CPU reference...\n");
        cpu_sgemm(M, N, K, 1.0f, h_A.get(), K, h_B.get(), N, 0.0f, h_ref.get(), N);
    } else {
        printf("CPU reference skipped (M>512)\n");
    }

    struct V {
        const char *name, *cubin, *sym;
        dim3 g, b;
    };
    std::vector<V> variants = {
        {"naive",           "naive.sm_86.cubin",            "sgemm_naive",
         dim3((N+15)/16,(M+15)/16,1), dim3(16,16,1)},
        {"tiled",           "tiled.sm_86.cubin",            "sgemm_tiled",
         dim3((N+31)/32,(M+31)/32,1), dim3(32,32,1)},
        {"register_blocked","register_blocked.sm_86.cubin","sgemm_register_blocked",
         dim3((N+63)/64,(M+63)/64,1), dim3(16,16,1)},
    };

    for (auto &v : variants) {
        CUfunction fn = driver.load_kernel(v.cubin, v.sym, false);
        if (!fn) { printf("  %-30s not found\n", v.name); continue; }

        void *args[] = { &d_A, &d_B, &d_C, &M, &N, &K };
        float ms = driver.benchmark_kernel(fn, v.g, v.b, 0, args);
        double gflops = compute_gflops_gemm(M, N, K, ms);

        if (have_ref) {
            driver.copy_d2h(h_C, d_C, c_bytes);
            driver.check(h_C.get(), h_ref.get(), M * N, 1e-3f, 1e-3f, v.name);
        }
        driver.print_result(v.name, M, N, K, ms, gflops);
    }
    return 0;
}
