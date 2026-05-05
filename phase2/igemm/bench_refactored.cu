/*
 * bench_refactored.cu — INT8 IGEMM benchmark using BenchDriver
 *
 * Demonstrates bench_driver.h with INT8 quantization and dequantization.
 * Before: ~180 lines
 * After:  ~75 lines
 *
 * Build:
 *   nvcc -arch=sm_86 -O2 -o bench_refactored bench_refactored.cu -lcuda -I../common
 */

#include <cuda.h>
#include <cstdint>
#include "../common/bench_driver.h"

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 2048;
    int N = (argc > 2) ? atoi(argv[2]) : 2048;
    int K = (argc > 3) ? atoi(argv[3]) : 2048;

    printf("=== IGEMM Benchmark (BenchDriver refactor) ===\n");
    printf("C[%dx%d] = A[%dx%d] * B[%dx%d] (INT8)\n\n", M, N, M, K, K, N);

    BenchDriver driver;
    driver.init_context();

    size_t a_elems = (size_t)M * K, b_elems = (size_t)K * N, c_elems = (size_t)M * N;

    auto d_A = driver.device_alloc<int8_t>(a_elems);
    auto d_B = driver.device_alloc<int8_t>(b_elems);
    auto d_C = driver.device_alloc<float>(c_elems);

    auto h_A = driver.host_alloc<int8_t>(a_elems);
    auto h_B = driver.host_alloc<int8_t>(b_elems);
    auto h_C = driver.host_alloc<float>(c_elems);
    auto h_ref = driver.host_alloc<float>(c_elems);

    // Simple symmetric quantization
    for (size_t i = 0; i < a_elems; i++) h_A[i] = (int8_t)((rand() % 255) - 127);
    for (size_t i = 0; i < b_elems; i++) h_B[i] = (int8_t)((rand() % 255) - 127);

    driver.copy_h2d(d_A, h_A, a_elems);
    driver.copy_h2d(d_B, h_B, b_elems);

    // CPU FP32 reference from INT8 inputs (dequantize first)
    bool have_ref = (M <= 512);
    if (have_ref) {
        auto h_Af = driver.host_alloc<float>(a_elems);
        auto h_Bf = driver.host_alloc<float>(b_elems);
        for (size_t i = 0; i < a_elems; i++) h_Af[i] = (float)h_A[i];
        for (size_t i = 0; i < b_elems; i++) h_Bf[i] = (float)h_B[i];
        cpu_sgemm(M, N, K, 1.0f, h_Af.get(), K, h_Bf.get(), N, 0.0f, h_ref.get(), N);
    }

    float scale_a = 1.0f, scale_b = 1.0f;

    struct V { const char *name, *cubin, *sym; dim3 g, b; unsigned smem; };
    std::vector<V> variants = {
        {"igemm_pipelined_cpasync", "igemm_pipelined_cpasync.sm_86.cubin",
         "igemm_pipelined_cpasync", dim3((N+63)/64,(M+63)/64,1), dim3(128,1,1), 0},
    };

    for (auto &v : variants) {
        CUfunction fn = driver.load_kernel(v.cubin, v.sym, false);
        if (!fn) { printf("  %-20s not found\n", v.name); continue; }

        void *args[] = { &d_A, &d_B, &d_C, &M, &N, &K, &scale_a, &scale_b };
        float ms = driver.benchmark_kernel(fn, v.g, v.b, v.smem, args);
        double tops = 2.0 * M * N * K / (ms / 1000.0) / 1e12;

        if (have_ref) {
            driver.copy_d2h(h_C, d_C, c_elems * sizeof(float));
            driver.check(h_C.get(), h_ref.get(), (int)c_elems, 0.5f, 0.1f, v.name);
        }
        printf("  %-20s %7.3f ms  %8.2f TOPS  [%s]\n",
               v.name, ms, tops, have_ref ? "CHECKED" : "PERF_ONLY");
    }
    return 0;
}
